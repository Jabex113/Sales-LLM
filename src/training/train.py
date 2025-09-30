import os
import json
import yaml
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, Optional
from tqdm import tqdm
import sys

# Disable tokenizers parallelism to avoid fork warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

sys.path.append(str(Path(__file__).parent.parent))

from model.transformer import SalesLLM
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace


class TextDataset(Dataset):
    
    def __init__(self, texts: list, tokenizer: Tokenizer, max_length: int = 2048):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        encoded = self.tokenizer.encode(text)
        tokens = encoded.ids
        
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        target_ids = torch.tensor(tokens[1:], dtype=torch.long)
        
        if len(input_ids) < self.max_length - 1:
            pad_length = (self.max_length - 1) - len(input_ids)
            input_ids = torch.cat([
                input_ids,
                torch.full((pad_length,), self.tokenizer.token_to_id("<|pad|>"), dtype=torch.long)
            ])
            target_ids = torch.cat([
                target_ids,
                torch.full((pad_length,), -1, dtype=torch.long)
            ])
        
        return input_ids, target_ids


class Trainer:
    
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = self._get_device()
        print(f"Using device: {self.device}")
        
        self.tokenizer = self._load_or_train_tokenizer()
        
        self.model = self._initialize_model()
        print(f"Model initialized with {self.model.get_num_params():,} parameters")
        
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        
        self.checkpoint_dir = Path("models/checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.best_val_loss = float('inf')
        self.global_step = 0
    
    def _get_device(self) -> torch.device:
        device_config = self.config['training']['device']
        
        if device_config == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        elif device_config == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    
    def _load_or_train_tokenizer(self) -> Tokenizer:
        tokenizer_path = Path("models/tokenizer.json")
        
        # Always retrain tokenizer if we have new data
        print("Training new tokenizer...")
        tokenizer = Tokenizer(BPE(unk_token="<|unk|>"))
        tokenizer.pre_tokenizer = Whitespace()
        
        special_tokens = self.config['tokenizer']['special_tokens']
        
        trainer = BpeTrainer(
            vocab_size=self.config['tokenizer']['vocab_size'],
            special_tokens=special_tokens,
            show_progress=True
        )
        
        data_path = Path(self.config.get('data_path', 'data/processed/processed_texts.jsonl'))
        
        if not data_path.exists():
            raise FileNotFoundError(f"Processed data not found at {data_path}")
        
        texts = []
        with open(data_path, 'r') as f:
            for line in f:
                doc = json.loads(line)
                texts.append(doc['text'])
        
        tokenizer.train_from_iterator(texts, trainer)
        
        tokenizer.save(str(tokenizer_path))
        print(f"Tokenizer saved to {tokenizer_path}")
        
        return tokenizer
    
    def _initialize_model(self) -> SalesLLM:
        model_config = self.config['model']
        
        model = SalesLLM(
            vocab_size=self.tokenizer.get_vocab_size(),
            max_seq_length=model_config['max_seq_length'],
            embed_dim=model_config['embedding_dim'],
            num_layers=model_config['num_layers'],
            num_heads=model_config['num_heads'],
            ff_dim=model_config['ff_dim'],
            dropout=model_config['dropout']
        )
        
        return model.to(self.device)
    
    def _setup_optimizer(self):
        training_config = self.config['training']
        
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=training_config['learning_rate'],
            weight_decay=training_config['weight_decay']
        )
    
    def _setup_scheduler(self):
        training_config = self.config['training']
        
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=training_config['max_epochs']
        )
    
    def load_data(self):
        data_path = Path(self.config.get('data_path', 'data/processed/processed_texts.jsonl'))
        
        print(f"Loading data from {data_path}")
        texts = []
        with open(data_path, 'r') as f:
            for line in f:
                doc = json.loads(line)
                texts.append(doc['text'])
        
        print(f"Loaded {len(texts)} documents")
        
        train_split = self.config['data']['train_split']
        split_idx = int(len(texts) * train_split)
        
        train_texts = texts[:split_idx]
        val_texts = texts[split_idx:]
        
        train_dataset = TextDataset(train_texts, self.tokenizer, self.config['model']['max_seq_length'])
        val_dataset = TextDataset(val_texts, self.tokenizer, self.config['model']['max_seq_length'])
        
        batch_size = self.config['training']['batch_size']
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True if self.device.type == "cuda" else False,
            persistent_workers=False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True if self.device.type == "cuda" else False,
            persistent_workers=False
        )
        
        return train_loader, val_loader
    
    def train_epoch(self, train_loader, epoch):
        self.model.train()
        total_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, (input_ids, target_ids) in enumerate(progress_bar):
            input_ids = input_ids.to(self.device)
            target_ids = target_ids.to(self.device)
            
            logits, loss = self.model(input_ids, target_ids)
            
            self.optimizer.zero_grad()
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config['training']['max_grad_norm']
            )
            
            self.optimizer.step()
            
            total_loss += loss.item()
            self.global_step += 1
            
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
            
            if self.global_step % self.config['training']['save_every'] == 0:
                self.save_checkpoint(f"checkpoint_step_{self.global_step}.pt")
        
        return total_loss / len(train_loader)
    
    @torch.no_grad()
    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        
        for input_ids, target_ids in tqdm(val_loader, desc="Validating"):
            input_ids = input_ids.to(self.device)
            target_ids = target_ids.to(self.device)
            
            logits, loss = self.model(input_ids, target_ids)
            total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def save_checkpoint(self, filename: str):
        checkpoint_path = self.checkpoint_dir / filename
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'config': self.config
        }, checkpoint_path)
        
        print(f"Checkpoint saved to {checkpoint_path}")
    
    def train(self):
        train_loader, val_loader = self.load_data()
        
        num_epochs = self.config['training']['max_epochs']
        
        print(f"\nStarting training for {num_epochs} epochs...")
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}\n")
        
        for epoch in range(1, num_epochs + 1):
            train_loss = self.train_epoch(train_loader, epoch)
            val_loss = self.validate(val_loader)
            
            print(f"\nEpoch {epoch}/{num_epochs}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint("best_model.pt")
                print("New best model saved!")
            
            self.scheduler.step()
            
            print(f"Learning rate: {self.optimizer.param_groups[0]['lr']:.6f}\n")
        
        print("Training complete!")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Sales LLM")
    parser.add_argument("--config", type=str, default="configs/model_config.yaml", help="Path to config file")
    
    args = parser.parse_args()
    
    trainer = Trainer(args.config)
    trainer.train()


if __name__ == "__main__":
    main()
