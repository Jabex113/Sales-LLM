import torch
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from model.transformer import SalesLLM
from tokenizers import Tokenizer


class CLIChatbot:
    
    def __init__(self, model_path: str, tokenizer_path: str):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        
        print(f"Using device: {self.device}")
        print("Loading model...")
        
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        
        checkpoint = torch.load(model_path, map_location=self.device)
        config = checkpoint['config']['model']
        
        self.model = SalesLLM(
            vocab_size=self.tokenizer.get_vocab_size(),
            max_seq_length=config['max_seq_length'],
            embed_dim=config['embedding_dim'],
            num_layers=config['num_layers'],
            num_heads=config['num_heads'],
            ff_dim=config['ff_dim'],
            dropout=0.0
        )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print("Model loaded!\n")
        
        self.conversation_history = []
    
    def generate_response(self, user_input: str) -> str:
        self.conversation_history.append({"role": "user", "content": user_input})
        
        prompt_parts = []
        for msg in self.conversation_history:
            prompt_parts.append(f"<|{msg['role']}|>{msg['content']}")
        prompt_parts.append("<|assistant|>")
        
        prompt = "\n".join(prompt_parts)
        
        encoded = self.tokenizer.encode(prompt)
        input_ids = torch.tensor([encoded.ids], dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids,
                max_new_tokens=150,
                temperature=0.8,
                top_k=50,
                top_p=0.9
            )
        
        generated_text = self.tokenizer.decode(generated_ids[0].tolist())
        
        response_parts = generated_text.split("<|assistant|>")
        if len(response_parts) > 1:
            response = response_parts[-1].split("<|endoftext|>")[0].strip()
        else:
            response = generated_text
        
        self.conversation_history.append({"role": "assistant", "content": response})
        
        return response
    
    def run(self):
        print("=" * 60)
        print("Sales LLM Chat (Stupidest model)")
        print("=" * 60)
        print("Type 'quit' or 'exit' to end the conversation")
        print("Type 'clear' to clear conversation history")
        print("=" * 60)
        print()
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit']:
                    print("Goodbye!")
                    break
                
                if user_input.lower() == 'clear':
                    self.conversation_history = []
                    print("Conversation history cleared.\n")
                    continue
                
                response = self.generate_response(user_input)
                print(f"\nAssistant: {response}\n")
            
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}\n")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="CLI Chat with Sales LLM")
    parser.add_argument("--model", type=str, default="models/checkpoints/best_model.pt", help="Path to model checkpoint")
    parser.add_argument("--tokenizer", type=str, default="models/tokenizer.json", help="Path to tokenizer")
    
    args = parser.parse_args()
    
    chatbot = CLIChatbot(args.model, args.tokenizer)
    chatbot.run()


if __name__ == "__main__":
    main()
