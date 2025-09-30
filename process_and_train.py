import os
import sys
import json
import re
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict

import PyPDF2
import pytesseract
from PIL import Image
from tqdm import tqdm


@dataclass
class TextDocument:
    text: str
    source: str
    metadata: Dict


class SmartProcessor:
    
    def __init__(self, upload_dir: str = "data/upload"):
        self.upload_dir = Path(upload_dir)
        self.processed_dir = Path("data/processed")
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        if not self.upload_dir.exists():
            self.upload_dir.mkdir(parents=True, exist_ok=True)
            print(f"Created upload directory: {self.upload_dir}")
    
    def process_all_files(self):
        print(f"Scanning {self.upload_dir} for files...")
        
        documents = []
        
        all_files = list(self.upload_dir.glob("**/*"))
        file_list = [f for f in all_files if f.is_file()]
        
        print(f"Found {len(file_list)} files\n")
        
        for file_path in tqdm(file_list, desc="Processing files"):
            ext = file_path.suffix.lower()
            
            if ext == ".pdf":
                docs = self._process_pdf(file_path)
                documents.extend(docs)
            
            elif ext == ".json":
                docs = self._process_json(file_path)
                documents.extend(docs)
            
            elif ext == ".txt":
                docs = self._process_txt(file_path)
                documents.extend(docs)
            
            elif ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]:
                docs = self._process_image(file_path)
                documents.extend(docs)
            
            else:
                print(f"Skipping unsupported file: {file_path.name}")
        
        print(f"\nProcessed {len(documents)} documents")
        
        if len(documents) == 0:
            print("\nWARNING: No documents were processed!")
            print(f"   Please add files to: {self.upload_dir.absolute()}")
            print("   Supported formats: PDF, JSON, TXT, JPG, PNG")
            return False
        
        self._save_documents(documents)
        return True
    
    def _process_pdf(self, file_path: Path) -> List[TextDocument]:
        try:
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                text_parts = []
                
                for page in pdf_reader.pages:
                    text = page.extract_text()
                    if text:
                        text_parts.append(text)
                
                full_text = "\n".join(text_parts)
                full_text = self._clean_text(full_text)
                
                if len(full_text) > 100:
                    return [TextDocument(
                        text=full_text,
                        source=str(file_path.name),
                        metadata={"type": "pdf", "num_pages": len(pdf_reader.pages)}
                    )]
        
        except Exception as e:
            print(f"Error processing PDF {file_path.name}: {e}")
        
        return []
    
    def _process_json(self, file_path: Path) -> List[TextDocument]:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                formatted = []
                for msg in data:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    formatted.append(f"<|{role}|>{content}")
                
                text = "\n".join(formatted) + "<|endoftext|>"
                
                return [TextDocument(
                    text=text,
                    source=str(file_path.name),
                    metadata={"type": "chat_json", "num_messages": len(data)}
                )]
        
        except Exception as e:
            print(f"Error processing JSON {file_path.name}: {e}")
        
        return []
    
    def _process_txt(self, file_path: Path) -> List[TextDocument]:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            text = self._clean_text(text)
            
            if len(text) > 50:
                return [TextDocument(
                    text=text,
                    source=str(file_path.name),
                    metadata={"type": "text"}
                )]
        
        except Exception as e:
            print(f"Error processing TXT {file_path.name}: {e}")
        
        return []
    
    def _process_image(self, file_path: Path) -> List[TextDocument]:
        try:
            image = Image.open(file_path)
            text = pytesseract.image_to_string(image)
            text = self._clean_text(text)
            
            if len(text) > 50:
                return [TextDocument(
                    text=text,
                    source=str(file_path.name),
                    metadata={"type": "image_ocr"}
                )]
        
        except Exception as e:
            print(f"Error processing image {file_path.name}: {e}")
        
        return []
    
    def _clean_text(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        text = text.strip()
        return text
    
    def _save_documents(self, documents: List[TextDocument]):
        output_file = self.processed_dir / "processed_texts.jsonl"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for doc in documents:
                json_obj = {
                    "text": doc.text,
                    "source": doc.source,
                    "metadata": doc.metadata
                }
                f.write(json.dumps(json_obj, ensure_ascii=False) + "\n")
        
        print(f"Saved to: {output_file}")
        
        stats = {
            "total_documents": len(documents),
            "total_characters": sum(len(doc.text) for doc in documents),
            "document_types": {}
        }
        
        for doc in documents:
            doc_type = doc.metadata.get("type", "unknown")
            stats["document_types"][doc_type] = stats["document_types"].get(doc_type, 0) + 1
        
        print(f"\nStatistics:")
        print(f"  Total documents: {stats['total_documents']}")
        print(f"  Total characters: {stats['total_characters']:,}")
        print(f"  By type:")
        for doc_type, count in stats["document_types"].items():
            print(f"    - {doc_type}: {count}")
        
        stats_file = self.processed_dir / "statistics.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)


def train_model():
    print("\n" + "="*60)
    print("Starting Model Training")
    print("="*60 + "\n")
    
    import subprocess
    result = subprocess.run(
        [sys.executable, "src/training/train.py", "--config", "configs/model_config.yaml"],
        cwd=Path(__file__).parent
    )
    
    return result.returncode == 0


def main():
    print("="*60)
    print("Sales LLM - Process & Train")
    print("="*60)
    print()
    
    upload_dir = Path("data/upload")
    
    print(f"Upload your files to: {upload_dir.absolute()}")
    print("   Supported: PDF, JSON, TXT, JPG, PNG")
    print()
    
    if not upload_dir.exists():
        upload_dir.mkdir(parents=True, exist_ok=True)
    
    file_count = len([f for f in upload_dir.glob("**/*") if f.is_file()])
    
    if file_count == 0:
        print("WARNING: No files found in upload directory!")
        print(f"\nPlease add your files to: {upload_dir.absolute()}")
        print("\nThen run this script again:")
        print("  python process_and_train.py")
        return
    
    processor = SmartProcessor(upload_dir="data/upload")
    
    success = processor.process_all_files()
    
    if not success:
        return
    
    print("\n" + "="*60)
    response = input("\nStart training now? (yes/no): ").strip().lower()
    
    if response in ['yes', 'y']:
        train_model()
    else:
        print("\nTo train later, run:")
        print("  python src/training/train.py")


if __name__ == "__main__":
    main()
