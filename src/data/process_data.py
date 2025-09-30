import os
import json
import re
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass

import PyPDF2
import pytesseract
from PIL import Image
from tqdm import tqdm


@dataclass
class TextDocument:
    text: str
    source: str
    metadata: Dict


class DataProcessor:
    
    def __init__(self, raw_data_dir: str, processed_data_dir: str):
        self.raw_data_dir = Path(raw_data_dir)
        self.processed_data_dir = Path(processed_data_dir)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        
    def process_all(self):
        print("Starting data processing pipeline...")
        
        documents = []
        
        chat_docs = self.process_chat_logs()
        documents.extend(chat_docs)
        print(f"Processed {len(chat_docs)} chat logs")
        
        pdf_docs = self.process_pdfs()
        documents.extend(pdf_docs)
        print(f"Processed {len(pdf_docs)} PDFs")
        
        image_docs = self.process_images()
        documents.extend(image_docs)
        print(f"Processed {len(image_docs)} images")
        
        self.save_documents(documents)
        print(f"Total documents processed: {len(documents)}")
        
        return documents
    
    def process_chat_logs(self) -> List[TextDocument]:
        documents = []
        chat_dir = self.raw_data_dir / "chatlogs"
        
        if not chat_dir.exists():
            print(f"Warning: {chat_dir} not found. Skipping chat logs.")
            return documents
        
        for file_path in tqdm(list(chat_dir.glob("**/*")), desc="Processing chat logs"):
            if file_path.is_file():
                if file_path.suffix == ".json":
                    docs = self._process_json_chat(file_path)
                    documents.extend(docs)
                elif file_path.suffix == ".txt":
                    docs = self._process_txt_chat(file_path)
                    documents.extend(docs)
        
        return documents
    
    def _process_json_chat(self, file_path: Path) -> List[TextDocument]:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                formatted_text = self._format_conversation(data)
                return [TextDocument(
                    text=formatted_text,
                    source=str(file_path),
                    metadata={"type": "chat_json", "num_messages": len(data)}
                )]
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
        
        return []
    
    def _process_txt_chat(self, file_path: Path) -> List[TextDocument]:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            text = self._clean_text(text)
            
            if len(text) > 50:
                return [TextDocument(
                    text=text,
                    source=str(file_path),
                    metadata={"type": "chat_txt"}
                )]
        
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
        
        return []
    
    def _format_conversation(self, messages: List[Dict]) -> str:
        formatted = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            formatted.append(f"<|{role}|>{content}")
        
        return "\n".join(formatted) + "<|endoftext|>"
    
    def process_pdfs(self) -> List[TextDocument]:
        documents = []
        pdf_dir = self.raw_data_dir / "pdfs"
        
        if not pdf_dir.exists():
            print(f"Warning: {pdf_dir} not found. Skipping PDFs.")
            return documents
        
        for file_path in tqdm(list(pdf_dir.glob("**/*.pdf")), desc="Processing PDFs"):
            try:
                with open(file_path, 'rb') as f:
                    pdf_reader = PyPDF2.PdfReader(f)
                    text_parts = []
                    
                    for page_num, page in enumerate(pdf_reader.pages):
                        text = page.extract_text()
                        if text:
                            text_parts.append(text)
                    
                    full_text = "\n".join(text_parts)
                    full_text = self._clean_text(full_text)
                    
                    if len(full_text) > 100:
                        documents.append(TextDocument(
                            text=full_text,
                            source=str(file_path),
                            metadata={"type": "pdf", "num_pages": len(pdf_reader.pages)}
                        ))
            
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
        
        return documents
    
    def process_images(self) -> List[TextDocument]:
        documents = []
        image_dir = self.raw_data_dir / "images"
        
        if not image_dir.exists():
            print(f"Warning: {image_dir} not found. Skipping images.")
            return documents
        
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        
        for file_path in tqdm(list(image_dir.glob("**/*")), desc="Processing images (OCR)"):
            if file_path.suffix.lower() in image_extensions:
                try:
                    image = Image.open(file_path)
                    text = pytesseract.image_to_string(image)
                    text = self._clean_text(text)
                    
                    if len(text) > 50:
                        documents.append(TextDocument(
                            text=text,
                            source=str(file_path),
                            metadata={"type": "image_ocr", "size": image.size}
                        ))
                
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
        
        return documents
    
    def _clean_text(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        text = text.strip()
        return text
    
    def save_documents(self, documents: List[TextDocument]):
        output_file = self.processed_data_dir / "processed_texts.jsonl"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for doc in documents:
                json_obj = {
                    "text": doc.text,
                    "source": doc.source,
                    "metadata": doc.metadata
                }
                f.write(json.dumps(json_obj, ensure_ascii=False) + "\n")
        
        print(f"Saved processed documents to {output_file}")
        
        stats = {
            "total_documents": len(documents),
            "total_characters": sum(len(doc.text) for doc in documents),
            "document_types": {}
        }
        
        for doc in documents:
            doc_type = doc.metadata.get("type", "unknown")
            stats["document_types"][doc_type] = stats["document_types"].get(doc_type, 0) + 1
        
        stats_file = self.processed_data_dir / "statistics.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"Statistics saved to {stats_file}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Process data for Sales LLM training")
    parser.add_argument("--raw-dir", type=str, default="data/raw", help="Directory containing raw data")
    parser.add_argument("--output-dir", type=str, default="data/processed", help="Directory to save processed data")
    
    args = parser.parse_args()
    
    processor = DataProcessor(args.raw_dir, args.output_dir)
    processor.process_all()


if __name__ == "__main__":
    main()
