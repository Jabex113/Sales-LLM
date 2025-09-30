import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
from typing import Optional, List
import sys

sys.path.append(str(Path(__file__).parent.parent))

from model.transformer import SalesLLM
from tokenizers import Tokenizer


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    max_tokens: int = 150
    temperature: float = 0.8
    top_k: Optional[int] = 50
    top_p: Optional[float] = 0.9


class ChatResponse(BaseModel):
    response: str
    tokens_generated: int


class SalesLLMInference:
    
    def __init__(self, model_path: str, tokenizer_path: str, device: str = "cpu"):
        self.device = torch.device(device)
        print(f"Loading model on {self.device}...")
        
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        print(f"Tokenizer loaded with vocab size: {self.tokenizer.get_vocab_size()}")
        
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
        
        print("Model loaded successfully!")
    
    def format_prompt(self, messages: List[ChatMessage]) -> str:
        formatted = []
        
        for msg in messages:
            formatted.append(f"<|{msg.role}|>{msg.content}")
        
        formatted.append("<|assistant|>")
        
        return "\n".join(formatted)
    
    def generate_response(
        self,
        messages: List[ChatMessage],
        max_tokens: int = 150,
        temperature: float = 0.8,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 0.9
    ) -> tuple[str, int]:
        
        prompt = self.format_prompt(messages)
        
        encoded = self.tokenizer.encode(prompt)
        input_ids = torch.tensor([encoded.ids], dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )
        
        generated_tokens = generated_ids[0].tolist()
        generated_text = self.tokenizer.decode(generated_tokens)
        
        response_parts = generated_text.split("<|assistant|>")
        if len(response_parts) > 1:
            response = response_parts[-1].split("<|endoftext|>")[0].strip()
        else:
            response = generated_text
        
        tokens_generated = len(generated_ids[0]) - len(input_ids[0])
        
        return response, tokens_generated


app = FastAPI(
    title="Sales LLM API",
    description="AI-powered Sales Lead Assistant",
    version="1.0.0"
)

inference_engine: Optional[SalesLLMInference] = None


@app.on_event("startup")
async def startup_event():
    global inference_engine
    
    model_path = "models/checkpoints/best_model.pt"
    tokenizer_path = "models/tokenizer.json"
    
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    
    try:
        inference_engine = SalesLLMInference(model_path, tokenizer_path, device)
        print("API ready for requests!")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("API will start but requests will fail until model is loaded.")


@app.get("/")
async def root():
    return {
        "status": "running",
        "model_loaded": inference_engine is not None
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        response, tokens_generated = inference_engine.generate_response(
            messages=request.messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p
        )
        
        return ChatResponse(
            response=response,
            tokens_generated=tokens_generated
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation error: {str(e)}")


@app.get("/info")
async def model_info():
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "vocab_size": inference_engine.tokenizer.get_vocab_size(),
        "max_seq_length": inference_engine.model.max_seq_length,
        "num_parameters": inference_engine.model.get_num_params(),
        "device": str(inference_engine.device)
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )
