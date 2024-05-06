from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel
from optimum.neuron import pipeline

app = FastAPI()

pipe = pipeline("text-generation", "/mnt/store/llama-3-8b-bf16")

class InputData(BaseModel):
    input_data: str
    stop_token: Optional[str] = None

@app.post("/generate")
async def generate(data: InputData):
    if not hasattr(pipe, "tokenizer") or pipe.tokenizer is None:
        raise ValueError("Tokenizer has not been loaded properly.")
    eos_token_id = pipe.tokenizer.convert_tokens_to_ids(data.stop_token) if data.stop_token else None
    
    outputs = pipe(
        data.input_data,
        max_new_tokens=128,
        temperature=0.01,
        eos_token_id=eos_token_id,
    )
    out_str = outputs[0]["generated_text"][len(data.input_data):].strip() # type: ignore
    n_tokens = len(pipe.tokenizer.encode(out_str))
    return {"generated_text": out_str, "tokens": n_tokens}