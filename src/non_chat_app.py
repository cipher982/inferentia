from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel
from optimum.neuron import pipeline

app = FastAPI()

pipe = pipeline("text-generation", "/mnt/store/llama-3-8b-bf16")

class InputData(BaseModel):
    input_data: str
    max_new_tokens: int = 2046
    temperature: float = 0.01
    stop_sequence: Optional[str] = None

@app.post("/generate")
async def generate(data: InputData):
    if not hasattr(pipe, "tokenizer") or pipe.tokenizer is None:
        raise ValueError("Tokenizer has not been loaded properly.")
    
    outputs = pipe(
        data.input_data,
        max_new_tokens=data.max_new_tokens,
        temperature=data.temperature,
        stop_sequence=data.stop_sequence,
    )
    out_str = outputs[0]["generated_text"][len(data.input_data):].strip() # type: ignore
    n_tokens = len(pipe.tokenizer.encode(out_str))
    return {"generated_text": out_str, "tokens": n_tokens}
    