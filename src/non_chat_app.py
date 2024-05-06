from fastapi import FastAPI
from optimum.neuron import pipeline

app = FastAPI()

pipe = pipeline("text-generation", "/mnt/store/llama-3-8b-bf16")

@app.post("/generate")
async def generate(input_data: str, stop_token: str = None):
    eos_token_id = pipe.tokenizer.convert_tokens_to_ids(stop_token) if stop_token else None
    outputs = pipe(
        input_data,
        max_new_tokens=128,
        temperature=0.01,
        eos_token_id=eos_token_id,
    )
    
    n_tokens = len(pipe.tokenizer.encode(out_str))
    return {"generated_text": out_str, "tokens": n_tokens}