from fastapi import FastAPI
from optimum.neuron import pipeline

app = FastAPI()

pipe = pipeline("text-generation", "/mnt/store/llama-3-8b-bf16")

@app.post("/generate")
async def generate(input_data: str):
    outputs = pipe(
        input_data,
        max_new_tokens=128,
        temperature=0.01,
        # eos_token_id=pipe.tokenizer.convert_tokens_to_ids("\\n"),
        # eos_token_id=pipe.tokenizer.convert_tokens_to_ids("."),
    )
    
    out_str = outputs[0]["generated_text"].strip()
    return {"generated_text": out_str, "tokens": outputs[0]["tokens"]}