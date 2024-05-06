from fastapi import FastAPI
from optimum.neuron import pipeline
from pydantic import BaseModel
from typing import List

app = FastAPI()

class Message(BaseModel):
    role: str
    content: str

class ChatInputData(BaseModel):
    messages: List[Message]
    max_new_tokens: int = 2046
    temperature: float = 0.01

pipe = pipeline("text-generation", "/mnt/store/llama-3-8b-instruct-bf16")

@app.post("/generate")
async def generate(input_data: ChatInputData) -> dict:
    if not hasattr(pipe, "tokenizer") or pipe.tokenizer is None:
        raise ValueError("Tokenizer has not been loaded properly.")

    messages_dict = [{"role": msg.role, "content": msg.content} for msg in input_data.messages]
    inputs = pipe.tokenizer.apply_chat_template(
        messages_dict, 
        add_generation_prompt=True, 
        tokenize=False
    )
    outputs = pipe(
        inputs,
        max_new_tokens=input_data.max_new_tokens, 
        eos_token_id=pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        temperature=input_data.temperature
    )
    
    out_str = outputs[0]["generated_text"][len(inputs):].strip() # type: ignore
    n_tokens = len(pipe.tokenizer.encode(out_str))
    return {"generated_text": out_str, "tokens": n_tokens}