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

pipe = pipeline("text-generation", "/mnt/store/llama-3-8b-instruct-bf16")

@app.post("/generate")
async def generate(input_data: ChatInputData):
    inputs = pipe.tokenizer.apply_chat_template(
        input_data.messages, 
        add_generation_prompt=True, 
        tokenize=False
    )
    outputs = pipe(
        inputs, 
        max_new_tokens=128, 
        eos_token_id=pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        temperature=0.01
    )
    
    out_str = outputs[0]["generated_text"][len(inputs):].strip()
    return {"generated_text": out_str, "tokens": outputs[0]["tokens"]}