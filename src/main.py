from fastapi import FastAPI
from optimum.neuron import pipeline
from pydantic import BaseModel
from typing import List, Union

app = FastAPI()

class Message(BaseModel):
    role: str
    content: str

class InputData(BaseModel):
    messages: List[Message] = []
    raw_text: str = ""

model_fp = "/mnt/store/llama-3-8b-bf16"
pipe = pipeline("text-generation", model_fp)

@app.post("/generate")
async def generate(input_data: InputData):
    if input_data.messages:
        inputs = pipe.tokenizer.apply_chat_template(
            input_data.messages, 
            add_generation_prompt=True, 
            tokenize=False
        )
    elif input_data.raw_text:
        inputs = input_data.raw_text
    else:
        return {"error": "Either 'messages' or 'raw_text' must be provided."}

    outputs = pipe(
        inputs, 
        max_new_tokens=128, 
        eos_token_id=pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        temperature=0.01
    )
    out_str = outputs[0]["generated_text"][len(inputs):].strip()
    return {"generated_text": out_str, "tokens": outputs[0]["tokens"]}