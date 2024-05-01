import time
import logging
from typing import List, Dict

from optimum.neuron import NeuronModelForCausalLM, pipeline
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def compile_and_save_model(model_id: str, save_directory: str) -> None:
    """
    Load a model from the Hugging Face Hub, compile it into Neuron format, and save the compiled model and tokenizer to a local directory.

    Args:
        model_id (str): The ID of the model to load from the Hugging Face Hub.
        save_directory (str): The directory to save the compiled model and tokenizer.
    """
    logger.info(f"Loading model: {model_id}")
    start_time = time.time()

    compiler_args = {"num_cores": 2, "auto_cast_type": "fp16"}
    input_shapes = {
        "sequence_length": 2048,
        "batch_size": 1
    }

    logger.info("Compiling model into Neuron format...")
    llm = NeuronModelForCausalLM.from_pretrained(model_id, export=True, **input_shapes, **compiler_args)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    logger.info(f"Saving compiled model and tokenizer to: {save_directory}")
    llm.save_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)

    end_time = time.time()
    logger.info(f"Model compilation and saving completed in {end_time - start_time:.2f} seconds")

def load_and_run_inference(compiled_model_directory: str, messages: List[Dict[str, str]], max_tokens: int) -> None:
    """
    Load a compiled model from a local directory and run inference on the provided messages.

    Args:
        compiled_model_directory (str): The directory containing the compiled model and tokenizer.
        messages (List[Dict[str, str]]): The input messages for inference.
        max_tokens (int): The maximum number of tokens to generate.
    """
    logger.info(f"Loading compiled model from: {compiled_model_directory}")
    pipe = pipeline("text-generation", compiled_model_directory)

    inputs = pipe.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    logger.info("Running inference...")
    start_time = time.time()
    outputs = pipe(inputs, max_new_tokens=max_tokens)
    end_time = time.time()

    out_str = outputs[0]["generated_text"][len(inputs):].strip()
    tokens = pipe.tokenizer.encode(out_str)

    logger.info(f"Generated output: {out_str[:50]}...{out_str[-50:]}")
    logger.info(f"Total tokens: {len(tokens)}")
    logger.info(f"Tokens per second: {len(tokens) / (end_time - start_time):.2f}")

def main():
    model_id = "meta-llama/Llama-2-7b-chat-hf"
    compiled_model_directory = "/mnt/store/llama_neuron"
    max_tokens = 100

    compile_and_save_model(model_id, compiled_model_directory)

    messages = [{"role": "user", "content": "Tell me a long story about WW2"}]
    load_and_run_inference(compiled_model_directory, messages, max_tokens)

if __name__ == "__main__":
    main()