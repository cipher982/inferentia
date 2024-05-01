## Llama-2-7B Neuron Inference
This repository contains code for compiling the Llama-2-7B model from the Hugging Face Hub into AWS Neuron format and running inference on AWS Inferentia inf2 instances.

## Overview
The Llama-2-7B model is a powerful language model developed by Meta AI. By compiling the model into AWS Neuron format and running inference on AWS Inferentia inf2 instances, we can achieve high-performance and cost-effective inference for various natural language processing tasks.
This repository provides a Python script that automates the process of compiling the Llama-2-7B model into Neuron format, saving the compiled model, and running inference on user-provided messages.

## Prerequisites
To use this code, you need the following:
An AWS account with access to Inferentia inf2 instances
Python 3.7 or higher
The following Python packages:
- `optimum-neuron`
- `transformers`

## Setup
1. Set up an AWS Inferentia inf2 instance:
- Launch an EC2 instance with an Inferentia inf2 instance type (e.g., `inf2.8xlarge`). I used `Deep Learning AMI Neuron (Ubuntu 22.04) 20240429` for this example.
- Connect to the instance using SSH.
2. Install the necessary dependencies on the instance. It should have most installed by default, but I like to create a fresh Conda environment and install everything from there:
3. Clone this repository on the instance:
- `git clone https://github.com/cipher982/inferentia.git`

## Usage
The main script for compiling and running inference is `main.py`. It provides two main functions:
1. `compile_and_save_model(model_id, save_directory)`: This function loads the Llama-2-7B model from the Hugging Face Hub, compiles it into Neuron format, and saves the compiled model and tokenizer to the specified directory.
2. ``load_and_run_inference(compiled_model_directory, messages, max_tokens)`: This function loads the compiled model from the specified directory and runs inference on the provided messages, generating output up to the specified maximum number of tokens.

To use the script, update the main function with your desired model ID, compiled model directory, and inference parameters, and then run the script:
```python main.py```

The script will compile the model, save it to the specified directory, and then run inference on the provided messages. The generated output, along with performance metrics, will be logged to the console.
