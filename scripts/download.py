from huggingface_hub import snapshot_download
 
# compiled model id
compiled_model_id = "aws-neuron/Llama-2-7b-chat-hf-seqlen-2048-bs-2"
 
# save compiled model to local directory
save_directory = "/mnt/store/llama_neuron"
# Downloads our compiled model from the HuggingFace Hub
# using the revision as neuron version reference
# and makes sure we exlcude the symlink files and "hidden" files, like .DS_Store, .gitignore, etc.
snapshot_download(compiled_model_id, revision="2.15.0", local_dir=save_directory, local_dir_use_symlinks=False, allow_patterns=["[!.]*.*"])
 
 
###############################################
# COMMENT IN BELOW TO COMPILE DIFFERENT MODEL #
###############################################
#
# from optimum.neuron import NeuronModelForCausalLM
# from transformers import AutoTokenizer
#
# # model id you want to compile
# vanilla_model_id = "meta-llama/Llama-2-7b-chat-hf"
#
# # configs for compiling model
# compiler_args = {"num_cores": 2, "auto_cast_type": "fp16"}
# input_shapes = {
#   "sequence_length": 2048, # max length to generate
#   "batch_size": 1 # batch size for the model
#   }
#
# llm = NeuronModelForCausalLM.from_pretrained(vanilla_model_id, export=True, **input_shapes, **compiler_args)
# tokenizer = AutoTokenizer.from_pretrained(model_id)
#
# # Save locally or upload to the HuggingFace Hub
# save_directory = "llama_neuron"
# llm.save_pretrained(save_directory)
# tokenizer.save_pretrained(save_directory)