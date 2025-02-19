import os

import torch
from dataprep_pipeline.tokenization import TokenizerPipeline
from transformers import AutoModelForCausalLM
from dotenv import load_dotenv
from accelerate import Accelerator
# Load environment variables from .env file
load_dotenv(dotenv_path=".env")


# Environment configurations
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = "0.0" # Allow more memory allocation
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = "1" # Enable fallback if MPS fails

proxies = {"HTTP_PROXY" : "http://zproxy-global.shell.com:80", "HTTPS_PROXY" : "http://zproxy-global.shell.com:80"}
# Check MPS availability
accelerator = Accelerator()
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"MPS built: {torch.backends.mps.is_built()}")


def infernce(input_text_filepath, model_name):

    tokenization_pipeline = TokenizerPipeline(model_name, input_text_filepath)
    input_text_ids = tokenization_pipeline.run()
    model = AutoModelForCausalLM.from_pretrained(model_name , proxies=proxies)
    generated_tokens_with_prompt = model.generate(input_ids=input_text_ids)

    return generated_tokens_with_prompt


if __name__ == "__main__":
    model_name = os.getenv("MODEL_NAME")
    file_path = os.getenv("FILE_PATH")

    print(infernce(file_path, model_name))
