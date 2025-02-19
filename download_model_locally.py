import os
import torch
from dotenv import load_dotenv
from accelerate import Accelerator
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
# Load environment variables from .env file
load_dotenv(dotenv_path=".env")
# Environment configurations
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = "0.0" # Allow more memory allocation
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = "1" # Enable fallback if MPS fails
# Check MPS availability
accelerator = Accelerator()
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"MPS built: {torch.backends.mps.is_built()}")
MODEL_NAME =  os.getenv("MODEL_NAME")

def download_model(model_name, local_model_dir):
    """
    Download the model if not available locally and save it for future use.
    Parameters:
    - model_name (str): Name of the model on Hugging Face.
    - local_model_dir (str): Directory to save/load the model.
    Returns:
    - model: The loaded model.
    - tokenizer: The loaded tokenizer.
    """
    os.makedirs(local_model_dir, exist_ok=True) # Ensure the directory exists
    # Configuration for 8-bit loading
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    print(f"Downloading model: {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(model_name, proxy = {"HTTP_PROXY" : "http://zproxy-global.shell.com:80", "HTTPS_PROXY" : "http://zproxy-global.shell.com:80"})
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    try:
        print(f"Saving model to {local_model_dir}...")
        model.save_pretrained(local_model_dir)
        tokenizer.save_pretrained(local_model_dir)
        print("Model and tokenizer saved successfully.")
    except Exception as e:
        print(f"Error saving model/tokenizer: {e}")

def main():
    # Example usage
    LOCAL_MODEL_PATH = os.getenv('LOCAL_MODEL_PATH')
    if not LOCAL_MODEL_PATH:
        raise ValueError("Environment variable 'LOCAL_MODEL_PATH' is not set or empty.")
    download_model(MODEL_NAME, LOCAL_MODEL_PATH)

if __name__ == '__main__':
    main()