import os
import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(dotenv_path='dataprep_pipeline/data/data.env')

def load_dataset(file_path):
    dataset = pd.read_csv(file_path)
    questions = dataset['PRB'].tolist()
    answers = dataset['SOL'].tolist()
    return {"questions": questions, "answers": answers}

def main():
    file_path = os.getenv('FILE_PATH')
    if not file_path:
        raise ValueError("FILE_PATH environment variable not set")

    finetuning_dataset = load_dataset(file_path)
    finetuning_dataset = Dataset.from_pandas(pd.DataFrame(data=finetuning_dataset))

    # Get the Hugging Face dataset repository from environment variables
    hugging_face_dataset_repo = os.getenv('HUGGING_FACE_DATASET_REPO')
    if not hugging_face_dataset_repo:
        raise ValueError("HUGGING_FACE_DATASET_REPO environment variable not set")

    # Push the dataset to the Hugging Face Hub
    finetuning_dataset.push_to_hub(hugging_face_dataset_repo)

if __name__ == "__main__":
    main()