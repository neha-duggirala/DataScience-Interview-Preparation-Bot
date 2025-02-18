import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from datasets import Dataset

# Load pre-trained model from Hugging Face
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")


def load_dataset(file_path):
    dataset = pd.read_csv(file_path)

    questions = dataset["PRB"].tolist()
    answers = dataset["SOL"].tolist()

    # Combine questions and answers into a single corpus
    corpus = [question + " " + answer for question, answer in zip(questions, answers)]
    return {"qna": corpus}


def tokenize_data(corpus):
    max_len = 170
    tokenizer.pad_token = tokenizer.eos_token
    encoded_texts_longest = tokenizer(
        corpus["qna"], max_length=max_len, truncation=True, padding=True
    )
    return encoded_texts_longest


# Load the dataset and get the corpus
file_path = "dataprep_pipeline/data/dataset_test.csv"
corpus = load_dataset(file_path)
# tokenize_data(corpus)
corpus_dataset = Dataset.from_pandas(pd.DataFrame(data=corpus))

tokenized_corpus = corpus_dataset.map(
    tokenize_data,
    batched=True,
    batch_size=1,
    drop_last_batch=True
)

print(tokenized_corpus)