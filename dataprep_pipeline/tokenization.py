import pandas as pd
from transformers import AutoTokenizer
from datasets import Dataset

class TokenizerPipeline:
    def __init__(self, model_name, file_path, max_len=170):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.file_path = file_path
        self.max_len = max_len
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def load_dataset(self):
        dataset = pd.read_csv(self.file_path)
        questions = dataset["PRB"].tolist()
        answers = dataset["SOL"].tolist()
        corpus = [question + " " + answer for question, answer in zip(questions, answers)]
        return {"qna": corpus}

    def tokenize_data(self, corpus):
        encoded_texts_longest = self.tokenizer(
            corpus["qna"], max_length=self.max_len, truncation=True, padding=True
        )
        return encoded_texts_longest

    def run(self):
        corpus = self.load_dataset()
        corpus_dataset = Dataset.from_pandas(pd.DataFrame(data=corpus))
        tokenized_corpus = corpus_dataset.map(
            self.tokenize_data,
            batched=True,
            batch_size=1,
            drop_last_batch=True
        )
        return tokenized_corpus

if __name__ == "__main__":
    model_name = "EleutherAI/pythia-70m"
    file_path = "dataprep_pipeline/data/dataset_test.csv"
    pipeline = TokenizerPipeline(model_name, file_path)
    tokenized_corpus = pipeline.run()
    print(tokenized_corpus)