import pandas as pd
from gensim.models import Word2Vec
import multiprocessing
from time import time

# Count the number of cores in a computer
cores = multiprocessing.cpu_count()

# Step 2: Read the dataset
dataset = pd.read_csv('data/dataset_test.csv')

# Step 3: Extract the questions and answers columns
questions = dataset['PRB'].tolist()
answers = dataset['SOL'].tolist()

sample_question = questions[0]
sample_answer = answers[0]

print('Sample Question:', sample_question)
print('Sample Answer:', sample_answer)

# Combine questions and answers into a single corpus
corpus = questions + answers

# Tokenize the corpus using simple split
tokenized_corpus = [doc.split() for doc in corpus]

# Initialize and build the Word2Vec model
w2v_model = Word2Vec(vector_size=100, window=5, min_count=1, workers=cores-1)

t = time()
w2v_model.build_vocab(tokenized_corpus, progress_per=10)
print('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))

t = time()
w2v_model.train(tokenized_corpus, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)
print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))

# Get the word embeddings for the sample question and answer
sample_question_tokens = sample_question.split()
sample_answer_tokens = sample_answer.split()

question_vector = [w2v_model.wv[word] for word in sample_question_tokens if word in w2v_model.wv]
answer_vector = [w2v_model.wv[word] for word in sample_answer_tokens if word in w2v_model.wv]

# print('Word2Vec Embeddings for Sample Question:', question_vector)
# print('Word2Vec Embeddings for Sample Answer:', answer_vector)

# Example of finding similar words
similar_words = w2v_model.wv.most_similar("regression")
for word, similarity in similar_words:
    print(f"{word}: {similarity}")