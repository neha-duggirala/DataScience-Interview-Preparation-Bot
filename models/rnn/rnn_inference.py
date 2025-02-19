import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
from gensim.models import Word2Vec

# Load the trained model
model = load_model('models/rnn/rnn_model.h5')

# Load the dataset to get the word_index and max_length
dataset = pd.read_csv('dataprep_pipeline/data/dataset_test.csv')

# Extract the questions and answers columns
questions = dataset['PRB'].tolist()
answers = dataset['SOL'].tolist()

# Combine questions and answers into a single corpus
corpus = questions + answers

# Tokenize the corpus using simple split
tokenized_corpus = [doc.split() for doc in corpus]

# Train a Word2Vec model on the tokenized corpus
w2v_model = Word2Vec(vector_size=100, window=5, min_count=1, workers=4)
w2v_model.build_vocab(tokenized_corpus)
w2v_model.train(tokenized_corpus, total_examples=w2v_model.corpus_count, epochs=30)

# Prepare the data for the RNN
def get_embedding_matrix(word2vec_model):
    vocab_size = len(word2vec_model.wv)
    embedding_matrix = np.zeros((vocab_size, word2vec_model.vector_size))
    for i, word in enumerate(word2vec_model.wv.index_to_key):
        embedding_matrix[i] = word2vec_model.wv[word]
    return embedding_matrix

embedding_matrix = get_embedding_matrix(w2v_model)
vocab_size, embedding_dim = embedding_matrix.shape

# Convert text to sequences of indices
word_index = {word: i for i, word in enumerate(w2v_model.wv.index_to_key)}
index_word = {i: word for word, i in word_index.items()}
sequences = [[word_index[word] for word in doc if word in word_index] for doc in tokenized_corpus]

# Pad sequences to ensure uniform length
max_length = max(len(seq) for seq in sequences)

# Function to preprocess the input question
def preprocess_question(question, word_index, max_length):
    tokens = question.split()
    sequence = [word_index[word] for word in tokens if word in word_index]
    padded_sequence = pad_sequences([sequence], maxlen=max_length, padding='post')
    return padded_sequence

# Function to decode the sequence of indices to a sentence
def decode_sequence(sequence, index_word):
    return ' '.join([index_word.get(index, '') for index in sequence])

# Function to ask a question and get an answer
def ask_question(question, max_length=20):
    preprocessed_question = preprocess_question(question, word_index, max_length)
    prediction = model.predict(preprocessed_question)
    predicted_sequence = np.argmax(prediction, axis=-1)
    answer = decode_sequence(predicted_sequence, index_word)
    return answer

# Function to generate text by continuously predicting the next words
def generate_text(seed_text, max_length=20, max_words=50):
    result = seed_text
    for _ in range(max_words):
        preprocessed_question = preprocess_question(result, word_index, max_length)
        prediction = model.predict(preprocessed_question)
        predicted_word_index = np.argmax(prediction[0,-1])
        predicted_word = index_word.get(predicted_word_index, '')
        if predicted_word == '':
            break
        result += ' ' + predicted_word
        if len(result.split()) >= max_length:
            break
    return result

# Example usage
question = "Define logistic regression predictor variable."
answer = ask_question(question)
print("Predicted Answer:", answer)

# Example of generating text
seed_text = "logistic regression"
generated_text = generate_text(seed_text)
print("Generated Text:", generated_text)