import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# Load the dataset
dataset = pd.read_csv('data/dataset_test.csv')

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
sequences = [[word_index[word] for word in doc if word in word_index] for doc in tokenized_corpus]

# Pad sequences to ensure uniform length
max_length = max(len(seq) for seq in sequences)
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')

# Split the data into training and testing sets
X_train, X_test = train_test_split(padded_sequences, test_size=0.2, random_state=42)

# Define the RNN model
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, weights=[embedding_matrix], input_length=max_length, trainable=False))
model.add(LSTM(units=128, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, np.ones((len(X_train), 1)), epochs=10, batch_size=32, validation_data=(X_test, np.ones((len(X_test), 1))))

# Save the model
model.save('models/rnn_model.h5')