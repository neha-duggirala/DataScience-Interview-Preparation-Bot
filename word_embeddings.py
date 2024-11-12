import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

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

# Initialize the TfidfVectorizer
vectorizer = TfidfVectorizer()

# Fit the vectorizer on the corpus
vectorizer.fit(corpus)

# Transform the sample question and answer into TF-IDF vectors
question_vector = vectorizer.transform([sample_question])
answer_vector = vectorizer.transform([sample_answer])

print('TF-IDF Vector for Sample Question:', question_vector.toarray())
print('TF-IDF Vector for Sample Answer:', answer_vector.toarray())