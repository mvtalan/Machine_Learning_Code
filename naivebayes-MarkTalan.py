import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

data = {
    'document': [
        'I love playing football',
        'The new smartphone is amazing',
        'Football match tonight',
        'Latest technology trends',
        'Smartphone reviews',
        'Football fans cheering'
    ],
    'category': [0, 1, 0, 1, 1, 0]   # 0 - sports category 1-tech category
}

df = pd.DataFrame(data)

print(df)

X = df['document']
y=df['category']

#countvectorizer tokenizes the text (splits into words) and creates binary vectore
#representing the presence (1) or absence (0) of each word
vectorizer = CountVectorizer(binary=True)
X_vectorized = vectorizer.fit_transform(X)

print(X_vectorized.toarray())

#splitting the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

#initialize the Naive Bayes classifier
clf = MultinomialNB()

#train the classifier with the training data
clf.fit(X_train, y_train)

#make predictions on the test set
y_pred = clf.predict(X_test)

#calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy of model: {accuracy}')

#create two new docs and classify them either as Sports or Tech
sample_documents = [
    'Football fever grips the nation',
    'Exciting new smartphone launch event'
]

#convert to binary feature vectors
sample_X_documents = vectorizer.transform(sample_documents)

print(sample_X_documents.toarray())

#use trained model to predict category of new document
predictions = clf.predict(sample_X_documents)

print(f'Predictions for the sample documents: {predictions}')
