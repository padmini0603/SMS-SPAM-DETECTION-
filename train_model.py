"""
SMS Spam Detection - Model Training Script
Author: Padmini
"""

import pandas as pd
import numpy as np
import string
import nltk
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle
import os

# Download stopwords
nltk.download('stopwords')
nltk.download('punkt')

# 1️⃣ Load Dataset
df = pd.read_csv("spam.csv", encoding='latin-1')
df = df[['v1', 'v2']]
df.columns = ['label', 'message']
df['label_num'] = df.label.map({'ham': 0, 'spam': 1})

# 2️⃣ Text Preprocessing
ps = PorterStemmer()
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = [ps.stem(word) for word in text 
         if word.isalnum() 
         and word not in stopwords.words('english') 
         and word not in string.punctuation]
    return " ".join(y)

df['transformed_message'] = df['message'].apply(transform_text)

# 3️⃣ Feature Extraction
cv = TfidfVectorizer(max_features=3000)
X = cv.fit_transform(df['transformed_message']).toarray()
y = df['label_num'].values

# 4️⃣ Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5️⃣ Train Model
model = MultinomialNB()
model.fit(X_train, y_train)

# 6️⃣ Evaluate Model
y_pred = model.predict(X_test)
print("\n--- Model Evaluation ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 7️⃣ Confusion Matrix Heatmap
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.title("Confusion Matrix Heatmap")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# 8️⃣ Save Model
os.makedirs("models", exist_ok=True)
pickle.dump(model, open("models/spam_model.pkl", "wb"))
pickle.dump(cv, open("models/vectorizer.pkl", "wb"))

print("\n✅ Model and vectorizer saved successfully in /models folder.")
