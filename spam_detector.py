import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import nltk
import string
import re
import os

nltk.download('stopwords')
from nltk.corpus import stopwords

# Load dataset
data_path = os.path.join("data", "spam.csv")
df = pd.read_csv(data_path, encoding='latin-1')[["v1", "v2"]]
df.columns = ["label", "message"]

# Preprocessing function
def preprocess(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()
    words = text.split()
    words = [word for word in words if word not in stopwords.words('english')]
    return ' '.join(words)

df["cleaned"] = df["message"].apply(preprocess)
df["label_num"] = df["label"].map({"ham": 0, "spam": 1})

# Features and labels
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["cleaned"])
y = df["label_num"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Predict custom SMS
def predict_sms(sms):
    cleaned = preprocess(sms)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)
    return "Spam" if prediction[0] == 1 else "Ham"

# Example usage
example_sms = "Congratulations! You've won a free cruise to the Bahamas. Call now!"
print(f"\nPrediction for: \"{example_sms}\" ->", predict_sms(example_sms))
