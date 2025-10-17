# 📱 SMS Spam Detector

A simple machine learning project to detect spam messages using Natural Language Processing and a Naive Bayes classifier.

## 🧠 Model

- Multinomial Naive Bayes
- Text vectorization with CountVectorizer
- Preprocessing with stopword removal and punctuation stripping

## 📁 Dataset

Download the SMS Spam Collection dataset from [UCI Repository](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection) and place it in the `data/` folder as `spam.csv`.

## 🚀 How to Run

```bash
git clone https://github.com/yourusername/sms-spam-detector.git
cd sms-spam-detector
pip install -r requirements.txt
python spam_detector.py
