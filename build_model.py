import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample
import pickle

# Load dataset
df = pd.read_csv('fake_news_dataset.csv')

# Clean text
def clean_text(text):
    text = re.sub(r"http\S+", "", str(text))
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text.lower()

df['text'] = df['text'].apply(clean_text)

# Balance dataset
df_major = df[df.label == 0]
df_minor = df[df.label == 1]
df_minor_upsampled = resample(df_minor, replace=True, n_samples=len(df_major), random_state=42)
df_balanced = pd.concat([df_major, df_minor_upsampled])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(df_balanced['text'], df_balanced['label'], test_size=0.2, random_state=42)

# Vectorize
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_vec = vectorizer.fit_transform(X_train)

# Train model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Save model and vectorizer using pickle
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("✅ Model and vectorizer saved successfully.")
print(df_balanced['label'].value_counts())
