import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import warnings

warnings.filterwarnings('ignore')

# Download required NLTK resources silently
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

def preprocess_text(text):
    """Strict text pre-processing pipeline"""
    # 1. Lowercase
    text = text.lower()
    
    # 2. Tokenization
    tokens = word_tokenize(text)
    
    # Remove punctuation (keep only alphabetic tokens)
    tokens = [word for word in tokens if word.isalpha()]
    
    # 3. Stop-Word Removal
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # 4. Lemmatization (reducing words to their base form)
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return " ".join(tokens)

def main():
    # 1. Load Unstructured Text Data (Product Reviews)
    print("Loading unstructured text data (Product Reviews)...")
    df = pd.read_csv('reviews_dataset.csv')
    print("Dataset ready.\n")

    # 2. Strict Text Pre-Processing
    print("Executing strict Text Pre-Processing pipeline...")
    df['Clean_Review'] = df['Review'].apply(preprocess_text)
    
    print("\n[Example of Pre-Processing]")
    print(f"Original Text:  '{df['Review'].iloc[0]}'")
    print(f"Processed Text: '{df['Clean_Review'].iloc[0]}'\n")

    # 3. TF-IDF Vectorization
    print("Converting text into mathematical arrays (TF-IDF)...")
    tfidf = TfidfVectorizer()
    X = tfidf.fit_transform(df['Clean_Review'])
    y = df['Sentiment']
    print(f"TF-IDF Matrix Shape (Rows=Reviews, Cols=Unique Words): {X.shape}\n")

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Train Classifiers (Naive Bayes and SVM)
    print("--- Training Naive Bayes Classifier ---")
    nb_model = MultinomialNB()
    nb_model.fit(X_train, y_train)
    nb_preds = nb_model.predict(X_test)
    print(f"Naive Bayes Accuracy: {accuracy_score(y_test, nb_preds) * 100:.2f}%")
    
    print("\n--- Training SVM Classifier ---")
    svm_model = SVC(kernel='linear', random_state=42)
    svm_model.fit(X_train, y_train)
    svm_preds = svm_model.predict(X_test)
    print(f"SVM Accuracy: {accuracy_score(y_test, svm_preds) * 100:.2f}%\n")

    # 5. Predict on New, Unseen Reviews
    print("--- Predicting on New Unseen Reviews ---")
    new_reviews = [
        "I really loved this new product, it is absolutely fantastic and works great!",
        "This is the worst thing I have ever bought, completely broken out of the box."
    ]
    
    for review in new_reviews:
        # Pass the new review through the exact same pipeline
        cleaned_review = preprocess_text(review)
        vectorized_review = tfidf.transform([cleaned_review])
        
        # Predict using the SVM model
        prediction = svm_model.predict(vectorized_review)[0]
        sentiment = "Positive" if prediction == 1 else "Negative"
        
        print(f"Review: '{review}'")
        print(f"Math Array (Non-Zero features): {vectorized_review.nnz}")
        print(f"Predicted Sentiment (SVM): {sentiment}\n")

if __name__ == "__main__":
    main()
