# train_model.py
import os
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

def generate_synthetic_data():
    """Generates a synthetic dataset for document classification."""
    documents = [
        # Invoices / Financials
        "Invoice INV-2026-001. Total amount due: $1,450. Please remit payment to accounts.",
        "Purchase order PO-9982. Billing address: 123 Corporate Way. Net 30 payment terms.",
        "Receipt for software subscription. Tax invoice total including VAT is $59.99.",
        
        # Resumes / CVs
        "Experienced software engineer proficient in Python, Flask, React, and SQL database design.",
        "Data Scientist CV. 5+ years experience with scikit-learn, TensorFlow, and predictive modeling.",
        "Human Resources professional specializing in talent acquisition, onboarding, and recruiting.",
        
        # Technical Papers / Reports
        "Abstract: This paper proposes a novel convolutional neural network architecture for image extraction.",
        "Deep learning models require significant computational resources for training optimization metrics.",
        "Analysis of quantum computing algorithms and their performance on cryptography protocols."
    ]
    
    # 0: Invoice, 1: Resume, 2: Technical Paper
    labels = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    return documents, labels

def train_and_save():
    print("Generating synthetic data...")
    X, y = generate_synthetic_data()
    
    print("Building TF-IDF + Logistic Regression pipeline...")
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', ngram_range=(1, 2))),
        ('clf', LogisticRegression(C=1.0))
    ])
    
    print("Training model...")
    pipeline.fit(X, y)
    
    # Ensure directory exists
    os.makedirs('models', exist_ok=True)
    
    model_path = 'models/document_classifier.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(pipeline, f)
        
    print(f"Model successfully saved to {model_path}")

if __name__ == '__main__':
    train_and_save()