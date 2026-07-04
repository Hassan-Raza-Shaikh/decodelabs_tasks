# DecodeLabs Data Analytics & ML Tasks

This repository contains four standalone Python scripts demonstrating various data processing, machine learning, and natural language processing pipelines.

## Project Structure
- `Dataset for Data Analytics - Sheet1.csv`: The original dataset used as the foundation for the tasks.

### Task 1: Data Cleaning & Feature Engineering
- **Directory**: `task 1/`
- **Script**: `data_cleaning.py`
- **Output**: `cleaned_dataset.csv`
- **Description**: Handles missing data using statistical imputation (Median for numeric, Mode for categorical). Identifies and neutralizes outliers using the Interquartile Range (IQR). Engineers three new predictive features (`OrderMonth`, `HighValueOrder`, `AvgPricePerItemInCart`).

### Task 2: Fraud Detection on Imbalanced Data
- **Directory**: `task 2/`
- **Script**: `fraud_detection.py`
- **Dataset**: `fraud_dataset.csv` (Highly imbalanced dataset)
- **Description**: Builds a classification pipeline utilizing SMOTE (Synthetic Minority Over-sampling Technique) to handle class imbalance. Trains and tunes Logistic Regression and Random Forest models using `GridSearchCV`. Discards simple accuracy in favor of robust evaluation metrics (Precision, Recall, ROC-AUC).

### Task 3: Customer Persona Clustering
- **Directory**: `task 3/`
- **Script**: `clustering.py`
- **Output**: `customer_personas.csv`
- **Description**: Uses dimensionality reduction (PCA) to reduce 32+ one-hot encoded features down to 3 principal components. Mathematically proves the optimal number of clusters (K=2) using the Elbow Method and Silhouette Score. Groups the unstructured retail data into distinct, actionable business personas.

### Task 4: NLP Text Classification
- **Directory**: `task 4/`
- **Script**: `nlp_classification.py`
- **Dataset**: `reviews_dataset.csv` (Product Reviews)
- **Description**: Implements a strict natural language text pre-processing pipeline (Tokenization, Stop-Word removal, Lemmatization) using NLTK. Vectorizes text into mathematical arrays via TF-IDF. Trains Support Vector Machine (SVM) and Naive Bayes classifiers to predict sentiment (Positive/Negative) on unseen reviews.

## Setup & Installation

Ensure you have Python 3.8+ installed. Install the dependencies using:

```bash
pip install -r requirements.txt
```

## Running the Tasks

Navigate into any task folder and execute the Python script directly. For example:
```bash
cd "task 1"
python data_cleaning.py
```
