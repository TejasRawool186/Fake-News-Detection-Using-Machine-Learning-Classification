# Fake News Detection Using Machine Learning Classification

This project detects whether a news article is fake or real using natural language processing and three supervised machine learning models. A Streamlit app loads the trained models and predicts the label of user-provided news text, then shows a final decision based on majority voting.

## Project Overview

The repository contains:

- `app.py`: Streamlit web app for inference
- `cleaning.ipynb`: notebook for data cleaning, feature extraction, model training, evaluation, and model export
- `Fake.csv` and `True.csv`: original datasets
- `clean_Fake.csv` and `clean_True.csv`: cleaned datasets generated during preprocessing
- `logistic_regression_model.pkl`: trained Logistic Regression model
- `naive_bayes_model.pkl`: trained Multinomial Naive Bayes model
- `svm_model.pkl`: trained SVM model
- `tfidf_vectorizer.pkl`: saved TF-IDF vectorizer used during inference

## Workflow

### 1. Data preparation

The notebook loads `Fake.csv` and `True.csv`, assigns labels, keeps the `text` and `label` columns, removes missing values, and cleans the text with:

- lowercasing
- URL removal
- removal of numbers and special characters
- stopword removal using NLTK English stopwords
- stemming using `PorterStemmer`

After cleaning, the fake and real datasets are merged, shuffled, and filtered to remove empty rows.

### 2. Feature engineering

The cleaned text is converted into numerical features with `TfidfVectorizer(max_df=0.7)`.

### 3. Model training

The notebook trains and evaluates:

- Logistic Regression
- Multinomial Naive Bayes
- Linear SVM

The dataset split uses `train_test_split(test_size=0.2, random_state=42)`.

### 4. Inference app

The Streamlit app:

- loads the saved vectorizer and trained models with `joblib`
- accepts article text or a headline from the user
- predicts with all three models
- returns a final result using majority voting

## Reported Model Accuracy

The app currently displays these performance numbers in the sidebar:

- SVM: `99.5%`
- Logistic Regression: `98.3%`
- Naive Bayes: `93.7%`

These values come from the project's training workflow and UI configuration.

## Installation

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Download NLTK stopwords once:

```bash
python -c "import nltk; nltk.download('stopwords')"
```

## Run the App

```bash
streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal, usually `http://localhost:8501`.

## Example Usage

1. Start the Streamlit app.
2. Paste a news headline or article into the text area.
3. Click `Analyze News`.
4. Review the prediction from each model and the majority-vote final decision.

## Dependencies

The main libraries used in this project are:

- Streamlit
- pandas
- scikit-learn
- NLTK
- joblib
- matplotlib

See `requirements.txt` for the installable dependency list.

## Notes

- The notebook is the source of truth for preprocessing and training.
- The app uses the saved `.pkl` artifacts and does not retrain models at runtime.
- Large CSV and model files are included in the project, so cloning and pushing may take longer than a small code-only repository.
