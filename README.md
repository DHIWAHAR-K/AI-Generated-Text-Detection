# AI Generated Text Detection

This project implements an ensemble-based text classification pipeline using LightGBM, CatBoost, Naive Bayes, SGD, and Random Forest classifiers. It features a custom-trained tokenizer based on Byte Pair Encoding (BPE) and a TF-IDF vectorization approach using 3-5 word n-grams. The model predicts the likelihood that an essay was AI-generated based on textual features.

## Project Structure

- `data/load_data.py`: Loads and combines various training datasets, filters prompts, removes duplicates.
- `tokenizer/tokenizer_builder.py`: Builds a custom BPE tokenizer using the Hugging Face `tokenizers` library.
- `features/tfidf_vectorizer.py`: Implements TF-IDF vectorization using tokenized n-gram sequences.
- `models/ensemble_model.py`: Defines a weighted soft-voting ensemble with five classifiers.
- `utils/helpers.py`: Contains utility functions (e.g., `dummy()` used as a tokenizer/preprocessor).
- `main.py`: The main script that runs the entire pipeline—from loading data to generating predictions.

## Setup and Installation

Make sure you have Python 3.7+ installed. Then install the required dependencies:

```bash
pip install pandas numpy scikit-learn lightgbm catboost transformers datasets tokenizers tqdm
```


## Datasets

Place your datasets in the following directories:

1. /data_1/train_v2_drcat_02.csv

2. /data_1/train_drcat_04.csv

3. /data_2/train_essays.csv

4. /data_2/test_essays.csv

5. /data_4/sample_submission.csv


## Usage

To run the pipeline and generate predictions:

```bash
python main.py
```

## Pipeline Steps:

	1.	Load Datasets – Merges multiple training files and removes duplicate essays.

	2.	Tokenizer Training – Trains a custom BPE tokenizer on the training text.

	3.	TF-IDF Vectorization – Applies n-gram TF-IDF (3-5) on the tokenized texts.

	4.	Model Training – Trains a VotingClassifier using five base models.

	5.	Prediction – Generates prediction probabilities on the test set.

	6.	Submission – Saves the output in submission.csv.


## Model Ensemble:

The ensemble uses a soft-voting strategy with the following weights:

Model                Weight
MultinomialNB        0.10
SGDClassifier        0.51
LightGBM             0.28
CatBoostClassifier   0.85  
RandomForest         0.35

## Features:

	•	🔡 Custom BPE Tokenizer: Improves vocabulary representation for diverse essays.

	•	📈 TF-IDF Vectorization: Captures word patterns through 3–5 n-gram analysis.

	•	🧠 Weighted Voting Ensemble: Combines multiple classifiers for robust performance.

	•	⚙️ Efficient Memory Handling: Uses gc.collect() to manage memory during vectorization.

## License:

This project is open-source and available for educational and research purposes. Contributions are welcome!