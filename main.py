import gc
from tqdm.auto import tqdm
from data.load_data import load_datasets
from models.ensemble_model import get_model
from features.tfidf_vectorizer import get_vectorizer
from tokenizer.tokenizer_builder import train_tokenizer

if __name__ == "__main__":
    train, test, sub = load_datasets()

    tokenizer = train_tokenizer(train['text'].tolist())
    tokenized_train = [tokenizer.tokenize(t) for t in tqdm(train['text'].tolist())]
    tokenized_test = [tokenizer.tokenize(t) for t in tqdm(test['text'].tolist())]

    vectorizer = get_vectorizer(tokenized_train)
    tf_train = vectorizer.fit_transform(tokenized_train)
    tf_test = vectorizer.transform(tokenized_test)

    y_train = train['label'].values
    model = get_model()
    model.fit(tf_train, y_train)

    preds = model.predict_proba(tf_test)[:, 1]
    sub['generated'] = preds
    sub.to_csv('submission.csv', index=False)

    del vectorizer
    gc.collect()