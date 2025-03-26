from utils.helpers import dummy
from sklearn.feature_extraction.text import TfidfVectorizer

def get_vectorizer(tokenized_texts_train):
    initial_vectorizer = TfidfVectorizer(ngram_range=(3, 5),
                                         lowercase=False,
                                         sublinear_tf=True,
                                         analyzer='word',
                                         tokenizer=dummy,
                                         preprocessor=dummy,
                                         token_pattern=None,
                                         strip_accents='unicode')
    initial_vectorizer.fit(tokenized_texts_train)
    vocab = initial_vectorizer.vocabulary_

    final_vectorizer = TfidfVectorizer(ngram_range=(3, 5),
                                       lowercase=False,
                                       sublinear_tf=True,
                                       vocabulary=vocab,
                                       analyzer='word',
                                       tokenizer=dummy,
                                       preprocessor=dummy,
                                       token_pattern=None,
                                       strip_accents='unicode')
    return final_vectorizer