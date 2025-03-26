from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import VotingClassifier, RandomForestClassifier


def get_model():
    clf = MultinomialNB(alpha=0.0225)
    sgd_model = SGDClassifier(max_iter=9000, tol=1e-4, loss="modified_huber", random_state=6743)
    
    lgb = LGBMClassifier(n_iter=3000, verbose=-1, objective='cross_entropy', metric='auc',
                         learning_rate=0.0028, colsample_bytree=0.78,
                         colsample_bynode=0.8, random_state=6743)

    cat = CatBoostClassifier(iterations=3000, verbose=0, random_seed=6543,
                             learning_rate=0.0026, subsample=0.45,
                             allow_const_label=True, loss_function='CrossEntropy')
    
    rf_model = RandomForestClassifier(random_state=6743)

    ensemble = VotingClassifier(estimators=[
        ('mnb', clf),
        ('sgd', sgd_model),
        ('lgb', lgb),
        ('cat', cat),
        ('rf', rf_model)],
        weights=[0.1, 0.51, 0.28, 0.85, 0.35],
        voting='soft', n_jobs=-1
    )
    return ensemble