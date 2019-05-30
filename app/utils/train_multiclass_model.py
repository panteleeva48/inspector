import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from scipy import sparse
import os
from config import BASE_DIR

DATA_PATH = os.path.join(BASE_DIR, 'data', 'data_for_train', 'multi_class.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'data', 'models', 'multi_model.pickle')
VECTORIZER_PATH = os.path.join(BASE_DIR, 'data', 'models', 'vectorizer_multi.pickle')
tf_idf = TfidfVectorizer()
SEED = 23
important_features = ['corrected_vv', 'num_linkings', 'av_len_sent',
                      'ls', 'num_gerunds', 'der_level3', 'nv', 'der_level4', 'num_inf',
                      'ndw', 'der_level6', 'num_func_ngrams', 'num_cl', 'num_acl', 'av_tok_before_root',
                      'num_sent', 'num_past_simple', 'freq_aux', 'corrected_vs', 'der_level5',
                      'density', 'num_coord', 'num_poss']


def train_model(classifier, X, Y):
    num_features = X[important_features].values
    text_features = tf_idf.fit_transform(X['text'])
    with open(VECTORIZER_PATH, 'wb') as f:
        pickle.dump(tf_idf.vocabulary_, f)
    X_stacked = sparse.hstack([num_features, text_features])
    model = classifier.fit(X_stacked, Y)
    return model


if __name__ == '__main__':
    df = pd.read_csv(DATA_PATH)
    X = df[important_features + ['text']]
    Y = df['class']
    classifier = LogisticRegression(class_weight='balanced',
                                    random_state=SEED)
    multi_model = train_model(classifier=classifier, X=X, Y=Y)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(multi_model, f)
