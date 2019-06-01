import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from scipy import sparse
import os
from config import BASE_DIR

DATA_PATH = os.path.join(BASE_DIR, 'data', 'data_for_train', 'type_class.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'data', 'models', 'type_model.pickle')
VECTORIZER_PATH = os.path.join(BASE_DIR, 'data', 'models', 'vectorizer_type.pickle')
# DATA_PATH = '/Users/ira/Downloads/inspector/app/data/data_for_train/type_class.csv'
# MODEL_PATH = '/Users/ira/Downloads/inspector/app/data/models/type_model.pickle'
# VECTORIZER_PATH = '/Users/ira/Downloads/inspector/app/data/models/vectorizer_type.pickle'

tf_idf = TfidfVectorizer()
SEED = 23


def train_model(classifier, X, Y):
    text_features = tf_idf.fit_transform(X['text'])
    with open(VECTORIZER_PATH, 'wb') as f_vect:
        pickle.dump(tf_idf.vocabulary_, f_vect)
    X_stacked = sparse.hstack([text_features])
    model = classifier.fit(X_stacked, Y)
    return model


if __name__ == '__main__':
    df = pd.read_csv(DATA_PATH)
    X = df[['text']]
    Y = df['type']
    classifier = LogisticRegression(class_weight='balanced',
                                    random_state=SEED)
    multi_model = train_model(classifier=classifier, X=X, Y=Y)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(multi_model, f)
