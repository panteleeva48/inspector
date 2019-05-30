import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from scipy import sparse
import os
from config import BASE_DIR

DATA_PATH = os.path.join(BASE_DIR, 'data', 'data_for_train', 'bin_class.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'data', 'models', 'binary_model.pickle')
VECTORIZER_PATH = os.path.join(BASE_DIR, 'data', 'models', 'vectorizer_binary.pickle')
tf_idf = TfidfVectorizer()
SEED = 23


def train_model(classifier, X, Y):
    num_features = X[numeric_columns].values
    text_features = tf_idf.fit_transform(X['text'])
    with open(VECTORIZER_PATH, 'wb') as f:
        pickle.dump(tf_idf.vocabulary_, f)
    X_stacked = sparse.hstack([num_features, text_features])
    model = classifier.fit(X_stacked, Y)
    return model


if __name__ == '__main__':
    df = pd.read_csv(DATA_PATH)
    numeric_columns = [col for col in df.columns if col not in ['class', 'text']]
    X = df[numeric_columns + ['text']]
    Y = df['class']
    classifier = RandomForestClassifier(n_estimators=50,
                                        max_depth=20,
                                        class_weight='balanced',
                                        random_state=SEED)
    binary_model = train_model(classifier=classifier, X=X, Y=Y)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(binary_model, f)
