import pandas as pd
import os
import re
import string
from pandas import DataFrame
import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics


# support from https://www.kaggle.com/code/gauravduttakiit/resume-screening-using-machine-learning
def clean_resume(text: str):
    # clean links
    text = re.sub(r"http\S+", " ", text)

    # remove all punctuation
    expr = re.escape(string.punctuation)
    text = re.sub(r"[{}]".format(expr), " ", text)

    # remove all non ascii
    text = re.sub(r"[^\x00-\x7f]", " ", text)

    # remove extra whitespace
    text = re.sub(r"\s+", " ", text)

    return text


def clean_data(df: DataFrame):
    df.drop_duplicates(subset="Resume", keep="first", inplace=True)
    df.reset_index(inplace=True, drop=True)
    return df


# call this
def prepare_data(df: DataFrame) -> DataFrame:
    df = clean_data(df)
    df["Clean"] = df["Resume"].apply(clean_resume)
    return df


def train_save_knn():
    df = pd.read_csv("UpdatedResumeDataSet.csv", encoding="utf-8")
    df = prepare_data(df)

    le = LabelEncoder()
    df["Category"] = le.fit_transform(df["Category"])

    resumes = df["Clean"].values
    categories = df["Category"].values

    # vectorize resume data by vocab by TF-IDF strategy
    vectorizer = TfidfVectorizer(sublinear_tf=True, stop_words="english")
    vectorizer.fit(resumes)
    features = vectorizer.transform(resumes)

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        categories,
        test_size=0.2,
        shuffle=True,
        stratify=categories,
        random_state=2024,
    )

    # choose highest freq label in neighbors
    model = KNeighborsClassifier()
    model.fit(X_train, y_train)

    print(f"train accuracy {model.score(X_train, y_train)}")
    print(f"test accuracy {model.score(X_test, y_test)}")
    print(f"classifying {model}")

    pred = model.predict(X_test)
    print(metrics.classification_report(y_test, pred, zero_division=True))

    # wb write binary data
    with open("model.pickle", "wb") as file:
        pickle.dump(model, file)

    with open("encoder.pickle", "wb") as file:
        pickle.dump(le, file)

    with open("vectorizer.pickle", "wb") as file:
        pickle.dump(vectorizer, file)


# TODO: also remove name, email, location, phone number from resume
class ResumeKNN:
    model: KNeighborsClassifier = None
    encoder: LabelEncoder = None
    vectorizer: TfidfVectorizer = None

    def __init__(self):
        dir = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(dir, "model.pickle"), "rb") as file:
            self.model = pickle.load(file)

        with open(os.path.join(dir, "encoder.pickle"), "rb") as file:
            self.encoder = pickle.load(file)

        with open(os.path.join(dir, "vectorizer.pickle"), "rb") as file:
            self.vectorizer = pickle.load(file)

    def get_categories(self):
        return self.encoder.classes_

    def predict(self, data):
        if isinstance(data, str):
            data = [data]
        if not isinstance(data, list):
            print(f"predict with {type(data)}")
            assert False

        data = [clean_resume(x) for x in data]
        features = self.vectorizer.transform(data)
        pred = self.model.predict(features)
        return self.encoder.inverse_transform(pred)

    def predict_proba(self, data):
        if not isinstance(data, list):
            data = [data]

        data = [clean_resume(x) for x in data]
        features = self.vectorizer.transform(data)
        return self.model.predict_proba(features)


if __name__ == "__main__":
    # train_save_knn()
    df = pd.read_csv("UpdatedResumeDataSet.csv", encoding="utf-8")
    df = prepare_data(df)
    sample = df["Clean"][0]
    knn = ResumeKNN()
    print(knn.predict_proba(sample))
