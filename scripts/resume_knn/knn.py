import pandas as pd
import re
import string
from pandas import DataFrame

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
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


def encode_data(df: DataFrame):
    le = LabelEncoder()
    df["Category"] = le.fit_transform(df["Category"])
    return df


def clean_data(df: DataFrame):
    df.drop_duplicates(subset="Resume", keep="first", inplace=True)
    df.reset_index(inplace=True, drop=True)
    return df


# call this
def prepare_data(df: DataFrame) -> DataFrame:
    df = clean_data(df)
    df["Clean"] = df["Resume"].apply(clean_resume)
    df = encode_data(df)
    return df


def train_knn():
    df = pd.read_csv("UpdatedResumeDataSet.csv", encoding="utf-8")
    df = prepare_data(df)

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
    model = OneVsRestClassifier(KNeighborsClassifier())
    model.fit(X_train, y_train)

    print(f"train accuracy {model.score(X_train, y_train)}")
    print(f"test accuracy {model.score(X_test, y_test)}")
    print(f"classifying {model}")
    pred = model.predict(X_test)
    print(metrics.classification_report(y_test, pred, zero_division=True))

    return model


if __name__ == "__main__":
    train_knn()
