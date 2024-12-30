import pandas as pd
import re
import string
from pandas import DataFrame
from sklearn.preprocessing import LabelEncoder


def clean_resume(text: str):
    # clean links
    text = re.sub(r"http\S+", " ", text)

    # remove all punctuation (imagine putting expr in {} placeholder )
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
def prepare_data(df: DataFrame):
    df = clean_data(df)
    df["Clean"] = df["Resume"].apply(clean_resume)
    df = encode_data(df)


if __name__ == "__main__":
    df = pd.read_csv("UpdatedResumeDataSet.csv", encoding="utf-8")
    df = prepare_data(df)
