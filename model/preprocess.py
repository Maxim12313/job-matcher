import os
import spacy
import json
from spacy.tokens import DocBin
from datasets import load_dataset

dir = os.path.dirname(os.path.abspath(__file__))


def keep_row(row):
    return all(col is not None for col in row)


def replace_newlines(row):
    row["content"] = row["content"].replace("\n", " ")
    return row


def load_data_split():
    ds = load_dataset("json", data_files=dir + "/data.json", split="train")
    ds = ds.select_columns(["content", "annotation"])
    ds = ds.filter(keep_row)
    ds = ds.map(replace_newlines)
    ds = ds.shuffle(seed=2024)

    # train 70, valid 15, test 15
    ds, validate = ds.train_test_split(test_size=0.15).values()
    train, test = ds.train_test_split(test_size=0.15 / 0.85).values()

    return train, validate, test


def save_data(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def convert_to_spacy(ds, name):
    # data set peculiarities explained here: https://www.kaggle.com/code/taha07/ner-on-resumes-using-spacy
    # [(text, { "entitites:", [(start, end, label)] })]
    data = []
    for it in ds:
        (content, notes) = it.values()
        ents = []
        for note in notes:
            # sometimes multiple or 1
            labels = note["label"]
            if not isinstance(labels, list):
                labels = [labels]

            (start, end, text) = note["points"][0].values()

            # remove corrupted protected information
            if "indeed.com/r/" in text:
                continue

            # pos include outer white space, change to exclude for model
            start += len(text) - len(text.lstrip())
            end -= len(text) - len(text.rstrip())

            for label in labels:
                # dataset [start, end] while spacy requires [start, end)
                ents.append((start, end + 1, label))
        data.append((content, {"entities": ents}))

    path = f"{dir}/{name}.json"
    save_data(path, data)
    print(f"saved at {path}")


def process():
    train, validate, test = load_data_split()
    convert_to_spacy(train, "train")
    convert_to_spacy(validate, "validate")
    convert_to_spacy(test, "test")


if __name__ == "__main__":
    process()
