import spacy
import pandas as pd
import re
from spacy.tokens import DocBin


def add_ents(df):
    exclude = ["UNKNOWN", "Email Address"]
    count = 0
    all = []
    for row in df.itertuples():
        ents = []
        for it in row.annotation:
            # sometimes multiple or 1
            labels = it["label"]
            if not isinstance(labels, list):
                labels = [labels]

            (start, end, text) = it["points"][0].values()
            # don't trust start += len(text)- len(text.lstrip())
            # indexes aren't perfectly aligned for tuple text

            # pos include outer white space, change to exclude for model
            while start + 1 < len(row.content) and row.content[start].isspace():
                start += 1

            while end - 1 >= 0 and row.content[end].isspace():
                end -= 1

            # mistakes in dataset
            if start >= end:
                continue

            for label in labels:
                if label in exclude:
                    continue
                count += 1
                ents.append((start, end + 1, label))

        all.append(ents)
    print(f"total of {count}")
    return all


# merge overlapping ent ranges
# if [0, 10] person, [5, 100] org, make [0, 100] org
# choose longest or earliest in tie
def merge_ents(ents):
    # sort based on start
    ents = sorted(ents, key=lambda x: x[0])
    res = [(-100, -101, "PLACEHOLDER")]
    longest = 0
    for left, right, label in ents:
        if left <= res[-1][1]:
            new_label = res[1][2]
            if right - left + 1 > longest:
                new_label = label
                longest = right - left + 1

            res[-1] = (res[-1][0], max(right, res[-1][1]), new_label)
        else:
            res.append((left, right, label))
            longest = right - left + 1
    return res[1:]


def test_merge_ents(ents):
    ents = [(0, 10, "PERSON"), (5, 100, "ORG")]
    ents = merge_ents(ents)
    print(ents)


def process_data():
    df = pd.read_json("./data.json", lines=True)
    df = df.drop(["extras"], axis=1)
    df["content"] = df["content"].map(lambda x: x.replace("\n", " "))
    df["ents"] = add_ents(df)
    df["ents"] = df["ents"].map(merge_ents)

    for it in df.itertuples():
        for left, right, label in it.ents:
            print(label)
            print("|" + it.content[left:right] + "|")
            print()

    test_size = int(0.2 * len(df))
    train, test = df[test_size:], df[:test_size]

    return train, test


def write_data(df, name):
    nlp = spacy.blank("en")
    db = DocBin()
    for it in df.itertuples():
        doc = nlp(it.content)
        ents = []
        for start, end, label in it.ents:
            span = doc.char_span(start, end, label=label)
            if span:
                ents.append(span)
        doc.ents = ents
        db.add(doc)
    # db.to_disk("./" + name + ".spacy")


def get_all_labels(df):
    labels = set()
    for it in df.itertuples():
        for _, _, label in it.ents:
            labels.add(label)
    print(labels)


if __name__ == "__main__":
    train, test = process_data()
    # get_all_labels(train)
    write_data(train, "train")
    write_data(test, "validate")
