import json
import os
import spacy
import re
from collections import defaultdict
from spacy.tokens import DocBin


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
    db.to_disk("./" + name + ".spacy")


def load():
    # Define the path to the directory containing the JSON files
    directory_path = "ResumesJsonAnnotated"

    # Load all JSON files
    data = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".json"):
            with open(os.path.join(directory_path, filename), "r") as file:
                data.append(json.load(file))

    test_size = int(0.2 * len(data))
    train, test = data[test_size:], data[:test_size]
    return train, test


def clean_text(text):
    text = re.sub(r"\s", " ", text)
    # replace utf-16 extended characters with whitespace
    text = text.encode("utf-8", errors="replace").decode("utf-8")
    return text


def clean_ents(ents, text):
    res = []
    for left, right, label in ents:
        while left + 1 < len(text) and text[left].isspace():
            left += 1
        while right >= 0 and text[right - 1].isspace():
            right -= 1
        if left < right:
            res.append((left, right, label))
    return res


# just include by first
def merge_ents(ents):
    # sort based on start
    ents = sorted(ents, key=lambda x: x[0])
    res = [(-100, -101, "PLACEHOLDER")]
    for left, right, label in ents:
        if left > res[-1][1]:
            res.append((left, right, label))
    return res[1:]


def convert_to_spacy(data, name):
    nlp = spacy.blank("en")
    db = DocBin()
    total = 0
    bad = 0
    for it in data:
        text, annotations = it.values()

        text = clean_text(text)
        doc = nlp.make_doc(text)

        annotations = clean_ents(annotations, text)
        annotations = merge_ents(annotations)

        ents = []
        for start, end, label in annotations:
            span = doc.char_span(start, end, label="SKILL")
            if span:
                ents.append(span)
            else:
                # print(start, end, label)
                # print("|" + text[start:end] + "|")
                # print()
                bad += 1
            total += 1
        doc.ents = ents
        db.add(doc)
    path = f"./{name}.spacy"
    print(f"saved {path}")
    print(f"bad/total = {bad}/{total}")
    db.to_disk(path)


def get_all_labels(data):
    labels = defaultdict(int)
    count = 0
    for it in data:
        for _, _, label in it["annotations"]:
            labels[label] += 1
            count += 1

    res = sorted(labels.items(), key=lambda x: x[1])
    print(res)
    print(len(res))
    print(count)


if __name__ == "__main__":
    train, test = load()
    convert_to_spacy(train, "train")
    convert_to_spacy(test, "validate")
