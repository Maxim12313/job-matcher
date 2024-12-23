import spacy
from spacy.tokens import DocBin
from datasets import load_dataset


def keep_row(row):
    return all(col is not None for col in row)


def replace_newlines(row):
    row["content"] = row["content"].replace("\n", " ")
    return row


def clean_data():
    ds = load_dataset("json", data_files="./training-data.json", split="train")
    ds = ds.select_columns(["content", "annotation"])
    ds = ds.filter(keep_row)
    ds = ds.map(replace_newlines)
    return ds


def convert_to_spacy():
    # data set peculiarities explained here: https://www.kaggle.com/code/taha07/ner-on-resumes-using-spacy
    # format guide https://spacy.io/usage/training#training-data
    ds = clean_data()
    nlp = spacy.blank("en")
    db = DocBin()
    for it in ds:
        (content, notes) = it.values()
        doc = nlp(content)
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
                span = doc.char_span(start, end + 1, label=label)
                ents.append(span)

        db.add(doc)
    db.to_disk("./training-data.spacy")


if __name__ == "__main__":
    convert_to_spacy()
