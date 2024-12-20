import sys
import spacy
from pypdf import PdfReader


def readPdf(path):
    reader = PdfReader(path)
    text = reader.pages[0].extract_text()
    # do any cleaning
    return text


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: parser.py [pdf path]")
        exit(1)
    path = sys.argv[1]
    text = readPdf(path)
    print(text)
