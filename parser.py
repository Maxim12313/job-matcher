import sys
import spacy
import re
from pypdf import PdfReader

SECTION_TITLES = [
    "education",
    "experience",
    "projects",
    "skills",
    "coursework",
    "research",
    "achievements",
    "technologies",
]


def getSectionData(text):
    sectionData = dict()
    for title in SECTION_TITLES:
        sectionData[title] = []

    section = ""
    for line in text.split("\n"):
        line = line.strip()
        lower = line.lower().split()
        if len(lower) <= 2 and lower[0] in sectionData:
            section = lower[0]
        elif len(lower) == 2 and lower[1] in sectionData:
            section = lower[1]
        elif section in sectionData:
            sectionData[section].append(line)
    return sectionData


def getLabelData(nlp, doc, label):
    id = nlp.vocab.strings[label]
    data = [it.text for it in list(doc.ents) if it.label == id]
    return data


def getPhone(text):
    # https://regex101.com/library/wZ4uU6?orderBy=RELEVANCE&search=phone
    expr = r"(?:([+]\d{1,4})[-.\s]?)?(?:[(](\d{1,3})[)][-.\s]?)?(\d{1,4})[-.\s]?(\d{1,4})[-.\s]?(\d{1,9})"
    matches = re.findall(expr, text)
    if len(matches):
        return " ".join(x for x in matches[0] if x)
    return ""


def getEmail(text):
    # https://regex101.com/library/mX1xW0?orderBy=RELEVANCE&search=email
    expr = r"([\w-]+(?:\.[\w-]+)*)@((?:[\w-]+\.)*\w[\w-]{0,66})\.([a-z]{2,6}(?:\.[a-z]{2})?)"
    matches = re.findall(expr, text)
    if len(matches):
        return matches[0][0] + "@" + ".".join(matches[0][1:])
    return ""


def getName(nlp, doc):
    return getLabelData(nlp, doc, "PERSON")[0]


def readPdf(path):
    reader = PdfReader(path)
    text = reader.pages[0].extract_text()
    # TODO: do any cleaning
    return text


def testBasic():
    path = sys.argv[1]
    text = readPdf(path)
    print("phone:", getPhone(text))
    print("email:", getEmail(text))


def testNLP():
    path = sys.argv[1]
    text = readPdf(path)
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    for it in list(doc.ents):
        print(it.label_, it.text)


def testSection():
    path = sys.argv[1]
    text = readPdf(path)
    sectionData = getSectionData(text)
    for name, data in sectionData.items():
        print(name)
        print(data)
        print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: parser.py [pdf path]")
        exit(1)
    testNLP()
    # testSection()
