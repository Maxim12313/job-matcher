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
        if line.lower() in sectionData:
            section = line.lower()
        elif section in sectionData:
            sectionData[section].append(line)
    return sectionData


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
    sectionData = getSectionData(text)
    for name, data in sectionData.items():
        print(name)
        print(data)
        print()
