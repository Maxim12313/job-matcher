import re
import pymupdf


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


# assume just first line
def get_name(text):
    i = 0
    while i < len(text) and text[i] != "\n":
        i += 1
    return text[:i]


def get_phone(text):
    # https://regex101.com/library/wZ4uU6?orderBy=RELEVANCE&search=phone
    expr = r"(?:([+]\d{1,4})[-.\s]?)?(?:[(](\d{1,3})[)][-.\s]?)?(\d{1,4})[-.\s]?(\d{1,4})[-.\s]?(\d{1,9})"
    matches = re.findall(expr, text)
    if len(matches):
        return "-".join(x for x in matches[0] if x)
    return ""


def get_email(text):
    # https://regex101.com/library/mX1xW0?orderBy=RELEVANCE&search=email
    expr = r"(([\w-]+(?:\.[\w-]+)*)@((?:[\w-]+\.)*\w[\w-]{0,66})\.([a-z]{2,6}(?:\.[a-z]{2})?))"
    matches = re.findall(expr, text)
    if len(matches):
        return matches[0][0]
    return ""


def get_gpa(text):
    expr = r"([1-4](\.[0-9]{1,2}))?\/4(\.0)?"
    matches = re.findall(expr, text)
    if len(matches):
        return matches[0][0]
    return ""


def get_section_data(text):
    sectionData = dict()
    for title in SECTION_TITLES:
        sectionData[title] = []

    section = ""
    for line in text.split("\n"):
        line = line.strip()
        lower = line.lower().split()
        if len(lower) == 0:
            continue

        if len(lower) <= 2 and lower[0] in sectionData:
            section = lower[0]
        elif len(lower) == 2 and lower[1] in sectionData:
            section = lower[1]
        elif section in sectionData:
            sectionData[section].append(line)

    return sectionData


def parse_resume(name):
    with pymupdf.open("pdf", name) as pdf:
        page = pdf[0]
        text = page.get_text()

        data = dict()
        data["name"] = [get_name(text)]
        data["email"] = [get_email(text)]
        data["phone"] = [get_phone(text)]
        data["gpa"] = [get_gpa(text)]

        data |= get_section_data(text)
        data = {key: value for key, value in data.items() if len(value)}
        return data
