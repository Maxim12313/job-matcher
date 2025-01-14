import string
import re

from .parser import parse_resume
from .resume_knn.knn import ResumeKNN


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
