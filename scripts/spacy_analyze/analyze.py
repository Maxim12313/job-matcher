import sys
from collections import defaultdict
import pandas as pd
import spacy
import nltk
from spacy import displacy
from pyresparser import ResumeParser
import warnings

warnings.filterwarnings(action="ignore", category=UserWarning)


class Analyzer:
    df = None
    nlp = None

    def get_doc(self, text):
        return self.nlp(text)

    def get_skills(self, text):
        doc = self.nlp(text)
        skills = [ent.text for ent in doc.ents if ent.label_ == "SKILL"]
        skill_freq = defaultdict(int)
        for skill in skills:
            skill_freq[skill] += 1
        return skill_freq

    def get_data(self):
        df = pd.read_csv("./Resume.csv")
        df = df.sample(frac=1)
        return df

    def load_nlp(self):
        nlp = spacy.load("en_core_web_lg")
        ruler = nlp.add_pipe("entity_ruler")
        ruler.from_disk("jz_skill_patterns.jsonl")
        return nlp

    def __init__(self):
        self.df = self.get_data()
        self.nlp = self.load_nlp()


if __name__ == "__main__":
    # analyze = Analyzer()
    # text = analyze.df["Resume_str"][0]
    data = ResumeParser(sys.argv[1]).get_extracted_data()
    print(data)

    # html = displacy.render(analyze.get_doc(text), style="ent", jupyter=False)
    # print(html)
