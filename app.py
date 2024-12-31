import pandas as pd
import pymupdf
import streamlit as st
from scripts import ResumeKNN
from streamlit_pdf_viewer import pdf_viewer
from scripts.parser import parse_resume


def display_parsed(file_value):
    data = parse_resume(file_value)
    for key in data:
        data[key] = "\n".join(data[key])

    df = pd.DataFrame.from_dict(data, orient="index", columns=["value"])
    st.table(df.style.set_properties(**{"white-space": "pre-wrap"}))


def knn_results(file_value):
    knn = ResumeKNN()
    # TODO: getting text manually twice
    # improve code when parse is figured out
    # also must clean for only experience + skill section
    with pymupdf.open("pdf", file_value) as pdf:
        text = pdf[0].get_text()
        labels = knn.get_categories()
        pred = knn.predict(text)[0]
        st.write(f"Your skill is best suited to {pred}")

        pred_prob = knn.predict_proba(text)[0]
        probs = [(labels[i], pred_prob[i]) for i in range(len(labels))]
        table = pd.DataFrame(probs, columns=["Occupation", "Probability"])
        table.sort_values("Probability", inplace=True, ascending=False)
        table.reset_index(inplace=True, drop=True)
        st.table(table)


file = st.file_uploader("Upload PDF Resume", type="pdf")
if file:
    file_value = file.getvalue()
    pdf_viewer(file_value)
    display_parsed(file_value)
    knn_results(file_value)


st.write(file)
