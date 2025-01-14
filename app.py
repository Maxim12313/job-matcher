import pandas as pd
import pymupdf
import streamlit as st
from scripts import ResumeKNN
from streamlit_pdf_viewer import pdf_viewer
from scripts.parser import parse_resume


def display_parsed(data):
    view = {key: "\n".join(data[key]) for key in data}
    df = pd.DataFrame.from_dict(view, orient="index", columns=["value"])
    st.table(df.style.set_properties(**{"white-space": "pre-wrap"}))


def knn_results(data):
    knn = ResumeKNN()
    labels = knn.get_categories()

    excluded = set(["education", "phone", "email", "name", "gpa"])

    text = ""
    for key in data:
        if key in excluded:
            continue
        text += "".join(x for x in data[key])

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
    data = parse_resume(file_value)
    display_parsed(data)
    knn_results(data)


st.write(file)
