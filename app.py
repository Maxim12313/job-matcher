import pandas as pd
import streamlit as st
from streamlit_pdf_viewer import pdf_viewer
from scripts.parser import parse_resume


file = st.file_uploader("Upload PDF Resume", type="pdf")
if file:
    st.write(file.name)

    button = st.download_button("Download PDF", file.getvalue())
    st.write(button)

    pdf_viewer(file.getvalue())
    data = parse_resume(file.getvalue())
    for key in data:
        data[key] = "\n".join(data[key])
    df = pd.DataFrame.from_dict(data, orient="index", columns=["value"])
    st.table(df.style.set_properties(**{"white-space": "pre-wrap"}))


st.write(file)
