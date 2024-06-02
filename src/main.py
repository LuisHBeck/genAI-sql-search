import streamlit as st
import langchain_helper as lch

st.title("SQL AI Search Tool")

question = st.text_input(label="Input your question here:")

if st.button("Search"):
    response = lch.run_db_chain(question)
    st.subheader("Answer: ")
    st.write(response)