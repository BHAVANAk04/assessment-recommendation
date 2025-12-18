
import streamlit as st
import requests

API_URL = st.secrets.get("API_URL","https://YOUR_API_URL/recommend")

st.set_page_config(page_title="SHL Assessment Recommendation", layout="centered")
st.title("Recommended SHL Assessments")

query = st.text_area("Job Description / Hiring Requirements", height=150)

if st.button("Get Recommendations"):
    if not query.strip():
        st.warning("Please enter a query.")
    else:
        with st.spinner("Fetching recommendations..."):
            r = requests.post(API_URL, json={"query": query}, timeout=30)
            data = r.json()
            for item in data.get("recommended_assessments", []):
                st.markdown(f"- [{item['name']}]({item['url']})")
