import streamlit as st
import requests

st.title("SHL Assessment Recommendation System")

query = st.text_area("Enter job description")

if st.button("Get Recommendations"):
    if query.strip() == "":
        st.warning("Enter a job description")
    else:
        response = requests.post(
            "https://assessment-recommendation-2.onrender.com/recommend",
            json={"query": query}
        )

        if response.status_code == 200:
            data = response.json()
            for rec in data["recommended_assessments"]:
                st.subheader(rec["name"])
                st.write(rec["description"])
                st.write("Duration:", rec["duration"])
                st.write("URL:", rec["url"])
        else:
            st.error("API error")
