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
                st.subheader(rec.get("name", "Unnamed Assessment"))

                # ✅ FIX: do NOT assume description exists
                if "description" in rec:
                    st.write(rec["description"])

                if "duration" in rec:
                    st.write("Duration:", rec["duration"])

                if "url" in rec:
                    st.write("URL:", rec["url"])
        else:
            st.error("API error")
