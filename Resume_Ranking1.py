import streamlit as st
import pandas as pd
import sys
import os
from resume_processing import process_resumes


# Streamlit UI Configuration
st.set_page_config(page_title="AI Resume Ranker", layout="wide")

# Header
st.title("📄 AI Resume Screening & Ranking System")
st.write("Upload resumes and enter a job description to rank candidates based on relevance.")

# Job description input
st.header("📌 Job Description")
job_desc = st.text_area("Enter the job description here", height=150)

# Resume upload section
st.header("📂 Upload Resumes (PDF format)")
files = st.file_uploader("Upload multiple PDF resumes", type="pdf", accept_multiple_files=True)

if files and job_desc:
    st.subheader("🔄 Processing Resumes...")

    # Process resumes and get ranked DataFrame
    results = process_resumes(job_desc, files)

    # Display ranked results
    st.subheader("🏆 Ranked Resumes")
    st.dataframe(results, hide_index=True)

    # Option to download results as CSV
    csv = results.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Results as CSV",
        data=csv,
        file_name="resume_rankings.csv",
        mime="text/csv",
    )

else:
    st.warning("⚠️ Please upload resumes and enter a job description to proceed.")
