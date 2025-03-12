"""
Resume Screening and Ranking App
Created by Anurag Pandey
"""

import streamlit as st
import pdfplumber
import docx
import pandas as pd
import subprocess
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import spacy.cli

# Ensure spaCy model is installed at runtime
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    spacy.cli.download("en_core_web_sm")  # Uses spacy's internal download function
    nlp = spacy.load("en_core_web_sm")


# Initialize session state for history
if "history" not in st.session_state:
    st.session_state.history = []

# Function to extract text from PDFs
def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text + "\n"
    return text.strip()

# Function to extract text from DOCX
def extract_text_from_docx(docx_file):
    doc = docx.Document(docx_file)
    return "\n".join([para.text for para in doc.paragraphs]).strip()

# Function to preprocess text
def preprocess_text(text):
    doc = nlp(text.lower())
    return " ".join([token.text for token in doc if not token.is_punct and not token.is_stop])

# Function to rank resumes
def rank_resumes(job_desc, resumes):
    job_desc = preprocess_text(job_desc)
    processed_resumes = [preprocess_text(resume) for resume in resumes if resume.strip()]

    if not processed_resumes:
        return []

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([job_desc] + processed_resumes)
    scores = cosine_similarity(vectors[0], vectors[1:]).flatten()

    return sorted(zip(resumes, scores), key=lambda x: x[1], reverse=True)

# Streamlit UI
st.title("📄 Resume Screening and Ranking App")
st.markdown("**Created by Anurag Pandey**")

# Sidebar for navigation
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Go to", ["Home", "History", "About"])

# Home Page - Resume Ranking
if page == "Home":
    st.subheader("🔍 Rank Resumes Based on Job Description")
    
    # Job description input
    job_desc = st.text_area("✍️ Paste Job Description Here")

    # Resume upload
    uploaded_files = st.file_uploader("📂 Upload Resumes (PDF or DOCX)", type=["pdf", "docx"], accept_multiple_files=True)
    resumes_text = []

    if uploaded_files:
        for file in uploaded_files:
            if file.name.endswith(".pdf"):
                text = extract_text_from_pdf(file)
            elif file.name.endswith(".docx"):
                text = extract_text_from_docx(file)
            else:
                text = ""
            if text.strip():
                resumes_text.append(text)

        if resumes_text and st.button("🚀 Rank Resumes"):
            ranked_resumes = rank_resumes(job_desc, resumes_text)

            if ranked_resumes:
                st.subheader("🏆 Ranked Resumes")
                history_entry = []
                for i, (resume, score) in enumerate(ranked_resumes, 1):
                    st.write(f"**Rank {i}: Score {score:.2f}**")
                    st.text_area(f"📜 Resume {i} Extracted Text", resume, height=150)
                    history_entry.append(f"Rank {i}: Score {score:.2f}")
                
                # Store in session state history
                st.session_state.history.append(history_entry)

            else:
                st.warning("⚠️ No valid resumes found! Ensure your files contain text.")

# History Page - Displays previously ranked resumes
elif page == "History":
    st.subheader("📜 History of Ranked Resumes")
    if st.session_state.history:
        for idx, entry in enumerate(st.session_state.history, 1):
            st.markdown(f"**Ranking {idx}:**")
            for line in entry:
                st.write(line)
    else:
        st.info("No rankings yet! Go to 'Home' to rank resumes.")

# About Page - Displays app details
elif page == "About":
    st.subheader("ℹ️ About This App")
    st.markdown("""
        - **Developed by:** Anurag Pandey  
        - **Purpose:** This app helps recruiters automatically rank resumes based on a job description using NLP and machine learning.
        - **Tech Stack:** Python, Streamlit, SpaCy, scikit-learn, PDFPlumber, and DocX.
    """)

# Custom Footer (LinkedIn & GitHub) - Bigger, Centered, Fixed at Bottom
footer = """
    <style>
        .footer {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            background-color: #262730;
            color: white;
            text-align: center;
            padding: 15px;
            font-size: 18px;
        }
        .footer a {
            color: #4CAF50;
            text-decoration: none;
            font-weight: bold;
        }
        .footer a:hover {
            text-decoration: underline;
        }
    </style>
    <div class="footer">
        🔗 Connect with me: 
        <a href=""https://www.linkedin.com/in/anurag-pandey-15559534a" target="_blank">LinkedIn</a> | 
        <a href="https://github.com/Anurag637" target="_blank">GitHub</a>
    </div>
"""
st.markdown(footer, unsafe_allow_html=True)
