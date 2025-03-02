import streamlit as st
import pdfplumber
import docx
import spacy
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load NLP model
nlp = spacy.load("en_core_web_sm")

# Function to extract text from PDFs safely
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
    return " ".join([token.text for token in doc if not token.is_punct])

# Function to rank resumes
def rank_resumes(job_desc, resumes):
    job_desc = preprocess_text(job_desc)
    processed_resumes = [preprocess_text(resume) for resume in resumes if resume.strip()]

    # Check if we have valid resumes
    if not processed_resumes:
        return []

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([job_desc] + processed_resumes)

    scores = cosine_similarity(vectors[0], vectors[1:]).flatten()
    return sorted(zip(resumes, scores), key=lambda x: x[1], reverse=True)

# Streamlit UI
st.title("Resume Screening and Ranking App")

# Job description input
job_desc = st.text_area("Paste Job Description Here")

# Resume upload
uploaded_files = st.file_uploader("Upload Resumes", type=["pdf", "docx"], accept_multiple_files=True)
resumes_text = []

if uploaded_files:
    for file in uploaded_files:
        if file.name.endswith(".pdf"):
            text = extract_text_from_pdf(file)
        elif file.name.endswith(".docx"):
            text = extract_text_from_docx(file)
        else:
            text = ""
        if text.strip():  # Only add non-empty resumes
            resumes_text.append(text)

    if resumes_text and st.button("Rank Resumes"):
        ranked_resumes = rank_resumes(job_desc, resumes_text)
        
        if ranked_resumes:
            st.subheader("Ranked Resumes")
            for i, (resume, score) in enumerate(ranked_resumes, 1):
                st.write(f"**Rank {i}: Score {score:.2f}**")
                st.text_area(f"Resume {i} Extracted Text", resume, height=150)
        else:
            st.warning("No valid resumes found! Ensure your files contain text.")
