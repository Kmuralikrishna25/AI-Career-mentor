# career_mentor_langchain.py

import os
import streamlit as st
from PyPDF2 import PdfReader
from docx import Document
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# load environment variables from .env file
load_dotenv()
# ---------------- CONFIG ----------------
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")  # set your Gemini API key
# ----------------------------------------

# Initialize Gemini LLM via LangChain
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro", 
    google_api_key=GOOGLE_API_KEY,
    temperature=0.7
)

# ----------- HELPERS -----------

def read_pdf(file):
    """Extract text from PDF resume"""
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        text += page.extract_text() + "\n"
    return text

def read_docx(file):
    """Extract text from DOCX resume"""
    doc = Document(file)
    text = "\n".join([p.text for p in doc.paragraphs])
    return text

# ----------- PROMPT TEMPLATE -----------

resume_analysis_prompt = PromptTemplate(
    input_variables=["resume", "job_desc"],
    template="""
You are an **AI Career Mentor**.
Compare the following **resume** and **job description**.

### Resume:
{resume}

### Job Description:
{job_desc}

Provide the following analysis:
1. ✅ Skill match analysis (aligned skills).
2. ❌ Missing/required skills.
3. 📝 Suggestions to improve the resume (rewrite bullet points if needed).
4. 🎯 Top 5 likely interview questions for this JD.
"""
)

# Create LangChain LLMChain
analysis_chain = LLMChain(
    llm=llm,
    prompt=resume_analysis_prompt
)

# ----------- STREAMLIT UI -----------

st.set_page_config(page_title="AI Career Mentor", layout="wide")
st.title("👩‍💼 AI Career Mentor (Google Gemini + LangChain)")

uploaded_resume = st.file_uploader("Upload your Resume (PDF/DOCX)", type=["pdf", "docx"])
job_desc = st.text_area("Paste the Job Description here:")

if st.button("Analyze"):
    if uploaded_resume and job_desc.strip():
        # Extract resume text
        if uploaded_resume.name.endswith(".pdf"):
            resume_text = read_pdf(uploaded_resume)
        else:
            resume_text = read_docx(uploaded_resume)

        with st.spinner("Analyzing with Gemini..."):
            result = analysis_chain.run(resume=resume_text, job_desc=job_desc)

        st.subheader("📊 Career Mentor Analysis")
        st.write(result)
    else:
        st.warning("Please upload a resume and paste a job description.")
