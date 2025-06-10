import fitz  # PyMuPDF
import re
import streamlit as st
from sentence_transformers import SentenceTransformer, util

st.set_page_config(page_title="Resume Analyzer", layout="centered")


# Load SentenceTransformer model
@st.cache_resource
def load_model():
    return SentenceTransformer("all-mpnet-base-v2")

model = load_model()

# Function to extract text from uploaded PDF
def extract_text_from_pdf(uploaded_file):
    try:
        with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
            text = ""
            for page in doc:
                text += page.get_text()
        return re.sub(r'\s+', ' ', text).strip()
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""

# Preprocessing text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)
    return text.strip()

# Skill matcher
def extract_skills(resume_text, job_desc):
    skills_list = [
        "python", "java", "c++", "javascript", "sql", "docker", "kubernetes", "cloud computing",
        "aws", "azure", "devops", "backend development", "rest api", "debugging", "problem solving"
    ]
    resume_text = resume_text.lower()
    job_desc = job_desc.lower()

    matched_skills = [skill for skill in skills_list if skill in resume_text]
    missing_skills = [skill for skill in skills_list if skill in job_desc and skill not in resume_text]

    return matched_skills, missing_skills

# Resume and JD comparison
def compare_resume_with_job(resume_text, job_desc):
    resume_text = preprocess_text(resume_text)
    job_desc = preprocess_text(job_desc)

    if not resume_text or not job_desc:
        return None, [], []

    resume_embedding = model.encode(resume_text, convert_to_tensor=True)
    job_embedding = model.encode(job_desc, convert_to_tensor=True)

    similarity_score = util.cos_sim(resume_embedding, job_embedding).item()
    matched_skills, missing_skills = extract_skills(resume_text, job_desc)

    return similarity_score, matched_skills, missing_skills


st.title("📄 Resume to Job Description Matcher")
st.write("Upload your resume (PDF) and paste a job description to find out how well they match.")

uploaded_file = st.file_uploader("Upload Resume PDF", type=["pdf"])
job_desc = st.text_area("Paste Job Description", height=200)

if uploaded_file and job_desc:
    with st.spinner("Analyzing..."):
        resume_text = extract_text_from_pdf(uploaded_file)
        if not resume_text:
            st.error("Failed to extract text from PDF.")
        else:
            score, matched_skills, missing_skills = compare_resume_with_job(resume_text, job_desc)
            if score is not None:
                st.subheader("📊 Similarity Score")
                st.success(f"Your resume matches the job description by **{round(score * 100, 2)}%**.")

                st.subheader("✅ Matched Skills")
                st.write(", ".join(matched_skills) if matched_skills else "None")

                st.subheader("❌ Missing Skills")
                st.write(", ".join(missing_skills) if missing_skills else "None")
            else:
                st.error("Could not compute similarity score.")
else:
    st.info("Please upload a resume and paste a job description to begin.")
