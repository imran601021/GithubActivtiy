%%writefile app.py
import fitz  # PyMuPDF
import re
import streamlit as st
from sentence_transformers import SentenceTransformer, util
import plotly.graph_objects as go

# Page config
st.set_page_config(page_title="Resume Analyzer", layout="centered")

# Load model
@st.cache_resource
def load_model():
    return SentenceTransformer("all-mpnet-base-v2")

model = load_model()

# Extract text from PDF
def extract_text_from_pdf(uploaded_file):
    try:
        pdf_file = uploaded_file.read()
        doc = fitz.open(stream=pdf_file, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return re.sub(r'\s+', ' ', text).strip()
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""

# Preprocessing
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)
    return text.strip()

# Skill extraction
def extract_skills(resume_text, job_desc, skills_list):
    resume_text = resume_text.lower()
    job_desc = job_desc.lower()
    matched = [s for s in skills_list if s in resume_text]
    missing = [s for s in skills_list if s in job_desc and s not in resume_text]
    return matched, missing

# Compare resume and JD
def compare_resume_with_job(resume_text, job_desc, skills_list):
    resume_text = preprocess_text(resume_text)
    job_desc = preprocess_text(job_desc)
    if not resume_text or not job_desc:
        return None, [], []
    resume_embedding = model.encode(resume_text, convert_to_tensor=True)
    jd_embedding = model.encode(job_desc, convert_to_tensor=True)
    similarity = util.cos_sim(resume_embedding, jd_embedding).item()
    matched_skills, missing_skills = extract_skills(resume_text, job_desc, skills_list)
    return similarity, matched_skills, missing_skills

# Gauge Chart
def gauge_chart(score):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#F39C12"},
            'steps': [
                {'range': [0, 50], 'color': "#FFCCCC"},
                {'range': [50, 80], 'color': "#FFE699"},
                {'range': [80, 100], 'color': "#C6EFCE"}
            ]
        },
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Overall Resume Score"}
    ))
    st.plotly_chart(fig, use_container_width=True)

# Custom gradient bar
def custom_animated_bar(label, value, color_from, color_to):
    progress_style = f"""
        <style>
        .custom-bar {{
            height: 24px;
            background: linear-gradient(to right, {color_from}, {color_to});
            width: {value}%;
            border-radius: 10px;
            text-align: center;
            color: white;
            font-weight: bold;
            animation: fillBar 1.5s ease-in-out;
        }}
        @keyframes fillBar {{
            from {{ width: 0%; }}
            to {{ width: {value}%; }}
        }}
        .bar-container {{
            background-color: #e0e0e0;
            border-radius: 10px;
            height: 24px;
            margin-bottom: 12px;
        }}
        </style>
    """
    st.markdown(f"<b>{label}</b>", unsafe_allow_html=True)
    st.markdown(progress_style, unsafe_allow_html=True)
    st.markdown(f"""
        <div class='bar-container'>
            <div class='custom-bar'>{value}%</div>
        </div>
    """, unsafe_allow_html=True)

# Sidebar
st.title("📄 Resume to Job Description Matcher")
st.write("Upload your resume (PDF) and paste a job description to analyze how well they match.")

user_skill_input = st.sidebar.text_input("Enter your skills (comma-separated)", "")
skills_list = [s.strip().lower() for s in user_skill_input.split(",") if s.strip()]

uploaded_file = st.file_uploader("Upload Resume PDF", type=["pdf"])
job_desc = st.text_area("Paste Job Description", height=200)

# Run analysis
if uploaded_file and job_desc:
    if not skills_list:
        st.warning("Please enter at least one skill in the sidebar.")
    else:
        with st.spinner("Analyzing..."):
            resume_text = extract_text_from_pdf(uploaded_file)
            if not resume_text:
                st.error("Failed to extract text from PDF.")
            else:
                score, matched_skills, missing_skills = compare_resume_with_job(resume_text, job_desc, skills_list)
                if score is not None:
                    overall_score = round(score * 100)

                    # Component scores
                    scores = {
                        "🎯 Tailoring (Skill Match)": overall_score,
                        "🧠 Content (Semantic Match)": round(len(matched_skills) / len(skills_list) * 100) if skills_list else 0,
                        "🎨 Style (Bullet Points)": 86,     # Placeholder
                        "⚙️ ATS Compatibility": 100,         # Placeholder
                        "📄 Sections": 29                   # Placeholder
                    }

                    # Visual
                    st.subheader("📊 Resume Score Overview")
                    gauge_chart(overall_score)

                    # Animated Gradient Bars
                    custom_animated_bar("🎯 Tailoring (Skill Match)", scores["🎯 Tailoring (Skill Match)"], "#FF416C", "#FF4B2B")
                    custom_animated_bar("🧠 Content (Semantic Match)", scores["🧠 Content (Semantic Match)"], "#36D1DC", "#5B86E5")
                    custom_animated_bar("🎨 Style (Bullet Points)", scores["🎨 Style (Bullet Points)"], "#F7971E", "#FFD200")
                    custom_animated_bar("⚙️ ATS Compatibility", scores["⚙️ ATS Compatibility"], "#11998e", "#38ef7d")
                    custom_animated_bar("📄 Sections", scores["📄 Sections"], "#f2709c", "#ff9472")

                    # Detailed Report
                    if st.button("🔓 Unlock Full Report"):
                        st.write("### 📝 Detailed Report")
                        st.markdown(f"**✅ Matched Skills:** {', '.join(matched_skills) if matched_skills else 'None'}")
                        st.markdown(f"**❌ Missing Skills:** {', '.join(missing_skills) if missing_skills else 'None'}")
                        st.write("### Recommendations:")
                        st.markdown("""
                        - Tailor your resume to better reflect the job description.
                        - Add missing technical or domain-specific skills.
                        - Improve formatting and section organization.
                        - Keep style consistent and concise.
                        """)
                else:
                    st.error("Could not compute similarity score.")
else:
    st.info("Please upload a resume and paste a job description to begin.")
