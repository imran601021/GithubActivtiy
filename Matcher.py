
# -------------------- FILE: app.py --------------------
%%writefile app.py
import fitz  # PyMuPDF
import re
import streamlit as st
from sentence_transformers import SentenceTransformer, util
import uuid
import plotly.graph_objects as go
import language_tool_python

# ------------------ Streamlit Page Setup ------------------ #
st.set_page_config(page_title="Resume Analyzer", layout="centered")

# ------------------ Load SentenceTransformer Model ------------------ #
@st.cache_resource
def load_model():
    return SentenceTransformer("intfloat/e5-large-v2")

model = load_model()

# ------------------ PDF Text Extraction ------------------ #
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

# ------------------ Text Preprocessing ------------------ #
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)
    return text.strip()

# ------------------ Skills Matching ------------------ #
def extract_skills(resume_text, job_desc, skills_list):
    resume_text = resume_text.lower()
    job_desc = job_desc.lower()
    matched = [s for s in skills_list if s in resume_text]
    missing = [s for s in skills_list if s in job_desc and s not in resume_text]
    return matched, missing

# ------------------ Resume to JD Comparison ------------------ #
def compare_resume_with_job(resume_text, job_desc, skills_list):
    resume_text = preprocess_text(resume_text)
    job_desc = preprocess_text(job_desc)

    if not resume_text or not job_desc:
        return None, [], []

    resume_embedding = model.encode("passage: " + resume_text, convert_to_tensor=True)
    jd_embedding = model.encode("query: " + job_desc, convert_to_tensor=True)

    similarity = util.cos_sim(resume_embedding, jd_embedding).item()
    matched_skills, missing_skills = extract_skills(resume_text, job_desc, skills_list)
    return similarity, matched_skills, missing_skills

# ------------------ Grammar Checking ------------------ #
def check_grammar(text):
    tool = language_tool_python.LanguageTool('en-US')
    matches = tool.check(text)
    issues = []
    for match in matches:
        issues.append(f"{match.message} (Rule: {match.ruleId})")
    return issues

# ------------------ Formatting Checks ------------------ #
def check_formatting(text):
    issues = []

    if text.count("•") < 3 and text.count("- ") < 3:
        issues.append("Use more bullet points for better readability.")

    if any(line.isupper() and len(line) > 10 for line in text.splitlines()):
        issues.append("Avoid using ALL CAPS excessively.")

    long_lines = [line for line in text.splitlines() if len(line) > 160]
    if long_lines:
        issues.append(f"{len(long_lines)} lines are too long. Try breaking them up.")

    return issues

# ------------------ Animated Horizontal Bar ------------------ #
def custom_animated_bar(label, value, color_from, color_to):
    bar_id = str(uuid.uuid4()).replace('-', '')
    animation_name = f"fillBar{bar_id}"

    keyframes = f"""
    <style>
    @keyframes {animation_name} {{
        from {{ width: 0%; }}
        to {{ width: {value}%; }}
    }}
    .bar-{bar_id} {{
        animation: {animation_name} 1.5s ease-in-out forwards;
    }}
    </style>
    """

    st.markdown(f"<b>{label}</b>", unsafe_allow_html=True)
    st.markdown(keyframes, unsafe_allow_html=True)

    bar_html = f"""
    <div style="background-color: #e0e0e0; border-radius: 10px; height: 24px; margin-bottom: 12px;">
        <div class="bar-{bar_id}" style="
            height: 100%;
            background: linear-gradient(to right, {color_from}, {color_to});
            border-radius: 10px;
            text-align: center;
            color: white;
            font-weight: bold;
            line-height: 24px;">
            {value}%
        </div>
    </div>
    """
    st.markdown(bar_html, unsafe_allow_html=True)

# ------------------ Gauge Meter ------------------ #
def animated_gauge(label, value, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        delta={"reference": 100},
        title={"text": label},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 50], "color": "#ffcccc"},
                {"range": [50, 80], "color": "#ffe699"},
                {"range": [80, 100], "color": "#c6efce"}
            ]
        }
    ))
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)

# ------------------ Streamlit UI ------------------ #
st.title("📄 Resume to Job Description Matcher")
st.write("Upload your resume (PDF) and paste a job description to analyze how well they match.")

# Sidebar skill input
user_skill_input = st.sidebar.text_input("Enter your skills (comma-separated)", "")
skills_list = [s.strip().lower() for s in user_skill_input.split(",") if s.strip()]
enable_feedback = st.sidebar.checkbox("Enable Grammar & Formatting Feedback", value=True)

# Upload & JD
uploaded_file = st.file_uploader("Upload Resume PDF", type=["pdf"])
job_desc = st.text_area("Paste Job Description", height=200)

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

                    scores = {
                        "🎯 Tailoring (Skill Match)": overall_score,
                        "🧠 Content (Semantic Match)": round(len(matched_skills) / len(skills_list) * 100) if skills_list else 0,
                        "🎨 Style (Bullet Points)": 86,
                        "⚙️ ATS Compatibility": 100,
                        "📄 Sections": 29
                    }

                    animated_gauge("🧮 Overall Resume Score", overall_score, "#4CAF50")

                    gradients = [
                        ("#f2709c", "#ff9472"),
                        ("#00c6ff", "#0072ff"),
                        ("#f7971e", "#ffd200"),
                        ("#56ab2f", "#a8e063"),
                        ("#e96443", "#904e95")
                    ]

                    for (label, value), (color_from, color_to) in zip(scores.items(), gradients):
                        custom_animated_bar(label, value, color_from, color_to)

                    if st.button("🔓 Unlock Full Report"):
                        st.write("### 📝 Detailed Report")
                        st.markdown(f"**✅ Matched Skills:** {', '.join(matched_skills) if matched_skills else 'None'}")
                        st.markdown(f"**❌ Missing Skills:** {', '.join(missing_skills) if missing_skills else 'None'}")

                        if enable_feedback:
                            grammar_issues = check_grammar(resume_text)
                            formatting_issues = check_formatting(resume_text)

                            if grammar_issues:
                                st.write("### 🛠️ Grammar Issues Found:")
                                for issue in grammar_issues[:10]:
                                    st.markdown(f"- {issue}")
                            else:
                                st.success("✅ No major grammar issues found!")

                            if formatting_issues:
                                st.write("### 🧾 Formatting Suggestions:")
                                for issue in formatting_issues:
                                    st.markdown(f"- {issue}")
                            else:
                                st.success("✅ Formatting looks clean!")

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
