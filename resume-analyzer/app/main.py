import sys
import os
import gradio as gr

# Add the root project directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now, you can import the models and utils modules
from models.model import load_model
from models.vectorizer import load_vectorizer
from utils.pdf_extraction import extract_text_from_pdf
from utils.text_processing import preprocess_text
from utils.skill_extraction import extract_skills
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

# Load pre-trained model and vectorizer
model = load_model()
vectorizer = load_vectorizer()

def analyze_resume(pdf_path, job_description):
    # Text extraction and cleaning
    resume_text = extract_text_from_pdf(pdf_path)
    resume_clean = preprocess_text(resume_text)
    jd_clean = preprocess_text(job_description)

    # Vectorization
    resume_vec = vectorizer.transform([resume_clean])
    jd_vec = vectorizer.transform([jd_clean])

    # Similarity calculation
    similarity = cosine_similarity(resume_vec, jd_vec)[0][0]

    # Skill analysis
    resume_skills = extract_skills(resume_clean)
    jd_skills = extract_skills(jd_clean)

    # Generate matches and gaps
    matched_skills = defaultdict(list)
    required_skills = defaultdict(list)

    for category in jd_skills:
        matched = resume_skills[category] & jd_skills[category]
        required = jd_skills[category] - resume_skills[category]

        if matched:
            matched_skills[category] = sorted(matched)
        if required:
            required_skills[category] = sorted(required)

    recommendations = []
    if similarity < 0.6:
        recommendations.append("🌟 Improve overall alignment with job requirements")

    for category, skills in required_skills.items():
        if skills:
            recommendations.append(f"📚 Learn {category.replace('_', ' ').title()} skills: {', '.join(skills[:3])}")

    if not recommendations:
        recommendations.append("✅ Strong match! Consider emphasizing your technical skills")

    return {
        "similarity_score": f"{similarity*100:.1f}%",
        "skill_match": dict(matched_skills),
        "skill_gaps": dict(required_skills),
        "recommendations": recommendations
    }

# Gradio interface
def format_output(output):
    formatted = f"Match Score: {output['similarity_score']}\n\n"

    formatted += "✅ Matched Skills:\n"
    for category, skills in output['skill_match'].items():
        formatted += f"  - {category.replace('_', ' ').title()}: {', '.join(skills)}\n"

    formatted += "\n🚧 Skill Gaps:\n"
    for category, skills in output['skill_gaps'].items():
        formatted += f"  - {category.replace('_', ' ').title()}: {', '.join(skills)}\n"

    formatted += "\n💡 Recommendations:\n"
    for rec in output['recommendations']:
        formatted += f"  - {rec}\n"

    return formatted

# Define Gradio interface
interface = gr.Interface(
    fn=lambda pdf, jd: format_output(analyze_resume(pdf, jd)),
    inputs=[
        gr.File(label="Upload Resume PDF"),
        gr.Textbox(label="Job Description", lines=5)
    ],
    outputs=gr.Textbox(label="Analysis Report", lines=20),
    title="🚀 AI-Powered Resume Optimizer",
    description="Technical Skills Analyzer for DevOps, Web & App Development Roles",
    allow_flagging="never"
)

if __name__ == "__main__":
    interface.launch()
