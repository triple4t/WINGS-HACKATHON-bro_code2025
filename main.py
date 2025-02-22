import re
import PyPDF2
import gradio as gr
from pdfminer.high_level import extract_text
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import os

class ResumeAnalyzer:
    SKILLS = {
        'programming': {'python', 'java', 'javascript', 'js', 'c++', 'ruby', 'php', 'swift', 'kotlin', 'typescript', 'golang', 'go', 'rust', 'scala', 'perl', 'shell', 'bash'},
        'web_tech': {'html', 'css', 'react', 'angular', 'vue', 'node.js', 'nodejs', 'django', 'flask', 'express', 'springboot', 'spring', 'asp.net', 'jquery', 'bootstrap', 'tailwind', 'webpack', 'redux', 'rest', 'restful', 'graphql'},
        'database': {'sql', 'mongodb', 'postgresql', 'mysql', 'redis', 'oracle', 'nosql', 'dynamodb', 'cassandra', 'elasticsearch', 'mariadb', 'sqlite'},
        'cloud': {'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'k8s', 'terraform', 'cloud', 'microservices', 'serverless', 'lambda', 'ec2', 's3'},
        'tools': {'git', 'jenkins', 'jira', 'confluence', 'slack', 'gradle', 'maven', 'docker', 'kubernetes', 'ci/cd', 'cicd', 'agile', 'scrum'},
        'machine_learning': {'tensorflow', 'pytorch', 'scikit-learn', 'sklearn', 'ml', 'ai', 'machine learning', 'deep learning', 'nlp', 'computer vision'}
    }

    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2), min_df=1, max_df=0.95)

    def extract_text_from_pdf(self, pdf_path):
        """Extract text from PDF with error handling"""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = []
                for page in reader.pages:
                    content = page.extract_text()
                    if content:
                        text.append(content)
                if not text:
                    return None, "PDF appears to be empty or unreadable"
                return ' '.join(text), None
        except Exception as e:
            return None, f"Error reading PDF: {str(e)}"

    def clean_text(self, text):
        """Enhanced text cleaning"""
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s\.]', ' ', text)
        text = ' '.join(text.split())
        return text

    def extract_skills(self, text):
        """Enhanced skill extraction"""
        text = self.clean_text(text)
        words = set(text.split())
        found_skills = defaultdict(set)

        word_pairs = [' '.join(text.split()[i:i+2]) for i in range(len(text.split())-1)]

        for category, skills in self.SKILLS.items():
            for skill in skills:
                if skill.lower() in words or skill.replace('.', '') in words or skill.lower() in word_pairs:
                    found_skills[category].add(skill)

        return found_skills

    def calculate_similarity(self, text1, text2):
        """Calculate weighted similarity score"""
        try:
            vectors = self.vectorizer.fit_transform([text1, text2])
            content_similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
            
            resume_skills = self.extract_skills(text1)
            jd_skills = self.extract_skills(text2)
            
            total_jd_skills = sum(len(skills) for skills in jd_skills.values())
            if total_jd_skills == 0:
                return content_similarity
            
            matched_skills = 0
            for category in jd_skills:
                if category in resume_skills:
                    matched_skills += len(resume_skills[category] & jd_skills[category])
            
            skill_similarity = matched_skills / total_jd_skills if total_jd_skills > 0 else 0
            
            final_similarity = (0.4 * content_similarity) + (0.6 * skill_similarity)
            adjusted_similarity = (final_similarity * 0.7) + 0.3
            
            return min(adjusted_similarity, 1.0)
        except Exception as e:
            return 0.0

    def calculate_resume_score(self, resume_text):
        """Evaluate the formatting of the resume"""
        score = 0
        sections = ["experience", "education", "skills", "certifications", "projects"]
        for section in sections:
            if section in resume_text:
                score += 10
        
        if "•" in resume_text or "-" in resume_text:
            score += 10
        
        word_count = len(resume_text.split())
        if word_count > 300:
            score += 10

        return score

    def evaluate_format(self, resume_text):
        """Evaluate Resume Format"""
        evaluation = {}

        font_issues = len(re.findall(r'[^\w\s,.()-]', resume_text))
        evaluation["Font Consistency"] = "Good" if font_issues < 5 else "Inconsistent fonts detected"

        sections = ["education", "experience", "skills", "projects", "summary"]
        found_sections = [section for section in sections if re.search(fr"\b{section}\b", resume_text, re.IGNORECASE)]
        evaluation["Headings"] = f"Sections found: {', '.join(found_sections)}" if found_sections else "No standard headings detected"

        bullet_points = len(re.findall(r"[-•*]\s", resume_text))
        evaluation["Bullet Points"] = "Used properly" if bullet_points > 2 else "Few or no bullet points found"

        empty_lines = len(re.findall(r"\n\s*\n", resume_text))
        evaluation["Whitespace Management"] = "Proper spacing" if empty_lines < 5 else "Excessive blank spaces detected"

        score = sum(1 for val in evaluation.values() if "Good" in val or "Used properly" in val)
        evaluation["Formatting Score"] = f"{(score / len(evaluation)) * 100:.2f}%"

        return evaluation

    def analyze_resume(self, pdf_path, job_description):
        """Main analysis function"""
        try:
            resume_text, pdf_error = self.extract_text_from_pdf(pdf_path)
            if pdf_error:
                return {
                    "ATS Score": "0%",
                    "Resume Score": "0%",
                    "Matched Skills": "",
                    "Missing Skills": "",
                    "Suggestions": ["⚠️ Could not process the PDF. Please check the file and try again."]
                }

            if not job_description.strip():
                return {
                    "ATS Score": "0%",
                    "Matched Skills": "",
                    "Missing Skills": "",
                    "Suggestions": ["⚠️ Please provide a job description."]
                }

            clean_resume = self.clean_text(resume_text)
            clean_jd = self.clean_text(job_description)

            resume_skills = self.extract_skills(clean_resume)
            jd_skills = self.extract_skills(clean_jd)

            similarity = self.calculate_similarity(clean_resume, clean_jd)
            resume_score = self.calculate_resume_score(resume_text)

            format_evaluation = self.evaluate_format(resume_text)
            formatted_score = float(format_evaluation["Formatting Score"].strip('%'))

            skill_gaps = defaultdict(set)
            for category, skills in jd_skills.items():
                missing = skills - resume_skills.get(category, set())
                if missing:
                    skill_gaps[category] = missing

            recommendations = []
            if similarity < 0.5:
                recommendations.append("💡 Consider tailoring your resume more specifically to this role")
            elif similarity < 0.7:
                recommendations.append("💡 Your resume matches many requirements but could be optimized further")

            if skill_gaps:
                for category, missing in skill_gaps.items():
                    if missing:
                        recommendations.append(f"📚 Consider highlighting or adding these {category} skills: {', '.join(missing)}")

            if formatted_score > 50:
                recommendations.append("✅ Your resume has good formatting!")

            if not recommendations:
                recommendations.append("✅ Your resume appears to be very well-matched to the job requirements")

            return {
                "ATS Score": f"{similarity * 100:.2f}%",
                "Resume Score": f"{resume_score}%",
                "Matched Skills": ", ".join([f"{k}: {', '.join(v)}" for k, v in resume_skills.items() if v]),
                "Missing Skills": ", ".join([f"{k}: {', '.join(v)}" for k, v in skill_gaps.items()]),
                "Suggestions": recommendations,
                "Font Consistency": format_evaluation["Font Consistency"],
                "Headings": format_evaluation["Headings"],
                "Bullet Points": format_evaluation["Bullet Points"],
                "Whitespace Management": format_evaluation["Whitespace Management"],
                "Formatting Score": format_evaluation["Formatting Score"]
            }

        except Exception as e:
            return {
                "ATS Score": "0%",
                "Resume Score": "0%",
                "Matched Skills": "",
                "Missing Skills": "",
                "Suggestions": [f"⚠️ An error occurred: {str(e)}"],
                "Font Consistency": "Error",
                "Headings": "Error",
                "Bullet Points": "Error",
                "Whitespace Management": "Error",
                "Formatting Score": "Error"
            }

def analyze_resume_interface(pdf_file, job_description):
    """Wrapper function for Gradio interface"""
    analyzer = ResumeAnalyzer()
    result = analyzer.analyze_resume(pdf_file.name, job_description)
    
    return (
        result["ATS Score"],
        result["Resume Score"],
        result["Matched Skills"],
        result["Missing Skills"],
        result["Suggestions"],
        result["Font Consistency"],
        result["Headings"],
        result["Bullet Points"],
        result["Whitespace Management"],
        result["Formatting Score"]
    )

# Create Gradio interface
interface = gr.Interface(
    fn=analyze_resume_interface,
    inputs=[
        gr.File(label="Upload Resume (PDF format)"),
        gr.Textbox(label="Job Description", lines=5, placeholder="Paste the job description here...")
    ],
    outputs=[
        gr.Textbox(label="ATS Score"),
        gr.Textbox(label="Missing Skills"),
        gr.Textbox(label="Recommendations"),
        gr.Textbox(label="Font Consistency"),
        gr.Textbox(label="Headings"),
        gr.Textbox(label="Bullet Points"),
        gr.Textbox(label="Whitespace Management"),
        gr.Textbox(label="Formatting Score")
    ],
    title="📄 Resume Analyzer",
    description="Upload your resume and job description to get personalized insights.",
    allow_flagging="never"
)

# Launch the interface
if __name__ == "__main__":
    interface.launch(debug=True)
