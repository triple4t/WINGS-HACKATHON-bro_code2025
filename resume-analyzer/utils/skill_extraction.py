# skill_extraction.py
from collections import defaultdict
import re

SKILLS = {
    'data_science': {'python', 'r', 'sql', 'tensorflow', 'pytorch', 'spark', 'hadoop', 'pandas', 'numpy', 'statistics', 'mlops'},
    'devops': {'docker', 'kubernetes', 'aws', 'azure', 'gcp', 'terraform', 'ansible', 'jenkins', 'gitlab', 'prometheus', 'grafana'},
    'web_dev': {'javascript', 'react', 'angular', 'vue', 'node.js', 'django', 'flask', 'php', 'wordpress', 'css', 'html5', 'rest', 'graphql'},
    'app_dev': {'swift', 'kotlin', 'flutter', 'react native', 'android', 'ios', 'xcode', 'firebase', 'mobile testing', 'push notifications'},
    'hr': {'recruitment', 'onboarding', 'employee relations', 'compensation', 'performance management', 'hr analytics', 'successfactors'},
    'advocate': {'litigation', 'contract law', 'ip law', 'legal research', 'corporate law', 'dispute resolution', 'compliance'}
}

def extract_skills(text):
    tokens = set(re.sub(r'\s+', ' ', text).split())
    found_skills = defaultdict(set)

    for skill in SKILLS:
        if skill in tokens or skill.replace('.', '') in tokens:
            for category in SKILLS[skill]:
                found_skills[category].add(skill)

    return found_skills
