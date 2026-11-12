"""
SmartHire - Intelligent Talent Acquisition & ATS Platform
Featuring rich saffron, orange, and golden amber dynamic color blends,
a single right-corner Dark/Light theme mode toggle, and complete recruiter tools.
"""
import streamlit as st
import sqlite3
import os
import io
import re
import bcrypt
import pandas as pd
import numpy as np
from datetime import datetime, timezone

# Optional / robust imports for file parsing
try:
    from PyPDF2 import PdfReader
except ImportError:
    PdfReader = None

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

try:
    import docx2txt
except ImportError:
    docx2txt = None

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm

# ---------------------------
# CONFIGURATION & PATHS
# ---------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_FILE = os.path.join(BASE_DIR, "users.db")
UPLOAD_DIR = os.path.join(BASE_DIR, "resumes")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
STATIC_DIR = os.path.join(BASE_DIR, "static")
STATIC_RESUMES_DIR = os.path.join(STATIC_DIR, "resumes")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(STATIC_RESUMES_DIR, exist_ok=True)

st.set_page_config(
    page_title="SmartHire | Talent Acquisition Suite",
    page_icon="SH",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------
# THEME STATE INITIALIZATION
# ---------------------------
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

# ---------------------------
# SKILL EXTRACTION DICTIONARY
# ---------------------------
COMMON_SKILLS = [
    "Python", "Java", "C++", "C#", "JavaScript", "TypeScript", "React", "Angular", "Vue", "Next.js",
    "Node.js", "Django", "Flask", "FastAPI", "Express", "SQL", "PostgreSQL", "MySQL", "MongoDB", "Redis",
    "Machine Learning", "Deep Learning", "NLP", "Scikit-Learn", "TensorFlow", "PyTorch", "LLMs", "LangChain",
    "RAG", "Vector Databases", "Transformers", "Pandas", "NumPy", "Apache Spark", "Kafka", "Airflow", "Snowflake",
    "Docker", "Kubernetes", "AWS", "GCP", "Azure", "Terraform", "Prometheus", "Grafana", "Linux", "Bash",
    "Git", "CI/CD", "REST APIs", "GraphQL", "Agile", "DevOps", "Microservices", "Data Science",
    "Cybersecurity", "SIEM", "Splunk", "OWASP", "Firewalls", "Penetration Testing", "SOC", "Cryptography",
    "Flutter", "Dart", "React Native", "Swift", "Kotlin", "Mobile UI", "Firebase", "SQLite",
    "Selenium", "Cypress", "Playwright", "PyTest", "API Testing", "Postman", "Automation",
    "C", "RTOS", "Microcontrollers", "ARM", "IoT", "Embedded", "Solidity", "Rust", "Go",
    "HTML5", "CSS3", "Tailwind CSS", "Redux", "Figma", "Analytics", "Tableau", "Power BI"
]

# ---------------------------
# VECTOR SVG LOGOS & ICON SYSTEM
# ---------------------------
SVG_LOGOS = {
    "code": '<svg width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle; display:inline-block;"><polyline points="16 18 22 12 16 6"></polyline><polyline points="8 6 2 12 8 18"></polyline></svg>',
    "server": '<svg width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle; display:inline-block;"><rect x="2" y="2" width="20" height="8" rx="2" ry="2"></rect><rect x="2" y="14" width="20" height="8" rx="2" ry="2"></rect><line x1="6" y1="6" x2="6.01" y2="6"></line><line x1="6" y1="18" x2="6.01" y2="18"></line></svg>',
    "palette": '<svg width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle; display:inline-block;"><circle cx="13.5" cy="6.5" r=".5"></circle><circle cx="17.5" cy="10.5" r=".5"></circle><circle cx="8.5" cy="7.5" r=".5"></circle><circle cx="6.5" cy="12.5" r=".5"></circle><path d="M12 2C6.5 2 2 6.5 2 12s4.5 10 10 10c.926 0 1.648-.746 1.648-1.688 0-.437-.18-.835-.437-1.125-.29-.289-.438-.652-.438-1.125a1.64 1.64 0 0 1 1.668-1.668h1.996c3.051 0 5.555-2.503 5.555-5.554C21.965 6.012 17.461 2 12 2z"></path></svg>',
    "brain": '<svg width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle; display:inline-block;"><path d="M9.5 2A2.5 2.5 0 0 1 12 4.5v15a2.5 2.5 0 0 1-4.96.44 2.5 2.5 0 0 1-2.96-3.08 3 3 0 0 1-.34-5.58 2.5 2.5 0 0 1 1.32-4.24 2.5 2.5 0 0 1 4.44-2.04z"></path><path d="M14.5 2A2.5 2.5 0 0 0 12 4.5v15a2.5 2.5 0 0 0 4.96.44 2.5 2.5 0 0 0 2.96-3.08 3 3 0 0 0 .34-5.58 2.5 2.5 0 0 0-1.32-4.24 2.5 2.5 0 0 0-4.44-2.04z"></path></svg>',
    "cloud": '<svg width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle; display:inline-block;"><path d="M18 10h-1.26A8 8 0 1 0 9 20h9a5 5 0 0 0 0-10z"></path></svg>',
    "shield": '<svg width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle; display:inline-block;"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path></svg>',
    "smartphone": '<svg width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle; display:inline-block;"><rect x="5" y="2" width="14" height="20" rx="2" ry="2"></rect><line x1="12" y1="18" x2="12.01" y2="18"></line></svg>',
    "database": '<svg width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle; display:inline-block;"><ellipse cx="12" cy="5" rx="9" ry="3"></ellipse><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"></path><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"></path></svg>',
    "cpu": '<svg width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle; display:inline-block;"><rect x="4" y="4" width="16" height="16" rx="2" ry="2"></rect><rect x="9" y="9" width="6" height="6"></rect><line x1="9" y1="1" x2="9" y2="4"></line><line x1="15" y1="1" x2="15" y2="4"></line><line x1="9" y1="20" x2="9" y2="23"></line><line x1="15" y1="20" x2="15" y2="23"></line><line x1="20" y1="9" x2="23" y2="9"></line><line x1="20" y1="14" x2="23" y2="14"></line><line x1="1" y1="9" x2="4" y2="9"></line><line x1="1" y1="14" x2="4" y2="14"></line></svg>',
    "check-circle": '<svg width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle; display:inline-block;"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg>',
    "alert-triangle": '<svg width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle; display:inline-block;"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>',
    "x-circle": '<svg width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle; display:inline-block;"><circle cx="12" cy="12" r="10"></circle><line x1="15" y1="9" x2="9" y2="15"></line><line x1="9" y1="9" x2="15" y2="15"></line></svg>',
    "megaphone": '<svg width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle; display:inline-block;"><path d="M3 11l18-5v12L3 14v-3z"></path><path d="M11.6 16.8a3 3 0 1 1-5.8-1.6"></path></svg>',
    "file-text": '<svg width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle; display:inline-block;"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line><polyline points="10 9 9 9 8 9"></polyline></svg>',
    "briefcase": '<svg width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle; display:inline-block;"><rect x="2" y="7" width="20" height="14" rx="2" ry="2"></rect><path d="M16 21V5a2 2 0 0 0-2-2h-4a2 2 0 0 0-2 2v16"></path></svg>',
    "target": '<svg width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle; display:inline-block;"><circle cx="12" cy="12" r="10"></circle><circle cx="12" cy="12" r="6"></circle><circle cx="12" cy="2"></circle></svg>',
    "zap": '<svg width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle; display:inline-block;"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon></svg>',
    "award": '<svg width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle; display:inline-block;"><circle cx="12" cy="8" r="7"></circle><polyline points="8.21 13.89 7 23 12 20 17 23 15.79 13.88"></polyline></svg>',
    "mail": '<svg width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle; display:inline-block;"><path d="M4 4h16c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2V6c0-1.1.9-2 2-2z"></path><polyline points="22,6 12,13 2,6"></polyline></svg>',
    "download": '<svg width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle; display:inline-block;"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="7 10 12 15 17 10"></polyline><line x1="12" y1="15" x2="12" y2="3"></line></svg>',
    "refresh": '<svg width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle; display:inline-block;"><polyline points="23 4 23 10 17 10"></polyline><polyline points="1 20 1 14 7 14"></polyline><path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"></path></svg>',
    "trash": '<svg width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle; display:inline-block;"><polyline points="3 6 5 6 21 6"></polyline><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path></svg>',
    "sun": '<svg width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle; display:inline-block;"><circle cx="12" cy="12" r="5"></circle><line x1="12" y1="1" x2="12" y2="3"></line><line x1="12" y1="21" x2="12" y2="23"></line><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line><line x1="1" y1="12" x2="3" y2="12"></line><line x1="21" y1="12" x2="23" y2="12"></line><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line></svg>',
    "moon": '<svg width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle; display:inline-block;"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path></svg>',
    "users": '<svg width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle; display:inline-block;"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"></path><circle cx="9" cy="7" r="4"></circle><path d="M23 21v-2a4 4 0 0 0-3-3.87"></path><path d="M16 3.13a4 4 0 0 1 0 7.75"></path></svg>',
    "book": '<svg width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle; display:inline-block;"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"></path><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"></path></svg>',
    "sparkles": '<svg width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle; display:inline-block;"><path d="M12 2l2.4 7.2L22 12l-7.6 2.8L12 22l-2.4-7.2L2 12l7.6-2.8z"></path></svg>',
    "info": '<svg width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle; display:inline-block;"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line></svg>'
}

def get_svg_icon(name, size=18, color=None):
    c = color or "#ea580c"
    template = SVG_LOGOS.get(name, SVG_LOGOS["target"])
    return template.replace("{s}", str(size)).replace("{c}", c)

ALL_DEPARTMENTS = [
    "Core Engineering",
    "Platform Engineering",
    "Product Engineering",
    "Data & Artificial Intelligence",
    "Cloud Infrastructure",
    "Information Security",
    "Mobile Engineering",
    "Data Platform",
    "Applied AI Research",
    "Quality Assurance & Testing",
    "Embedded Hardware Systems",
    "Decentralized Web3 Systems",
    "Enterprise IT Operations"
]

# ---------------------------
# TECHNICAL JOB DESCRIPTION SHORTCUTS
# ---------------------------
TECHNICAL_ROLE_TEMPLATES = {
    "Full-Stack Web Developer": {
        "icon": "code",
        "category": "Software Engineering",
        "department": "Core Engineering",
        "skills": ["React", "Node.js", "TypeScript", "JavaScript", "Python", "SQL", "PostgreSQL", "Docker", "REST APIs", "Git", "HTML5", "Next.js"],
        "exp": "2-4 Years (Mid-Level)",
        "default_vacancies": 3,
        "text": """Position: Full-Stack Web Developer
Department: Core Engineering
Requirements & Responsibilities:
- 3+ years experience building scalable web applications with React, TypeScript, and modern JavaScript.
- Strong backend development expertise with Node.js, Express, or Python (FastAPI/Django).
- Relational database management using PostgreSQL, MySQL, and ORMs (Prisma, SQLAlchemy).
- Experience designing and consuming robust RESTful and GraphQL APIs.
- Familiarity with containerization (Docker), Git version control, and CI/CD pipelines.
- Solid understanding of web security, responsive CSS/Tailwind, and performance tuning."""
    },
    "Backend Software Engineer": {
        "icon": "server",
        "category": "Software Engineering",
        "department": "Platform Engineering",
        "skills": ["Python", "FastAPI", "Django", "PostgreSQL", "Redis", "Docker", "Kubernetes", "Microservices", "REST APIs", "Kafka", "Linux", "Git"],
        "exp": "3-5 Years (Mid-Senior)",
        "default_vacancies": 2,
        "text": """Position: Backend Software Engineer
Department: Platform Engineering
Requirements & Responsibilities:
- High proficiency in server-side development with Python (FastAPI, Django) or Go/Java.
- Deep knowledge of relational databases (PostgreSQL), schema design, and caching layers (Redis).
- Building distributed systems, event-driven architectures (Kafka/RabbitMQ), and microservices.
- Containerization and orchestration using Docker and Kubernetes.
- Experience with asynchronous programming, unit testing, and API security (OAuth2, JWT).
- Strong algorithmic problem-solving, system design, and database query optimization."""
    },
    "Frontend UI/UX Engineer": {
        "icon": "palette",
        "category": "Software Engineering",
        "department": "Product Engineering",
        "skills": ["React", "Next.js", "TypeScript", "Tailwind CSS", "Redux", "JavaScript", "HTML5", "CSS3", "Figma", "Vite", "REST APIs", "Git"],
        "exp": "2-4 Years (Mid-Level)",
        "default_vacancies": 2,
        "text": """Position: Frontend UI/UX Engineer
Department: Product Engineering
Requirements & Responsibilities:
- Expert-level knowledge of React 18+, Next.js, TypeScript, and modern ES6+ JavaScript.
- Mastery of responsive design, CSS3, Tailwind CSS, and component libraries.
- State management utilizing Redux Toolkit, Zustand, or React Context.
- Translating Figma prototypes into accessible (WCAG compliant) pixel-perfect interactive web UIs.
- Client-side performance optimization (Core Web Vitals, SSR, lazy loading).
- Automated frontend testing with Jest, React Testing Library, and Cypress."""
    },
    "Data Scientist & AI Specialist": {
        "icon": "brain",
        "category": "Data & AI",
        "department": "Data & Artificial Intelligence",
        "skills": ["Python", "Machine Learning", "PyTorch", "TensorFlow", "Scikit-Learn", "Pandas", "NumPy", "NLP", "SQL", "Deep Learning", "Analytics"],
        "exp": "2-5 Years (Mid-Level)",
        "default_vacancies": 2,
        "text": """Position: Data Scientist & AI Specialist
Department: Data & Artificial Intelligence
Requirements & Responsibilities:
- Strong background in machine learning algorithms, statistical modeling, and predictive analytics.
- Hands-on experience with Python, Pandas, NumPy, Scikit-Learn, and Deep Learning (PyTorch or TensorFlow).
- Natural Language Processing (NLP), text embedding models, and transformer architectures.
- Advanced SQL querying for data extraction, feature engineering, and exploratory data analysis.
- Model evaluation, cross-validation, hyperparameter tuning, and production inference deployment.
- Experience with data visualization tools (Matplotlib, Seaborn, Tableau)."""
    },
    "DevOps & Cloud Architect": {
        "icon": "cloud",
        "category": "Infrastructure & Security",
        "department": "Cloud Infrastructure",
        "skills": ["AWS", "Terraform", "Docker", "Kubernetes", "CI/CD", "Linux", "Prometheus", "Grafana", "Bash", "Python", "DevOps"],
        "exp": "3-6 Years (Senior)",
        "default_vacancies": 1,
        "text": """Position: DevOps & Cloud Infrastructure Architect
Department: Cloud Infrastructure
Requirements & Responsibilities:
- Managing cloud infrastructure on AWS (EC2, S3, RDS, ECS, EKS) or Google Cloud / Azure.
- Infrastructure as Code (IaC) utilizing Terraform or CloudFormation.
- Deploying and maintaining production Kubernetes clusters and Docker containers.
- Designing automated CI/CD deployment pipelines (GitHub Actions, GitLab CI, Jenkins).
- Infrastructure monitoring, log aggregation, and alerting with Prometheus, Grafana, and ELK.
- Strong Linux systems administration, bash scripting, and zero-trust security practices."""
    },
    "Cybersecurity & SecOps Analyst": {
        "icon": "shield",
        "category": "Infrastructure & Security",
        "department": "Information Security",
        "skills": ["Cybersecurity", "Penetration Testing", "SIEM", "Splunk", "Network Security", "OWASP", "Firewalls", "SOC", "Linux", "Cryptography", "Python"],
        "exp": "2-5 Years (Mid-Level)",
        "default_vacancies": 2,
        "text": """Position: Cybersecurity & SecOps Analyst
Department: Information Security
Requirements & Responsibilities:
- Monitoring security events, intrusion detection, and incident response via SIEM (Splunk, Sentinel).
- Vulnerability assessment and penetration testing following OWASP Top 10 guidelines.
- Network security protocols, packet analysis, and firewall configuration.
- Compliance knowledge (NIST, ISO 27001, SOC 2, GDPR).
- Security scripting with Python/Bash and automated vulnerability remediation.
- Threat intelligence, malware triage, and access control audit policies."""
    },
    "Mobile Application Developer": {
        "icon": "smartphone",
        "category": "Mobile & App",
        "department": "Mobile Engineering",
        "skills": ["Flutter", "Dart", "React Native", "Swift", "Kotlin", "REST APIs", "SQLite", "Firebase", "Mobile UI", "Git"],
        "exp": "2-4 Years (Mid-Level)",
        "default_vacancies": 2,
        "text": """Position: Mobile Application Developer (Cross-Platform)
Department: Mobile Engineering
Requirements & Responsibilities:
- Proven experience building high-performance mobile apps with Flutter (Dart) or React Native.
- Native mobile development exposure with iOS (Swift) or Android (Kotlin).
- State management with Riverpod, BLoC, or Redux.
- Integration of secure mobile REST APIs, offline caching (SQLite/Hive), and push notifications.
- Publishing and lifecycle management on Apple App Store and Google Play Console.
- Material Design & Cupertino guidelines implementation."""
    },
    "Data Engineer & Big Data Specialist": {
        "icon": "database",
        "category": "Data & AI",
        "department": "Data Platform",
        "skills": ["SQL", "Python", "Apache Spark", "Kafka", "Airflow", "Snowflake", "PostgreSQL", "AWS", "Git", "Analytics"],
        "exp": "3-5 Years (Mid-Level)",
        "default_vacancies": 2,
        "text": """Position: Data Engineer & Big Data Specialist
Department: Data Platform
Requirements & Responsibilities:
- Designing and maintaining robust ETL/ELT pipelines with Apache Airflow or Prefect.
- Distributed data processing utilizing Apache Spark, PySpark, and Delta Lake.
- Real-time streaming data architectures with Apache Kafka or AWS Kinesis.
- Modern cloud data warehousing (Snowflake, BigQuery, Databricks).
- Advanced SQL optimization, schema modeling, and data quality governance.
- Automated data transformations and version-controlled data pipelines."""
    },
    "Generative AI & LLM Engineer": {
        "icon": "cpu",
        "category": "Data & AI",
        "department": "Applied AI Research",
        "skills": ["Python", "LLMs", "LangChain", "RAG", "Transformers", "Vector Databases", "PyTorch", "FastAPI", "NLP", "Machine Learning"],
        "exp": "1-4 Years (Emerging Tech)",
        "default_vacancies": 3,
        "text": """Position: Generative AI & LLM Engineer
Department: Applied AI Research
Requirements & Responsibilities:
- Building production Retrieval-Augmented Generation (RAG) pipelines and LLM agents.
- Working with LangChain, LlamaIndex, OpenAI/Gemini APIs, and HuggingFace Transformers.
- Vector database indexing (ChromaDB, Pinecone, FAISS, Weaviate).
- Fine-tuning open-source models (Llama, Mistral) and quantization (LoRA, QLoRA).
- Prompt evaluation, hallucination mitigation, and model benchmarking.
- Python backend deployment of AI endpoints with FastAPI and async streaming."""
    },
    "QA Automation Engineer": {
        "icon": "target",
        "category": "Quality & Systems",
        "department": "Quality Assurance",
        "skills": ["Selenium", "Cypress", "Playwright", "Python", "PyTest", "API Testing", "Postman", "CI/CD", "JavaScript", "Automation"],
        "exp": "2-4 Years (Mid-Level)",
        "default_vacancies": 2,
        "text": """Position: QA Automation Engineer
Department: Quality Assurance & Engineering
Requirements & Responsibilities:
- Developing end-to-end test automation suites using Playwright, Cypress, or Selenium.
- Automated backend API testing with Python PyTest, Requests, and Postman.
- Integrating automated regression testing into CI/CD pipelines (GitHub Actions/Jenkins).
- Performance and load testing using JMeter or Locust.
- Defect tracking, test plan documentation in Jira, and cross-browser test verification.
- Test-driven development (TDD) and behavior-driven development (BDD/Cucumber)."""
    },
    "Embedded Systems & IoT Engineer": {
        "icon": "zap",
        "category": "Quality & Systems",
        "department": "Hardware Engineering",
        "skills": ["C", "C++", "RTOS", "Linux", "Microcontrollers", "ARM", "IoT", "Embedded", "Git"],
        "exp": "2-5 Years (Mid-Level)",
        "default_vacancies": 1,
        "text": """Position: Embedded Systems & IoT Engineer
Department: Embedded Hardware
Requirements & Responsibilities:
- Low-level embedded C/C++ firmware development for ARM Cortex and ESP32 architectures.
- Real-time operating systems (FreeRTOS, Zephyr).
- Hardware communication protocols (I2C, SPI, UART, CAN, BLE, MQTT).
- Board bring-up, hardware debugging with oscilloscopes, logic analyzers, and JTAG.
- Memory-constrained optimization, low-power sleep modes, and OTA firmware updates.
- Linux device drivers and kernel module exposure."""
    },
    "Blockchain & Systems Engineer": {
        "icon": "award",
        "category": "Software Engineering",
        "department": "Decentralized Systems",
        "skills": ["Solidity", "Rust", "Go", "Smart Contracts", "Cryptography", "Docker", "Linux", "Microservices", "Git"],
        "exp": "2-5 Years (Mid-Level)",
        "default_vacancies": 1,
        "text": """Position: Blockchain & Systems Engineer
Department: Decentralized Systems
Requirements & Responsibilities:
- Developing and auditing secure smart contracts in Solidity or Rust.
- Deep understanding of consensus mechanisms (PoS, PoW), cryptography, and distributed state machines.
- Building decentralized backends with Go, Rust, or Node.js.
- Smart contract development tooling with Hardhat, Foundry, and Truffle.
- Gas optimization and smart contract security vulnerability mitigation.
- Integration with Web3.js, Ethers.js, and decentralized storage."""
    }
}

def extract_skills_from_text(text: str) -> list:
    """Extract recognized technical and professional skills from text."""
    if not text:
        return []
    text_lower = text.lower()
    matched = []
    for skill in COMMON_SKILLS:
        pattern = r"\b" + re.escape(skill.lower()) + r"\b"
        if re.search(pattern, text_lower):
            matched.append(skill)
    return matched

def generate_eligibility_report(cand_text: str, vacancy: dict) -> dict:
    """Evaluate candidate resume against an announced job vacancy and generate rejection diagnostics and improvement suggestions."""
    cand_skills = extract_skills_from_text(cand_text)
    cand_skills_lower = [s.lower() for s in cand_skills]

    # Target skills
    if vacancy.get("required_skills"):
        req_skills = [s.strip() for s in str(vacancy["required_skills"]).split(",") if s.strip()]
    else:
        req_skills = extract_skills_from_text(vacancy.get("jd_text", ""))

    if not req_skills:
        req_skills = ["Python", "SQL", "Git", "REST APIs", "Problem Solving"]

    matched = []
    missing = []
    for s in req_skills:
        if s.lower() in cand_skills_lower:
            matched.append(s)
        else:
            missing.append(s)

    coverage = len(matched) / max(len(req_skills), 1)

    # Compute NLP scores against vacancy JD
    jd_txt = vacancy.get("jd_text", "")
    sim_score = 0.0
    prob_score = 0.0
    try:
        s_arr, p_arr = compute_similarity_and_probs(jd_txt, [cand_text or "candidate profile"])
        sim_score = round(float(s_arr[0]), 4)
        prob_score = round(float(p_arr[0]), 4)
    except Exception:
        sim_score = round(coverage * 0.7, 4)
        prob_score = round(coverage * 0.8, 4)

    # Eligibility determination
    is_eligible = (coverage >= 0.40 and sim_score >= 0.15) or (prob_score >= 0.45) or (coverage >= 0.60)

    # Specific Rejection Reasons
    reasons = []
    if not is_eligible:
        if missing:
            top_missing = missing[:5]
            reasons.append(f"Missing Core Technical Competencies: Your profile is missing key technologies required for this role: {', '.join(top_missing)}.")
        if coverage < 0.40:
            reasons.append(f"Technical Competency Gap: You currently possess {len(matched)} of {len(req_skills)} required technical competencies ({coverage*100:.1f}% match coverage, while this vacancy benchmark requires at least 40% match).")
        if sim_score < 0.15:
            reasons.append(f"Role Semantic Relevance Below Benchmark: Your document relevance score ({sim_score*100:.1f}%) is below the minimum qualifying ATS threshold of 15.0%.")
        if not reasons:
            reasons.append("Overall qualification match score is below the active candidate benchmark for this open position.")

    # Tailored Suggestions & What to Learn
    tech_roadmap = []
    for m in missing[:4]:
        m_lower = m.lower()
        if "docker" in m_lower or "kubernetes" in m_lower:
            tech_roadmap.append("Containerization & Orchestration: Learn Docker containerization, multi-stage builds, and Kubernetes pod deployments.")
        elif "react" in m_lower or "next" in m_lower:
            tech_roadmap.append("Modern Frontend Architecture: Master React 18+ hooks, Next.js App Router, SSR, and state management.")
        elif "fastapi" in m_lower or "django" in m_lower:
            tech_roadmap.append("Asynchronous Python APIs: Build high-performance REST APIs with FastAPI, async/await, and Pydantic validation.")
        elif "sql" in m_lower or "postgres" in m_lower:
            tech_roadmap.append("Relational Database Mastery: Practice PostgreSQL schema modeling, query indexing, and connection pooling.")
        elif "redis" in m_lower or "kafka" in m_lower:
            tech_roadmap.append("Distributed Caching & Streaming: Learn message queuing with Kafka/RabbitMQ and caching with Redis.")
        elif "aws" in m_lower or "terraform" in m_lower:
            tech_roadmap.append("Cloud & IaC Engineering: Learn AWS core primitives (EC2, S3, ECS) and automated provisioning with Terraform.")
        elif "machine learning" in m_lower or "pytorch" in m_lower or "tensorflow" in m_lower:
            tech_roadmap.append("Machine Learning & Deep Learning: Study PyTorch tensor operations, model training loops, and Scikit-Learn pipelines.")
        elif "security" in m_lower or "siem" in m_lower or "owasp" in m_lower:
            tech_roadmap.append("Cybersecurity Protocols: Study OWASP Top 10 vulnerabilities, SIEM log analysis, and network packet inspection.")
        elif "flutter" in m_lower or "dart" in m_lower:
            tech_roadmap.append("Cross-Platform Mobile Development: Practice Flutter widget composition and state management with Riverpod.")
        elif "selenium" in m_lower or "playwright" in m_lower or "cypress" in m_lower:
            tech_roadmap.append("End-to-End QA Automation: Build automated regression testing suites with Playwright or Cypress and integrate into CI/CD.")
        elif "solidity" in m_lower or "rust" in m_lower:
            tech_roadmap.append("Smart Contract & Systems Engineering: Write secure smart contracts in Solidity/Rust and audit for gas optimization.")
        elif "c++" in m_lower or "rtos" in m_lower or "embedded" in m_lower:
            tech_roadmap.append("Embedded Firmware Engineering: Practice FreeRTOS task scheduling, memory constraints, and hardware bus protocols.")
        else:
            tech_roadmap.append(f"{m} Core Fundamentals: Complete interactive tutorials and build mini-projects utilizing {m}.")

    if not tech_roadmap:
        tech_roadmap.append("Advanced System Design: Deepen knowledge of distributed caching, microservice communication, and database sharding.")
        tech_roadmap.append("CI/CD Automation: Set up automated test pipelines and GitHub Actions for continuous deployment.")

    # Project recommendation tailored to role
    role_t = vacancy.get("role_title", "").lower()
    if "data" in role_t or "ai" in role_t:
        project_rec = "Build an end-to-end Machine Learning pipeline that ingests tabular or text data, trains a model with PyTorch/Scikit-Learn, tracks experiments with MLflow, and serves predictions via FastAPI in a Docker container."
    elif "devops" in role_t or "cloud" in role_t or "security" in role_t:
        project_rec = "Deploy an automated multi-tier web application onto AWS using Terraform IaC, configure Prometheus/Grafana monitoring, and enforce automated vulnerability scans via GitHub Actions."
    elif "mobile" in role_t:
        project_rec = "Develop and publish a production-grade Flutter application featuring clean Riverpod state management, offline SQLite caching, and push notifications."
    else:
        project_rec = "Build a full-stack microservice application with Python FastAPI/Node.js backend, PostgreSQL database, React/Next.js frontend, and containerize the entire stack using Docker Compose."

    return {
        "is_eligible": is_eligible,
        "coverage": coverage,
        "sim_score": sim_score,
        "prob_score": prob_score,
        "matched_skills": matched,
        "missing_skills": missing,
        "rejection_reasons": reasons,
        "tech_roadmap": tech_roadmap,
        "recommended_project": project_rec
    }

# ---------------------------
# PASSWORD VALIDATION
# ---------------------------
def is_strong_password(pw: str) -> bool:
    if len(pw) < 8:
        return False
    if not re.search(r"[A-Z]", pw):
        return False
    if not re.search(r"[a-z]", pw):
        return False
    if not re.search(r"\d", pw):
        return False
    if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", pw):
        return False
    return True

# ---------------------------
# DYNAMIC SAFFRON, ORANGE, YELLOW THEME STYLING
# ---------------------------
is_dark = st.session_state.theme == "dark"

if is_dark:
    # DARK MODE: Rich Twilight Saffron, Fiery Orange & Golden Amber
    css_theme = """
    /* Dark Theme: Saffron, Orange & Golden Amber Twilight (Non-Black, Non-White) */

    
    /* Domain Filter Pills: High Visibility & Clear Distinction for ALL Pills in Dark Mode (NO WHITE!) */
    [data-testid="stPills"],
    [data-testid*="ButtonGroup"],
    [data-testid*="stSegmentedControl"],
    [data-baseweb="tag-list"] {
        display: flex !important;
        flex-wrap: wrap !important;
        gap: 8px !important;
        margin: 8px 0 12px 0 !important;
    }
    button[data-testid="stBaseButton-pills"],
    button[data-testid="stBaseButton-pillsActive"],
    [data-testid="stBaseButton-pills"],
    [data-testid="stBaseButton-pillsActive"],
    [data-testid*="ButtonGroup"] button,
    [data-testid*="ButtonGroup"] [role="radio"],
    .e1mwqyj910,
    .e1mwqyj911,
    [data-testid="stPills"] button,
    [data-baseweb="tag"] {
        background: linear-gradient(135deg, rgba(42, 21, 9, 0.96) 0%, rgba(26, 13, 5, 0.96) 100%) !important;
        background-color: rgba(42, 21, 9, 0.96) !important;
        color: #fed7aa !important;
        border: 1.5px solid rgba(251, 146, 60, 0.65) !important;
        border-radius: 20px !important;
        padding: 7px 18px !important;
        font-weight: 800 !important;
        font-size: 13px !important;
        box-shadow: 0 3px 12px rgba(0, 0, 0, 0.45) !important;
        opacity: 1 !important;
        visibility: visible !important;
    }
    button[data-testid="stBaseButton-pills"] *,
    button[data-testid="stBaseButton-pillsActive"] *,
    [data-testid="stBaseButton-pills"] *,
    [data-testid="stBaseButton-pillsActive"] *,
    [data-testid*="ButtonGroup"] button *,
    [data-testid*="ButtonGroup"] [role="radio"] *,
    .e1mwqyj910 *,
    .e1mwqyj911 *,
    [data-testid="stPills"] button *,
    [data-baseweb="tag"] * {
        color: #fed7aa !important;
        font-weight: 800 !important;
        opacity: 1 !important;
        visibility: visible !important;
    }
    button[data-testid="stBaseButton-pills"]:hover,
    [data-testid*="ButtonGroup"] button:hover,
    .e1mwqyj910:hover,
    [data-testid="stPills"] button:hover {
        background: linear-gradient(135deg, #c2410c 0%, #ea580c 100%) !important;
        background-color: #ea580c !important;
        border-color: #f97316 !important;
        color: #fef3c7 !important;
        transform: translateY(-1px) !important;
    }
    button[data-testid="stBaseButton-pills"]:hover *,
    [data-testid*="ButtonGroup"] button:hover *,
    .e1mwqyj910:hover *,
    [data-testid="stPills"] button:hover * {
        color: #fef3c7 !important;
    }
    button[data-testid="stBaseButton-pillsActive"],
    [data-testid="stBaseButton-pillsActive"],
    .e1mwqyj911,
    [data-testid*="ButtonGroup"] button[aria-checked="true"],
    [data-testid*="ButtonGroup"] [role="radio"][aria-checked="true"],
    button[data-testid="stBaseButton-pills"][aria-checked="true"],
    button[data-testid="stBaseButton-pills"][aria-selected="true"],
    [data-testid="stPills"] button[aria-checked="true"],
    [data-testid="stPills"] button[aria-selected="true"],
    [data-testid="stPills"] [aria-pressed="true"],
    [data-testid="stPills"] [data-selected="true"] {
        background: linear-gradient(135deg, #c2410c 0%, #ea580c 50%, #f59e0b 100%) !important;
        background-color: #ea580c !important;
        color: #fffaf5 !important;
        border: 2px solid #facc15 !important;
        box-shadow: 0 4px 18px rgba(234, 88, 12, 0.6) !important;
        font-weight: 800 !important;
    }
    button[data-testid="stBaseButton-pillsActive"] *,
    [data-testid="stBaseButton-pillsActive"] *,
    .e1mwqyj911 *,
    [data-testid*="ButtonGroup"] button[aria-checked="true"] *,
    [data-testid*="ButtonGroup"] [role="radio"][aria-checked="true"] *,
    [data-testid="stPills"] button[aria-checked="true"] *,
    [data-testid="stPills"] button[aria-selected="true"] *,
    [data-testid="stPills"] [aria-pressed="true"] * {
        color: #fffaf5 !important;
        font-weight: 800 !important;
    }

    /* Universal List Background Color throughout the Webapp (Dark Mode) */
    .profile-card,
    .glass-panel,
    [data-testid="stExpander"],
    .inbound-list-card,
    .doc-vault-card,
    .vacancy-list-card {
        background: linear-gradient(145deg, rgba(32, 16, 6, 0.94) 0%, rgba(42, 20, 8, 0.90) 50%, rgba(20, 10, 4, 0.96) 100%) !important;
        background-color: rgba(32, 16, 6, 0.94) !important;
        border: 1.5px solid rgba(251, 146, 60, 0.45) !important;
        border-radius: 14px !important;
        box-shadow: 0 6px 22px rgba(0, 0, 0, 0.4) !important;
    }

    /* Universal Button Theming in Dark Mode: Warm Espresso & Burnished Saffron (Non-White) */
    .stButton > button,
    .stButton > button:not([kind="primary"]),
    .stButton > button[kind="secondary"],
    button[data-testid="baseButton-secondary"],
    button[data-testid="stBaseButton-secondary"] {
        background: linear-gradient(135deg, rgba(38, 18, 7, 0.95) 0%, rgba(26, 12, 4, 0.95) 100%) !important;
        background-color: rgba(38, 18, 7, 0.95) !important;
        color: #fed7aa !important;
        border: 1.5px solid rgba(251, 146, 60, 0.5) !important;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.35) !important;
        font-weight: 800 !important;
    }
    .stButton > button:hover,
    .stButton > button:not([kind="primary"]):hover,
    .stButton > button[kind="secondary"]:hover,
    button[data-testid="baseButton-secondary"]:hover,
    button[data-testid="stBaseButton-secondary"]:hover {
        background: linear-gradient(135deg, #ea580c 0%, #f97316 50%, #facc15 100%) !important;
        background-color: #ea580c !important;
        color: #fef3c7 !important;
        border-color: #f97316 !important;
        box-shadow: 0 6px 20px rgba(249, 115, 22, 0.6) !important;
        transform: translateY(-1px) !important;
    }
    .stButton > button *,
    .stButton > button:not([kind="primary"]) *,
    .stButton > button[kind="secondary"] *,
    button[data-testid="baseButton-secondary"] *,
    button[data-testid="stBaseButton-secondary"] * {
        color: #fed7aa !important;
    }
    .stButton > button:hover *,
    .stButton > button:not([kind="primary"]):hover *,
    .stButton > button[kind="secondary"]:hover *,
    button[data-testid="baseButton-secondary"]:hover *,
    button[data-testid="stBaseButton-secondary"]:hover * {
        color: #fef3c7 !important;
    }
    [data-testid="stSidebar"] button,
    [data-testid="stSidebar"] .stButton > button {
        background: linear-gradient(135deg, rgba(38, 18, 7, 0.95) 0%, rgba(26, 12, 4, 0.95) 100%) !important;
        background-color: rgba(38, 18, 7, 0.95) !important;
        color: #fed7aa !important;
        border: 1.5px solid rgba(251, 146, 60, 0.5) !important;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.35) !important;
    }
    [data-testid="stSidebar"] button:hover,
    [data-testid="stSidebar"] .stButton > button:hover {
        background: linear-gradient(135deg, #ea580c 0%, #f97316 50%, #facc15 100%) !important;
        color: #fef3c7 !important;
        border-color: #f97316 !important;
    }
    [data-testid="stSidebar"] button *,
    [data-testid="stSidebar"] .stButton > button * {
        color: #fed7aa !important;
    }
    [data-testid="stSidebar"] button:hover *,
    [data-testid="stSidebar"] .stButton > button:hover * {
        color: #fef3c7 !important;
    }

    .stApp, [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #170b03 0%, #291305 25%, #231238 65%, #0f172a 100%) !important;
        background-attachment: fixed !important;
        color: #f8fafc !important;
    }
    [data-testid="stHeader"] {
        background: transparent !important;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #150902 0%, #201006 50%, #101628 100%) !important;
        border-right: 1px solid rgba(249, 115, 22, 0.35) !important;
        box-shadow: 4px 0 24px rgba(0, 0, 0, 0.45) !important;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4, [data-testid="stSidebar"] strong {
        color: #fef3c7 !important;
        opacity: 1 !important;
        font-weight: 800 !important;
    }
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] div, [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] * {
        color: #fed7aa !important;
        opacity: 1 !important;
        font-weight: 700 !important;
    }
    [data-testid="stSidebar"] [data-baseweb="tab"] {
        color: #fed7aa !important;
        opacity: 1 !important;
        font-weight: 800 !important;
    }
    [data-testid="stSidebar"] [data-testid="stExpander"] {
        display: none !important;
    }
    [data-testid="stSidebar"] input {
        color: #fef3c7 !important;
        background: rgba(22, 11, 4, 0.85) !important;
        border: 1.5px solid rgba(249, 115, 22, 0.6) !important;
        font-weight: 700 !important;
        opacity: 1 !important;
    }
    /* Assign Role & Selectbox visibility in sidebar */
    [data-testid="stSidebar"] [data-testid="stSelectbox"] label,
    [data-testid="stSidebar"] [data-testid="stSelectbox"] label p,
    [data-testid="stSidebar"] [data-testid="stSelectbox"] [data-testid="stWidgetLabel"] * {
        color: #fef3c7 !important;
        font-weight: 800 !important;
        font-size: 14px !important;
        opacity: 1 !important;
        display: block !important;
        visibility: visible !important;
    }
    [data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"],
    [data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"] > div {
        background: rgba(22, 11, 4, 0.95) !important;
        background-color: rgba(22, 11, 4, 0.95) !important;
        border: 1.5px solid rgba(249, 115, 22, 0.7) !important;
        border-radius: 8px !important;
        color: #fef3c7 !important;
    }
    [data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"] * {
        color: #fef3c7 !important;
        opacity: 1 !important;
        font-weight: 800 !important;
        font-size: 14px !important;
    }
    [data-testid="stSidebar"] [data-testid="stSelectbox"] svg {
        fill: #f97316 !important;
        color: #f97316 !important;
    }
    .smarthire-navbar {
        background: linear-gradient(90deg, rgba(38, 18, 7, 0.95) 0%, rgba(48, 25, 12, 0.90) 100%) !important;
        border: 1px solid rgba(251, 146, 60, 0.45) !important;
        box-shadow: 0 8px 32px rgba(234, 88, 12, 0.22) !important;
    }
    .brand-title-text {
        color: #fef3c7 !important;
    }
    .brand-version-pill {
        background: rgba(249, 115, 22, 0.2) !important;
        color: #fb923c !important;
        border: 1px solid rgba(249, 115, 22, 0.45) !important;
    }
    .status-badge-container {
        color: #fef3c7 !important;
        opacity: 1 !important;
        font-weight: 800 !important;
    }
    .status-badge-container span {
        opacity: 1 !important;
        color: #fef3c7 !important;
        font-weight: 800 !important;
    }
    .glass-panel {
        background: linear-gradient(145deg, rgba(38, 19, 7, 0.65) 0%, rgba(20, 15, 34, 0.75) 100%) !important;
        border: 1px solid rgba(249, 115, 22, 0.3) !important;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3) !important;
    }
    .profile-card {
        background: linear-gradient(145deg, rgba(40, 20, 8, 0.6) 0%, rgba(18, 14, 32, 0.75) 100%) !important;
        border: 1px solid rgba(249, 115, 22, 0.3) !important;
    }
    .profile-card:hover {
        border-color: #f97316 !important;
        box-shadow: 0 8px 28px rgba(249, 115, 22, 0.3) !important;
    }
    .profile-name-text {
        color: #fef3c7 !important;
        opacity: 1 !important;
        font-weight: 800 !important;
    }
    .profile-file-text {
        color: #fed7aa !important;
        opacity: 1 !important;
        font-weight: 700 !important;
    }
    .metric-grid-card {
        background: linear-gradient(145deg, rgba(46, 23, 9, 0.75) 0%, rgba(22, 12, 5, 0.85) 100%) !important;
        border: 1px solid rgba(249, 115, 22, 0.35) !important;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.25) !important;
    }
    .metric-header-title {
        color: #fed7aa !important;
        opacity: 1 !important;
        font-weight: 800 !important;
    }
    .metric-hero-val {
        color: #fef3c7 !important;
        opacity: 1 !important;
        font-weight: 800 !important;
    }
    .skill-badge-item {
        background: rgba(249, 115, 22, 0.16) !important;
        color: #fed7aa !important;
        border: 1px solid rgba(251, 146, 60, 0.4) !important;
        opacity: 1 !important;
        font-weight: 700 !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(24, 12, 4, 0.75) !important;
        border: 1px solid rgba(249, 115, 22, 0.3) !important;
    }
    .stTabs [data-baseweb="tab"] {
        color: #fed7aa !important;
        opacity: 1 !important;
        font-weight: 800 !important;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #ea580c 0%, #f97316 50%, #eab308 100%) !important;
        color: #fef3c7 !important;
        box-shadow: 0 4px 18px rgba(234, 88, 12, 0.5) !important;
        opacity: 1 !important;
    }
    [data-testid="stAppViewContainer"] h1, [data-testid="stAppViewContainer"] h2,
    [data-testid="stAppViewContainer"] h3, [data-testid="stAppViewContainer"] h4,
    [data-testid="stAppViewContainer"] h5, [data-testid="stAppViewContainer"] h6 {
        color: #fef3c7 !important;
        opacity: 1 !important;
        font-weight: 800 !important;
    }
    [data-testid="stAppViewContainer"] p, [data-testid="stMarkdownContainer"] p {
        color: #fed7aa !important;
        opacity: 1 !important;
        font-weight: 600 !important;
    }
    [data-testid="stAppViewContainer"] strong, [data-testid="stAppViewContainer"] b,
    [data-testid="stMarkdownContainer"] strong, [data-testid="stMarkdownContainer"] b {
        color: #fef3c7 !important;
        opacity: 1 !important;
        font-weight: 800 !important;
    }
    .stTextInput input, .stTextArea textarea {
        background: rgba(20, 10, 4, 0.85) !important;
        border: 1px solid rgba(249, 115, 22, 0.4) !important;
        color: #fef3c7 !important;
        opacity: 1 !important;
        font-weight: 700 !important;
    }
    div[data-testid="stSelectbox"] label,
    div[data-testid="stSelectbox"] label p,
    div[data-testid="stSelectbox"] [data-testid="stWidgetLabel"] * {
        color: #fef3c7 !important;
        opacity: 1 !important;
        font-weight: 800 !important;
        display: block !important;
        visibility: visible !important;
    }
    div[data-testid="stSelectbox"] [data-baseweb="select"],
    div[data-testid="stSelectbox"] [data-baseweb="select"] > div {
        background: rgba(20, 10, 4, 0.95) !important;
        background-color: rgba(20, 10, 4, 0.95) !important;
        border: 1.5px solid rgba(249, 115, 22, 0.6) !important;
        border-radius: 8px !important;
        color: #fef3c7 !important;
    }
    div[data-testid="stSelectbox"] [data-baseweb="select"] * {
        color: #fef3c7 !important;
        opacity: 1 !important;
        font-weight: 800 !important;
    }
    div[data-baseweb="popover"] {
        background: #1c0e05 !important;
    }
    div[data-baseweb="popover"] * {
        color: #fed7aa !important;
        opacity: 1 !important;
    }
    li[role="option"] {
        background: #1c0e05 !important;
        color: #fed7aa !important;
        opacity: 1 !important;
    }
    li[role="option"]:hover, li[aria-selected="true"] {
        background: linear-gradient(135deg, #c2410c, #ea580c) !important;
        color: #fef3c7 !important;
    }
    .stTextArea textarea::placeholder {
        color: #fed7aa !important;
        opacity: 1 !important;
        font-weight: 600 !important;
    }
    [data-testid="stCaptionContainer"] p, .stCaption {
        color: #fed7aa !important;
        opacity: 1 !important;
        font-weight: 700 !important;
    }
    label, [data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] span {
        color: #fef3c7 !important;
        opacity: 1 !important;
        font-weight: 800 !important;
    }
    [data-testid="stCheckbox"] label, [data-testid="stCheckbox"] p, [data-testid="stCheckbox"] span {
        color: #fef3c7 !important;
        opacity: 1 !important;
        font-weight: 700 !important;
    }
    [data-testid="stSlider"] *, [data-testid="stRadio"] * {
        color: #fef3c7 !important;
        opacity: 1 !important;
        font-weight: 700 !important;
    }
    .main p, .main span, .main label, .main div {
        opacity: 1 !important;
    }
    [data-testid="stFileUploader"] {
        background: rgba(30, 15, 6, 0.65) !important;
        border: 2px dashed rgba(249, 115, 22, 0.5) !important;
        border-radius: 12px !important;
        padding: 12px !important;
    }
    [data-testid="stFileUploader"] * {
        color: #fef3c7 !important;
        opacity: 1 !important;
    }
    [data-testid="stFileUploaderDropzoneInstructions"] * {
        color: #fed7aa !important;
        opacity: 1 !important;
    }
    [data-testid="stFileUploader"] small {
        color: #fed7aa !important;
        opacity: 1 !important;
        font-weight: 600 !important;
    }
    [data-testid="stFileUploader"] button {
        background: linear-gradient(135deg, #ea580c, #f59e0b) !important;
        color: #fef3c7 !important;
        border: none !important;
        font-weight: 800 !important;
    }
    /* Explicit high-visibility styling for uploaded file cards (JD & Resumes in Dark Mode) */
    [data-testid="stFileUploaderFile"],
    .stFileUploaderFile,
    ul[data-testid="stFileUploaderFile"] li,
    [data-testid="stFileUploader"] ul li,
    [data-testid="stFileUploader"] li {
        background: linear-gradient(135deg, #241207 0%, #1a0b04 100%) !important;
        background-color: #241207 !important;
        border: 1.5px solid #ea580c !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.4) !important;
        padding: 6px 12px !important;
        margin: 6px 0 !important;
    }
    [data-testid="stFileUploaderFileData"],
    .stFileUploaderFileData {
        background: transparent !important;
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0 !important;
        margin: 0 !important;
    }
    [data-testid="stFileUploaderFileName"],
    [data-testid="stFileUploaderFileName"] *,
    .stFileUploaderFileName,
    .stFileUploaderFileName *,
    .ewslnz911,
    [data-testid="stFileUploaderFileData"] div,
    [data-testid="stFileUploaderFileData"] span,
    [data-testid="stFileUploaderFileData"] p,
    [data-testid="stFileUploaderFileData"] *,
    [data-testid="stFileUploaderFile"] span,
    [data-testid="stFileUploaderFile"] div {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        font-weight: 800 !important;
        font-size: 13.5px !important;
        letter-spacing: 0.2px !important;
        opacity: 1 !important;
    }
    [data-testid="stFileUploaderFileData"] small,
    [data-testid="stFileUploaderFile"] small,
    [data-testid="stFileUploaderFileSize"],
    [data-testid="stFileUploaderFileSize"] *,
    .stFileUploaderFileSize,
    .e1xq3mgd0 {
        color: #fed7aa !important;
        -webkit-text-fill-color: #fed7aa !important;
        font-weight: 700 !important;
        font-size: 11.5px !important;
        opacity: 0.95 !important;
    }
    [data-testid="stFileUploaderFile"] svg,
    [data-testid="stFileUploaderFileData"] svg,
    .ewslnz914 svg {
        color: #f97316 !important;
        fill: #f97316 !important;
    }
    [data-testid="stFileUploaderDeleteBtn"] button,
    [data-testid="stFileUploaderDeleteBtn"] button *,
    [data-testid="stFileUploaderDeleteBtn"] svg,
    [data-testid="stFileUploaderDeleteBtn"] svg * {
        color: #f97316 !important;
        fill: #f97316 !important;
        stroke: #f97316 !important;
    }
    [data-testid="stExpander"] {
        background: rgba(30, 15, 6, 0.65) !important;
        border: 1px solid rgba(249, 115, 22, 0.35) !important;
        border-radius: 12px !important;
    }
    [data-testid="stExpander"] summary, [data-testid="stExpander"] summary * {
        color: #fef3c7 !important;
        opacity: 1 !important;
        font-weight: 800 !important;
    }
    [data-testid="stExpander"] [data-testid="stExpanderDetails"] * {
        color: #fed7aa !important;
        opacity: 1 !important;
        font-weight: 600 !important;
    }
    pre, code, [data-testid="stText"] {
        color: #fed7aa !important;
        background: rgba(18, 9, 3, 0.85) !important;
        border: 1px solid rgba(249, 115, 22, 0.35) !important;
        font-family: 'Inter', monospace !important;
        opacity: 1 !important;
        font-weight: 600 !important;
    }
    /* Tables: Suitable Warm Light Background (No harsh white glare) */
    [data-testid="stDataFrame"] {
        position: relative !important;
        border: 1.5px solid rgba(249, 115, 22, 0.4) !important;
        border-radius: 12px !important;
        background: #fdf5ea !important;
        overflow: hidden !important;
        box-shadow: 0 4px 18px rgba(0, 0, 0, 0.25) !important;
    }
    [data-testid="stDataFrame"]::after {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: #fbf3e6;
        mix-blend-mode: multiply;
        pointer-events: none;
        border-radius: 12px;
        z-index: 5;
    }
    [data-testid="stAlert"] {
        background: linear-gradient(135deg, rgba(38, 19, 8, 0.95) 0%, rgba(24, 12, 5, 0.95) 100%) !important;
        border: 1.5px solid rgba(249, 115, 22, 0.45) !important;
        border-radius: 10px !important;
        box-shadow: 0 4px 16px rgba(0,0,0,0.3) !important;
    }
    [data-testid="stAlert"] * {
        color: #fed7aa !important;
        opacity: 1 !important;
        font-weight: 700 !important;
    }
    [data-testid="stAlert"] strong, [data-testid="stAlert"] b {
        color: #fef3c7 !important;
        font-weight: 800 !important;
    }
    [data-testid="stAlert"] code {
        background: rgba(234, 88, 12, 0.25) !important;
        color: #fef3c7 !important;
        border: 1px solid rgba(249, 115, 22, 0.5) !important;
        padding: 2px 8px !important;
        border-radius: 5px !important;
        font-weight: 800 !important;
    }
    [data-testid="stMetricValue"] {
        color: #fef3c7 !important;
        font-weight: 800 !important;
    }
    [data-testid="stMetricLabel"] * {
        color: #fed7aa !important;
        font-weight: 700 !important;
    }
    .stepper-deck {
        background: rgba(30, 15, 6, 0.65) !important;
        border: 1px solid rgba(249, 115, 22, 0.35) !important;
    }
    .step-descriptor {
        color: #fed7aa !important;
    }
    [data-testid="stHorizontalBlock"]:has(.smarthire-navbar) [data-testid="column"]:last-child button,
    [data-testid="stHorizontalBlock"]:has(.smarthire-navbar) [data-testid="stColumn"]:last-child button,
    [data-testid="stHorizontalBlock"]:has(.smarthire-navbar) button,
    [data-testid="stHorizontalBlock"]:has(.smarthire-navbar) [data-testid="baseButton-secondary"],
    [data-testid="stHorizontalBlock"]:has(.smarthire-navbar) [data-testid="stBaseButton-secondary"],
    [data-testid="stHorizontalBlock"]:has(.smarthire-navbar) .stButton > button,
    [data-testid="column"]:has(.theme-toggle-box) button,
    [data-testid="stColumn"]:has(.theme-toggle-box) button,
    [data-testid="column"]:has(#theme-btn-marker) button,
    [data-testid="stColumn"]:has(#theme-btn-marker) button,
    div:has(> .theme-toggle-box) + div button,
    div:has(.theme-toggle-box) + div button,
    div:has(.theme-toggle-box) ~ div button,
    .theme-toggle-box .stButton > button,
    .theme-toggle-box button {
        background: linear-gradient(135deg, rgba(38, 18, 7, 0.95) 0%, rgba(26, 12, 4, 0.95) 100%) !important;
        background-color: rgba(38, 18, 7, 0.95) !important;
        color: #fed7aa !important;
        border: 1.5px solid rgba(251, 146, 60, 0.45) !important;
        box-shadow: 0 4px 18px rgba(0, 0, 0, 0.35) !important;
        border-radius: 8px !important;
    }
    [data-testid="stHorizontalBlock"]:has(.smarthire-navbar) button p,
    [data-testid="stHorizontalBlock"]:has(.smarthire-navbar) button span,
    .theme-toggle-box .stButton > button p,
    .theme-toggle-box .stButton > button span {
        color: #fed7aa !important;
        font-weight: 800 !important;
    }
    [data-testid="stHorizontalBlock"]:has(.smarthire-navbar) button:hover,
    [data-testid="stHorizontalBlock"]:has(.smarthire-navbar) [data-testid="baseButton-secondary"]:hover,
    [data-testid="stHorizontalBlock"]:has(.smarthire-navbar) [data-testid="stBaseButton-secondary"]:hover,
    .theme-toggle-box .stButton > button:hover {
        background: linear-gradient(135deg, #ea580c 0%, #f97316 50%, #facc15 100%) !important;
        background-color: #ea580c !important;
        color: #fef3c7 !important;
        border-color: rgba(255, 255, 255, 0.5) !important;
        box-shadow: 0 6px 24px rgba(249, 115, 22, 0.6) !important;
        transform: translateY(-1px) !important;
    }
    [data-testid="stHorizontalBlock"]:has(.smarthire-navbar) button:hover p,
    [data-testid="stHorizontalBlock"]:has(.smarthire-navbar) button:hover span,
    .theme-toggle-box .stButton > button:hover p,
    .theme-toggle-box .stButton > button:hover span {
        color: #fef3c7 !important;
    }
    """
else:
    # LIGHT MODE: Warm Saffron, Radiant Orange & Golden Amber Sunrise (High-Contrast, Ultra-Professional)
    css_theme = """
    /* Light Theme: Saffron & Sunlit Porcelain (WCAG AAA High-Contrast, Professional Creative Web Design) */

    
    /* Domain Filter Pills: High Visibility & Clear Distinction for ALL Pills in Light Mode (NO WHITE GLARE) */
    [data-testid="stPills"],
    [data-testid*="ButtonGroup"],
    [data-testid*="stSegmentedControl"],
    [data-baseweb="tag-list"] {
        display: flex !important;
        flex-wrap: wrap !important;
        gap: 8px !important;
        margin: 8px 0 12px 0 !important;
    }
    button[data-testid="stBaseButton-pills"],
    button[data-testid="stBaseButton-pillsActive"],
    [data-testid="stBaseButton-pills"],
    [data-testid="stBaseButton-pillsActive"],
    [data-testid*="ButtonGroup"] button,
    [data-testid*="ButtonGroup"] [role="radio"],
    .e1mwqyj910,
    .e1mwqyj911,
    [data-testid="stPills"] button,
    [data-baseweb="tag"] {
        background: linear-gradient(135deg, #fff7ed 0%, #fed7aa 100%) !important;
        background-color: #ffedd5 !important;
        color: #431407 !important;
        border: 1.5px solid rgba(234, 88, 12, 0.5) !important;
        border-radius: 20px !important;
        padding: 7px 18px !important;
        font-weight: 800 !important;
        font-size: 13px !important;
        box-shadow: 0 2px 8px rgba(234, 88, 12, 0.15) !important;
        opacity: 1 !important;
        visibility: visible !important;
    }
    button[data-testid="stBaseButton-pills"] *,
    button[data-testid="stBaseButton-pillsActive"] *,
    [data-testid="stBaseButton-pills"] *,
    [data-testid="stBaseButton-pillsActive"] *,
    [data-testid*="ButtonGroup"] button *,
    [data-testid*="ButtonGroup"] [role="radio"] *,
    .e1mwqyj910 *,
    .e1mwqyj911 *,
    [data-testid="stPills"] button *,
    [data-baseweb="tag"] * {
        color: #431407 !important;
        font-weight: 800 !important;
        opacity: 1 !important;
        visibility: visible !important;
    }
    button[data-testid="stBaseButton-pills"]:hover,
    [data-testid*="ButtonGroup"] button:hover,
    .e1mwqyj910:hover,
    [data-testid="stPills"] button:hover {
        background: #fed7aa !important;
        background-color: #fed7aa !important;
        border-color: #ea580c !important;
        color: #261308 !important;
        transform: translateY(-1px) !important;
    }
    button[data-testid="stBaseButton-pills"]:hover *,
    [data-testid*="ButtonGroup"] button:hover *,
    .e1mwqyj910:hover *,
    [data-testid="stPills"] button:hover * {
        color: #261308 !important;
    }
    button[data-testid="stBaseButton-pillsActive"],
    [data-testid="stBaseButton-pillsActive"],
    .e1mwqyj911,
    [data-testid*="ButtonGroup"] button[aria-checked="true"],
    [data-testid*="ButtonGroup"] [role="radio"][aria-checked="true"],
    button[data-testid="stBaseButton-pills"][aria-checked="true"],
    button[data-testid="stBaseButton-pills"][aria-selected="true"],
    [data-testid="stPills"] button[aria-checked="true"],
    [data-testid="stPills"] button[aria-selected="true"] {
        background: linear-gradient(135deg, #ea580c 0%, #c2410c 100%) !important;
        background-color: #ea580c !important;
        color: #fffaf5 !important;
        border: 2px solid #ea580c !important;
        box-shadow: 0 4px 14px rgba(234, 88, 12, 0.35) !important;
        font-weight: 800 !important;
    }
    button[data-testid="stBaseButton-pillsActive"] *,
    [data-testid="stBaseButton-pillsActive"] *,
    .e1mwqyj911 *,
    [data-testid*="ButtonGroup"] button[aria-checked="true"] *,
    [data-testid*="ButtonGroup"] [role="radio"][aria-checked="true"] *,
    [data-testid="stPills"] button[aria-checked="true"] *,
    [data-testid="stPills"] button[aria-selected="true"] * {
        color: #fffaf5 !important;
        font-weight: 800 !important;
    }

    /* Universal List Background Color throughout the Webapp (Light Mode) */
    .profile-card,
    .glass-panel,
    [data-testid="stExpander"],
    .inbound-list-card,
    .doc-vault-card,
    .vacancy-list-card {
        background: linear-gradient(145deg, #fffbf5 0%, #fff3e2 100%) !important;
        background-color: #fffbf5 !important;
        border: 1.5px solid rgba(234, 88, 12, 0.35) !important;
        border-radius: 14px !important;
        box-shadow: 0 4px 18px rgba(234, 88, 12, 0.08) !important;
    }

    /* Universal Button Theming in Light Mode: Soft Warm Peach/Alabaster (Non-White) */
    .stButton > button,
    .stButton > button:not([kind="primary"]),
    .stButton > button[kind="secondary"],
    button[data-testid="baseButton-secondary"],
    button[data-testid="stBaseButton-secondary"] {
        background: linear-gradient(135deg, #ffedd5 0%, #fed7aa 100%) !important;
        background-color: #ffedd5 !important;
        color: #431407 !important;
        border: 1.5px solid rgba(234, 88, 12, 0.45) !important;
        box-shadow: 0 2px 8px rgba(234, 88, 12, 0.12) !important;
        font-weight: 800 !important;
    }
    .stButton > button:hover,
    .stButton > button:not([kind="primary"]):hover,
    .stButton > button[kind="secondary"]:hover,
    button[data-testid="baseButton-secondary"]:hover,
    button[data-testid="stBaseButton-secondary"]:hover {
        background: linear-gradient(135deg, #fed7aa 0%, #fdba74 100%) !important;
        background-color: #fed7aa !important;
        color: #261308 !important;
        border-color: #ea580c !important;
        box-shadow: 0 4px 14px rgba(234, 88, 12, 0.25) !important;
        transform: translateY(-1px) !important;
    }
    .stButton > button *,
    .stButton > button:not([kind="primary"]) *,
    .stButton > button[kind="secondary"] *,
    button[data-testid="baseButton-secondary"] *,
    button[data-testid="stBaseButton-secondary"] * {
        color: #431407 !important;
        font-weight: 800 !important;
    }
    .stButton > button:hover *,
    .stButton > button:not([kind="primary"]):hover *,
    .stButton > button[kind="secondary"]:hover *,
    button[data-testid="baseButton-secondary"]:hover *,
    button[data-testid="stBaseButton-secondary"]:hover * {
        color: #261308 !important;
    }
    [data-testid="stSidebar"] button,
    [data-testid="stSidebar"] .stButton > button {
        background: linear-gradient(135deg, #ffedd5 0%, #fed7aa 100%) !important;
        background-color: #ffedd5 !important;
        color: #431407 !important;
        border: 1.5px solid rgba(234, 88, 12, 0.45) !important;
    }
    [data-testid="stSidebar"] button *,
    [data-testid="stSidebar"] .stButton > button * {
        color: #431407 !important;
        font-weight: 800 !important;
    }

    .stApp, [data-testid="stAppViewContainer"] {
        background: 
            radial-gradient(ellipse at 85% 10%, rgba(254, 215, 170, 0.35) 0%, transparent 50%),
            radial-gradient(ellipse at 15% 20%, rgba(255, 237, 213, 0.45) 0%, transparent 45%),
            radial-gradient(ellipse at 60% 85%, rgba(254, 243, 199, 0.3) 0%, transparent 50%),
            linear-gradient(180deg, #fdfbf7 0%, #faf6f0 50%, #f6f0e6 100%) !important;
        background-attachment: fixed !important;
        color: #0f172a !important;
    }
    [data-testid="stHeader"] {
        background: transparent !important;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #fffaf5 0%, #fef4ea 50%, #fdebd7 100%) !important;
        border-right: 1px solid rgba(234, 88, 12, 0.22) !important;
        box-shadow: 4px 0 24px rgba(234, 88, 12, 0.08) !important;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4, [data-testid="stSidebar"] strong {
        color: #0f172a !important;
        opacity: 1 !important;
        font-weight: 800 !important;
    }
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] div, [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] * {
        color: #334155 !important;
        opacity: 1 !important;
        font-weight: 600 !important;
    }
    [data-testid="stSidebar"] [data-baseweb="tab"] {
        color: #64748b !important;
        opacity: 1 !important;
        font-weight: 700 !important;
    }
    [data-testid="stSidebar"] [data-baseweb="tab"][aria-selected="true"] {
        color: #c2410c !important;
        font-weight: 800 !important;
    }
    [data-testid="stSidebar"] [data-testid="stExpander"] {
        display: none !important;
    }
    [data-testid="stSidebar"] input {
        color: #0f172a !important;
        background: #fffaf5 !important;
        border: 1.5px solid rgba(234, 88, 12, 0.4) !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        opacity: 1 !important;
    }
    [data-testid="stSidebar"] input:focus {
        border-color: #ea580c !important;
        box-shadow: 0 0 0 3px rgba(234, 88, 12, 0.15) !important;
    }
    /* Assign Role & Selectbox visibility in sidebar (Light Mode) */
    [data-testid="stSidebar"] [data-testid="stSelectbox"] label,
    [data-testid="stSidebar"] [data-testid="stSelectbox"] label p,
    [data-testid="stSidebar"] [data-testid="stSelectbox"] [data-testid="stWidgetLabel"] * {
        color: #0f172a !important;
        font-weight: 800 !important;
        font-size: 14px !important;
        opacity: 1 !important;
        display: block !important;
        visibility: visible !important;
    }
    [data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"],
    [data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"] > div {
        background: #fffdfa !important;
        background-color: #fffdfa !important;
        border: 1.5px solid rgba(234, 88, 12, 0.4) !important;
        border-radius: 8px !important;
        color: #0f172a !important;
    }
    [data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"] * {
        color: #0f172a !important;
        opacity: 1 !important;
        font-weight: 700 !important;
        font-size: 14px !important;
    }
    [data-testid="stSidebar"] [data-testid="stSelectbox"] svg {
        fill: #ea580c !important;
        color: #ea580c !important;
    }
    .smarthire-navbar {
        background: rgba(255, 250, 243, 0.94) !important;
        border: 1.5px solid rgba(234, 88, 12, 0.28) !important;
        box-shadow: 0 4px 20px rgba(234, 88, 12, 0.08), 0 1px 3px rgba(0, 0, 0, 0.03) !important;
        backdrop-filter: blur(16px) !important;
    }
    .brand-title-text {
        color: #0f172a !important;
        font-weight: 800 !important;
    }
    .brand-version-pill {
        background: rgba(234, 88, 12, 0.12) !important;
        color: #c2410c !important;
        border: 1.5px solid rgba(234, 88, 12, 0.3) !important;
        font-weight: 800 !important;
    }
    .status-badge-container {
        color: #334155 !important;
        opacity: 1 !important;
        font-weight: 700 !important;
    }
    .status-badge-container span {
        opacity: 1 !important;
        color: #0f172a !important;
        font-weight: 800 !important;
    }
    .glass-panel {
        background: linear-gradient(145deg, #fffaf5 0%, #fff1e0 100%) !important;
        border: 1.5px solid rgba(234, 88, 12, 0.28) !important;
        box-shadow: 0 10px 30px rgba(234, 88, 12, 0.08), 0 1px 4px rgba(0,0,0,0.03) !important;
        color: #0f172a !important;
    }
    .profile-card {
        background: linear-gradient(145deg, #fffdfa 0%, #fff5ea 100%) !important;
        border: 1.5px solid rgba(234, 88, 12, 0.28) !important;
        box-shadow: 0 4px 16px rgba(234, 88, 12, 0.08), 0 1px 3px rgba(0,0,0,0.03) !important;
    }
    .profile-card:hover {
        border-color: #ea580c !important;
        box-shadow: 0 12px 32px rgba(234, 88, 12, 0.16), 0 2px 6px rgba(0,0,0,0.04) !important;
    }
    .profile-name-text {
        color: #0f172a !important;
        opacity: 1 !important;
        font-weight: 800 !important;
    }
    .profile-file-text {
        color: #64748b !important;
        opacity: 1 !important;
        font-weight: 600 !important;
    }
    .metric-grid-card {
        background: linear-gradient(145deg, #fffdfa 0%, #fff3e2 100%) !important;
        border: 1.5px solid rgba(234, 88, 12, 0.25) !important;
        box-shadow: 0 4px 18px rgba(234, 88, 12, 0.08), 0 1px 3px rgba(0,0,0,0.02) !important;
    }
    .metric-header-title {
        color: #7c2d12 !important;
        opacity: 1 !important;
        font-weight: 800 !important;
    }
    .metric-hero-val {
        color: #0f172a !important;
        opacity: 1 !important;
        font-weight: 800 !important;
    }
    .skill-badge-item {
        background: rgba(234, 88, 12, 0.12) !important;
        color: #9a3412 !important;
        border: 1.5px solid rgba(234, 88, 12, 0.38) !important;
        opacity: 1 !important;
        font-weight: 800 !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255, 247, 237, 0.95) !important;
        border: 1.5px solid rgba(234, 88, 12, 0.25) !important;
        box-shadow: 0 2px 8px rgba(234, 88, 12, 0.08) !important;
    }
    .stTabs [data-baseweb="tab"] {
        color: #64748b !important;
        opacity: 1 !important;
        font-weight: 700 !important;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #ea580c 0%, #f97316 50%, #f59e0b 100%) !important;
        color: #fef3c7 !important;
        box-shadow: 0 4px 16px rgba(234, 88, 12, 0.35) !important;
        opacity: 1 !important;
        font-weight: 800 !important;
    }
    [data-testid="stAppViewContainer"] h1, [data-testid="stAppViewContainer"] h2,
    [data-testid="stAppViewContainer"] h3, [data-testid="stAppViewContainer"] h4,
    [data-testid="stAppViewContainer"] h5, [data-testid="stAppViewContainer"] h6 {
        color: #0f172a !important;
        opacity: 1 !important;
        font-weight: 800 !important;
        letter-spacing: -0.3px !important;
    }
    [data-testid="stAppViewContainer"] p, [data-testid="stMarkdownContainer"] p {
        color: #334155 !important;
        opacity: 1 !important;
        font-weight: 500 !important;
        line-height: 1.6 !important;
    }
    [data-testid="stAppViewContainer"] strong, [data-testid="stAppViewContainer"] b,
    [data-testid="stMarkdownContainer"] strong, [data-testid="stMarkdownContainer"] b {
        color: #0f172a !important;
        opacity: 1 !important;
        font-weight: 800 !important;
    }
    .stTextInput input, .stTextArea textarea {
        background: #fffdfa !important;
        border: 1.5px solid rgba(234, 88, 12, 0.35) !important;
        border-radius: 8px !important;
        color: #0f172a !important;
        opacity: 1 !important;
        font-weight: 600 !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.02) !important;
    }
    .stTextInput input:focus, .stTextArea textarea:focus {
        border-color: #ea580c !important;
        box-shadow: 0 0 0 3px rgba(234, 88, 12, 0.15) !important;
    }
    div[data-testid="stSelectbox"] label,
    div[data-testid="stSelectbox"] label p,
    div[data-testid="stSelectbox"] [data-testid="stWidgetLabel"] * {
        color: #0f172a !important;
        opacity: 1 !important;
        font-weight: 800 !important;
        display: block !important;
        visibility: visible !important;
    }
    div[data-testid="stSelectbox"] [data-baseweb="select"],
    div[data-testid="stSelectbox"] [data-baseweb="select"] > div {
        background: #fffdfa !important;
        background-color: #fffdfa !important;
        border: 1.5px solid rgba(234, 88, 12, 0.35) !important;
        border-radius: 8px !important;
        color: #0f172a !important;
    }
    div[data-testid="stSelectbox"] [data-baseweb="select"] * {
        color: #0f172a !important;
        opacity: 1 !important;
        font-weight: 700 !important;
    }
    div[data-testid="stSelectbox"] svg {
        fill: #ea580c !important;
        color: #ea580c !important;
    }
    div[data-baseweb="popover"] {
        background: #fffdfa !important;
        border: 1.5px solid rgba(234, 88, 12, 0.28) !important;
        box-shadow: 0 12px 32px rgba(0, 0, 0, 0.12) !important;
        border-radius: 10px !important;
    }
    div[data-baseweb="popover"] * {
        color: #0f172a !important;
        opacity: 1 !important;
    }
    li[role="option"] {
        background: #fffdfa !important;
        color: #0f172a !important;
        opacity: 1 !important;
    }
    li[role="option"]:hover, li[aria-selected="true"] {
        background: linear-gradient(135deg, #fff7ed, #ffedd5) !important;
        color: #c2410c !important;
        font-weight: 800 !important;
    }
    .stTextArea textarea::placeholder, .stTextInput input::placeholder {
        color: #94a3b8 !important;
        opacity: 1 !important;
        font-weight: 500 !important;
    }
    [data-testid="stCaptionContainer"] p, .stCaption {
        color: #64748b !important;
        opacity: 1 !important;
        font-weight: 600 !important;
    }
    label, [data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] span {
        color: #0f172a !important;
        opacity: 1 !important;
        font-weight: 800 !important;
    }
    [data-testid="stCheckbox"] label, [data-testid="stCheckbox"] p, [data-testid="stCheckbox"] span {
        color: #1e293b !important;
        opacity: 1 !important;
        font-weight: 700 !important;
    }
    [data-testid="stSlider"] *, [data-testid="stRadio"] * {
        color: #0f172a !important;
        opacity: 1 !important;
        font-weight: 700 !important;
    }
    .main p, .main span, .main label, .main div {
        opacity: 1 !important;
    }
    [data-testid="stFileUploader"] {
        background: linear-gradient(135deg, #fffaf5 0%, #ffedd5 100%) !important;
        border: 2px dashed rgba(234, 88, 12, 0.5) !important;
        border-radius: 12px !important;
        padding: 12px !important;
    }
    [data-testid="stFileUploader"] * {
        color: #0f172a !important;
    }
    [data-testid="stFileUploaderDropzoneInstructions"] * {
        color: #475569 !important;
        font-weight: 600 !important;
    }
    [data-testid="stFileUploader"] small {
        color: #7c2d12 !important;
        font-weight: 700 !important;
    }
    [data-testid="stFileUploader"] button {
        background: linear-gradient(135deg, #ea580c, #f59e0b) !important;
        color: #fef3c7 !important;
        border: none !important;
        font-weight: 800 !important;
    }
    /* Explicit high-visibility styling for uploaded file cards (JD & Resumes in Light Mode) */
    [data-testid="stFileUploaderFile"],
    [data-testid="stFileUploaderFileData"],
    ul[data-testid="stFileUploaderFile"] li,
    [data-testid="stFileUploader"] ul li {
        background: linear-gradient(135deg, #ffedd5 0%, #fed7aa 100%) !important;
        background-color: #ffedd5 !important;
        border: 1.5px solid #ea580c !important;
        border-radius: 8px !important;
        box-shadow: 0 2px 8px rgba(234, 88, 12, 0.15) !important;
        padding: 6px 12px !important;
        margin: 6px 0 !important;
    }
    [data-testid="stFileUploaderFileName"],
    [data-testid="stFileUploaderFileData"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stFileUploaderFile"] span {
        color: #0f172a !important;
        font-weight: 800 !important;
        font-size: 13.5px !important;
        letter-spacing: 0.2px !important;
    }
    [data-testid="stFileUploaderFileSize"] {
        color: #7c2d12 !important;
        font-weight: 800 !important;
        font-size: 11.5px !important;
    }
    [data-testid="stFileUploaderDeleteBtn"] button,
    [data-testid="stFileUploaderDeleteBtn"] svg {
        color: #ea580c !important;
        fill: #ea580c !important;
    }
    [data-testid="stExpander"] {
        background: linear-gradient(145deg, #fffaf5 0%, #fff1e0 100%) !important;
        border: 1.5px solid rgba(234, 88, 12, 0.28) !important;
        border-radius: 12px !important;
        box-shadow: 0 2px 10px rgba(234, 88, 12, 0.06) !important;
    }
    [data-testid="stExpander"] summary, [data-testid="stExpander"] summary * {
        color: #0f172a !important;
        opacity: 1 !important;
        font-weight: 800 !important;
    }
    [data-testid="stExpander"] [data-testid="stExpanderDetails"] * {
        color: #1e293b !important;
        opacity: 1 !important;
        font-weight: 600 !important;
    }
    pre, code, [data-testid="stText"] {
        color: #9a3412 !important;
        background: #fff7ed !important;
        border: 1.5px solid rgba(234, 88, 12, 0.3) !important;
        font-family: 'Inter', monospace !important;
        opacity: 1 !important;
        font-weight: 600 !important;
    }
    /* Tables: Suitable Warm Light Background (No harsh white glare) */
    [data-testid="stDataFrame"] {
        position: relative !important;
        border: 1.5px solid rgba(234, 88, 12, 0.3) !important;
        border-radius: 12px !important;
        background: #fdf5ea !important;
        overflow: hidden !important;
        box-shadow: 0 2px 10px rgba(234, 88, 12, 0.08) !important;
    }
    [data-testid="stDataFrame"]::after {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: #fbf3e6;
        mix-blend-mode: multiply;
        pointer-events: none;
        border-radius: 12px;
        z-index: 5;
    }
    [data-testid="stAlert"] {
        background: linear-gradient(135deg, #fff7ed 0%, #ffedd5 100%) !important;
        border: 1.5px solid rgba(234, 88, 12, 0.45) !important;
        border-radius: 10px !important;
        color: #431407 !important;
        box-shadow: 0 2px 10px rgba(234, 88, 12, 0.08) !important;
    }
    [data-testid="stAlert"] * {
        color: #431407 !important;
        opacity: 1 !important;
        font-weight: 700 !important;
    }
    [data-testid="stAlert"] strong, [data-testid="stAlert"] b {
        color: #0f172a !important;
        font-weight: 800 !important;
    }
    [data-testid="stAlert"] code {
        background: #fed7aa !important;
        color: #7c2d12 !important;
        border: 1px solid #ea580c !important;
        padding: 2px 8px !important;
        border-radius: 5px !important;
        font-weight: 800 !important;
    }
    [data-testid="stMetricValue"] {
        color: #0f172a !important;
        font-weight: 800 !important;
    }
    [data-testid="stMetricLabel"] * {
        color: #64748b !important;
        font-weight: 700 !important;
    }
    .stepper-deck {
        background: linear-gradient(135deg, #fff7ed 0%, #ffedd5 100%) !important;
        border: 1.5px solid rgba(234, 88, 12, 0.3) !important;
        box-shadow: 0 4px 16px rgba(234, 88, 12, 0.06) !important;
    }
    .step-descriptor {
        color: #431407 !important;
        font-weight: 800 !important;
    }
    [data-testid="stHorizontalBlock"]:has(.smarthire-navbar) [data-testid="stColumn"]:last-child button,
    [data-testid="stHorizontalBlock"]:has(.smarthire-navbar) button,
    [data-testid="stHorizontalBlock"]:has(.smarthire-navbar) [data-testid="stBaseButton-secondary"],
    [data-testid="stHorizontalBlock"]:has(.smarthire-navbar) .stButton > button,
    [data-testid="stColumn"]:has(.theme-toggle-box) button,
    [data-testid="stColumn"]:has(#theme-btn-marker) button,
    div:has(> .theme-toggle-box) + div button,
    div:has(.theme-toggle-box) + div button,
    div:has(.theme-toggle-box) ~ div button,
    .theme-toggle-box .stButton > button,
    .theme-toggle-box button {
        background: linear-gradient(135deg, #ffedd5 0%, #fed7aa 100%) !important;
        background-color: #ffedd5 !important;
        color: #7c2d12 !important;
        border: 1.5px solid #ea580c !important;
        box-shadow: 0 2px 10px rgba(234, 88, 12, 0.16) !important;
        border-radius: 8px !important;
    }
    [data-testid="stHorizontalBlock"]:has(.smarthire-navbar) button p,
    [data-testid="stHorizontalBlock"]:has(.smarthire-navbar) button span,
    .theme-toggle-box .stButton > button p {
        color: #7c2d12 !important;
        font-weight: 800 !important;
    }
    [data-testid="stHorizontalBlock"]:has(.smarthire-navbar) button:hover,
    [data-testid="stHorizontalBlock"]:has(.smarthire-navbar) [data-testid="stBaseButton-secondary"]:hover,
    .theme-toggle-box .stButton > button:hover {
        background: linear-gradient(135deg, #fed7aa 0%, #fde68a 100%) !important;
        background-color: #fed7aa !important;
        color: #431407 !important;
        border-color: #ea580c !important;
        box-shadow: 0 4px 14px rgba(234, 88, 12, 0.25) !important;
    }
    [data-testid="stHorizontalBlock"]:has(.smarthire-navbar) button:hover p,
    .theme-toggle-box .stButton > button:hover p {
        color: #431407 !important;
    }
    """

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700;800&display=swap');
    
    html, body, h1, h2, h3, h4, h5, h6, p, label, button, input, textarea {{
        font-family: 'Plus Jakarta Sans', 'Inter', -apple-system, sans-serif;
        -webkit-font-smoothing: antialiased;
    }}
    /* Prevent icon ligature breakage: preserve Material Symbols font */
    [data-testid="stIconMaterial"], .material-symbols-rounded, .material-symbols-outlined,
    [data-testid="stExpanderToggleIcon"], [data-testid="stExpander"] summary span:first-child,
    [data-testid="stExpander"] summary [data-testid="stIconMaterial"] {{
        font-family: "Material Symbols Rounded", "Material Symbols Outlined", "Material Icons" !important;
    }}

    /* Suppress "Press Enter to apply" instruction badge on password and text inputs */
    [data-testid="InputInstructions"],
    div[data-testid="InputInstructions"],
    div[data-testid="InputInstructions"] *,
    .stTextInput [data-testid="InputInstructions"],
    .stTextInput [data-testid="InputInstructions"] *,
    .stNumberInput [data-testid="InputInstructions"],
    .stNumberInput [data-testid="InputInstructions"] * {{
        display: none !important;
        visibility: hidden !important;
        opacity: 0 !important;
        height: 0px !important;
        min-height: 0px !important;
        max-height: 0px !important;
        margin: 0px !important;
        padding: 0px !important;
        pointer-events: none !important;
        overflow: hidden !important;
    }}
    
    {css_theme}
    
    /* Top Brand Navigation Bar */
    .smarthire-navbar {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-radius: 16px;
        padding: 16px 26px;
        margin-bottom: 0px;
        backdrop-filter: blur(14px);
    }}
    .brand-group {{
        display: flex;
        align-items: center;
        gap: 14px;
    }}
    .brand-logo-mark {{
        background: linear-gradient(135deg, #f97316 0%, #ea580c 50%, #facc15 100%);
        color: #fef3c7;
        font-weight: 800;
        font-size: 17px;
        letter-spacing: -0.5px;
        padding: 8px 14px;
        border-radius: 12px;
        box-shadow: 0 4px 16px rgba(249, 115, 22, 0.45);
        border: 1px solid rgba(255, 255, 255, 0.25);
    }}
    .brand-title-text {{
        font-size: 22px;
        font-weight: 800;
        letter-spacing: -0.5px;
    }}
    .brand-version-pill {{
        font-size: 11px;
        font-weight: 700;
        padding: 3px 9px;
        border-radius: 8px;
        margin-left: 8px;
    }}
    .status-badge-container {{
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 13px;
        font-weight: 700;
    }}
    .status-pulse-dot {{
        width: 9px;
        height: 9px;
        background: #f59e0b;
        border-radius: 50%;
        box-shadow: 0 0 10px #f59e0b;
    }}
    
    /* Section Glass Cards */
    .glass-panel {{
        border-radius: 16px;
        padding: 22px;
        margin-bottom: 20px;
        backdrop-filter: blur(12px);
    }}
    
    /* Tab System Styling */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 12px !important;
        padding: 8px 12px !important;
        border-radius: 14px !important;
    }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 10px !important;
        font-weight: 700 !important;
        font-size: 14px !important;
        padding: 10px 22px !important;
        transition: all 0.25s ease !important;
        border: 1px solid transparent !important;
    }}
    
    /* Metric Cards Grid */
    .metric-grid-card {{
        border-radius: 14px;
        padding: 18px;
        text-align: center;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }}
    .metric-grid-card:hover {{
        transform: translateY(-2px);
    }}
    .metric-header-title {{
        font-size: 11px;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 6px;
    }}
    .metric-hero-val {{
        font-size: 28px;
        font-weight: 800;
        letter-spacing: -0.5px;
    }}
    
    /* Dynamic Candidate Profile Cards */
    .profile-card {{
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 16px;
        transition: all 0.25s ease;
    }}
    .profile-topbar {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 12px;
    }}
    .profile-name-text {{
        font-size: 18px;
        font-weight: 800;
        letter-spacing: -0.3px;
    }}
    .profile-file-text {{
        font-size: 12px;
        font-weight: 600;
    }}
    
    /* Status Badges */
    .status-pill-recommended {{
        background: rgba(16, 185, 129, 0.2);
        color: #10b981;
        font-size: 11px;
        font-weight: 800;
        padding: 5px 12px;
        border-radius: 20px;
        letter-spacing: 0.5px;
        border: 1.5px solid #10b981;
        text-transform: uppercase;
    }}
    .status-pill-declined {{
        background: rgba(239, 68, 68, 0.2);
        color: #ef4444;
        font-size: 11px;
        font-weight: 800;
        padding: 5px 12px;
        border-radius: 20px;
        letter-spacing: 0.5px;
        border: 1.5px solid #ef4444;
        text-transform: uppercase;
    }}
    
    /* Key Skills Chips */
    .skill-badge-item {{
        display: inline-block;
        font-size: 11px;
        font-weight: 700;
        padding: 4px 10px;
        border-radius: 8px;
        margin-right: 6px;
        margin-bottom: 6px;
    }}
    
    /* High-Impact Saffron/Amber Action Buttons */
    .stButton > button {{
        border-radius: 10px !important;
        font-weight: 800 !important;
        letter-spacing: 0.5px !important;
        text-transform: uppercase !important;
        font-size: 13px !important;
        padding: 10px 20px !important;
        transition: all 0.25s ease !important;
    }}
    .stButton > button[kind="primary"] {{
        background: linear-gradient(135deg, #c2410c 0%, #ea580c 45%, #f59e0b 100%) !important;
        color: #fef3c7 !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
        box-shadow: 0 4px 20px rgba(234, 88, 12, 0.45) !important;
    }}
    .stButton > button[kind="primary"]:hover {{
        background: linear-gradient(135deg, #ea580c 0%, #f97316 45%, #facc15 100%) !important;
        box-shadow: 0 6px 26px rgba(249, 115, 22, 0.6) !important;
        transform: translateY(-2px) !important;
    }}
    
    .stDownloadButton > button {{
        border-radius: 10px !important;
        font-weight: 800 !important;
        background: linear-gradient(135deg, #d97706 0%, #ea580c 100%) !important;
        color: #fef3c7 !important;
        border: 1px solid rgba(255, 255, 255, 0.25) !important;
        box-shadow: 0 4px 16px rgba(217, 119, 6, 0.35) !important;
    }}
    .stDownloadButton > button:hover {{
        background: linear-gradient(135deg, #f59e0b 0%, #f97316 100%) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(245, 158, 11, 0.5) !important;
    }}
    
    /* High-Impact Danger Delete Button */
    .delete-btn-box .stButton > button {{
        background: linear-gradient(135deg, #b91c1c 0%, #dc2626 100%) !important;
        color: #fef3c7 !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
        box-shadow: 0 4px 14px rgba(220, 38, 38, 0.4) !important;
        text-transform: uppercase !important;
        font-weight: 800 !important;
        border-radius: 10px !important;
    }}
    .delete-btn-box .stButton > button:hover {{
        background: linear-gradient(135deg, #dc2626 0%, #ef4444 100%) !important;
        box-shadow: 0 6px 18px rgba(239, 68, 68, 0.6) !important;
        transform: translateY(-1px) !important;
    }}
    
    /* Stepper Styling for Candidate */
    .stepper-deck {{
        display: flex;
        justify-content: space-between;
        margin: 20px 0 25px 0;
        padding: 18px 24px;
        border-radius: 14px;
    }}
    .step-unit {{
        display: flex;
        flex-direction: column;
        align-items: center;
        flex: 1;
        position: relative;
    }}
    .step-token {{
        width: 34px;
        height: 34px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 13px;
        font-weight: 800;
        margin-bottom: 8px;
    }}
    .token-complete {{
        background: #10b981;
        color: #fef3c7;
        box-shadow: 0 0 12px rgba(16, 185, 129, 0.6);
    }}
    .token-active {{
        background: #f97316;
        color: #fef3c7;
        box-shadow: 0 0 12px rgba(249, 115, 22, 0.6);
    }}
    .token-dormant {{
        background: rgba(148, 163, 184, 0.25);
        color: #94a3b8;
    }}
    .step-descriptor {{
        font-size: 12px;
        font-weight: 700;
        text-align: center;
    }}

    /* Medium Rectangular Theme Button in Exact Right Corner */
    .theme-toggle-box {{
        display: flex !important;
        justify-content: flex-end !important;
        align-items: center !important;
        width: 100% !important;
        margin-left: auto !important;
    }}
    [data-testid="stHorizontalBlock"]:has(.smarthire-navbar) [data-testid="column"]:last-child,
    [data-testid="stHorizontalBlock"]:has(.smarthire-navbar) [data-testid="stColumn"]:last-child,
    .theme-toggle-box .stButton {{
        width: auto !important;
        margin-left: auto !important;
        display: flex !important;
        justify-content: flex-end !important;
        align-items: center !important;
    }}
    [data-testid="stHorizontalBlock"]:has(.smarthire-navbar) [data-testid="column"]:last-child button,
    [data-testid="stHorizontalBlock"]:has(.smarthire-navbar) [data-testid="stColumn"]:last-child button,
    [data-testid="stHorizontalBlock"]:has(.smarthire-navbar) button,
    [data-testid="stHorizontalBlock"]:has(.smarthire-navbar) [data-testid="baseButton-secondary"],
    [data-testid="stHorizontalBlock"]:has(.smarthire-navbar) [data-testid="stBaseButton-secondary"],
    [data-testid="stHorizontalBlock"]:has(.smarthire-navbar) .stButton > button,
    [data-testid="column"]:has(.theme-toggle-box) button,
    [data-testid="stColumn"]:has(.theme-toggle-box) button,
    [data-testid="column"]:has(#theme-btn-marker) button,
    [data-testid="stColumn"]:has(#theme-btn-marker) button,
    div:has(> .theme-toggle-box) + div button,
    div:has(.theme-toggle-box) + div button,
    div:has(.theme-toggle-box) ~ div button,
    .theme-toggle-box .stButton > button,
    .theme-toggle-box button {{
        width: 140px !important;
        min-width: 140px !important;
        max-width: 140px !important;
        height: 38px !important;
        min-height: 38px !important;
        max-height: 38px !important;
        padding: 0 10px !important;
        font-size: 12.5px !important;
        line-height: 36px !important;
        border-radius: 8px !important;
        font-weight: 800 !important;
        letter-spacing: 0.3px !important;
        text-transform: none !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: clip !important;
        box-sizing: border-box !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        gap: 6px !important;
        cursor: pointer !important;
        transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease !important;
    }}
    [data-testid="stHorizontalBlock"]:has(.smarthire-navbar) button p,
    [data-testid="stHorizontalBlock"]:has(.smarthire-navbar) button span,
    .theme-toggle-box .stButton > button p {{
        font-size: 12.5px !important;
        font-weight: 800 !important;
        margin: 0 !important;
        padding: 0 !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: clip !important;
        line-height: 36px !important;
    }}
    [data-testid="stHorizontalBlock"]:has(.smarthire-navbar) button:hover,
    .theme-toggle-box .stButton > button:hover {{
        transform: translateY(-1px) !important;
    }}
    
    /* High-visibility Sidebar Arrow & Toggle Controls (Both Expanded and Collapsed) */
    [data-testid="stSidebarHeader"] {{
        display: flex !important;
        visibility: visible !important;
        opacity: 1 !important;
        min-height: 42px !important;
        padding-top: 10px !important;
        padding-bottom: 6px !important;
    }}

    [data-testid="stSidebarCollapseButton"],
    [data-testid="stExpandSidebarButton"],
    [data-testid="collapsedControl"] {{
        display: flex !important;
        visibility: visible !important;
        opacity: 1 !important;
        z-index: 999999 !important;
    }}

    /* The Expand Arrow Button (When Sidebar is Collapsed -> Top Left >> Arrow) */
    [data-testid="stExpandSidebarButton"],
    [data-testid="collapsedControl"] {{
        display: inline-flex !important;
        visibility: visible !important;
        opacity: 1 !important;
        background: linear-gradient(135deg, #ea580c 0%, #f97316 100%) !important;
        border: 1.5px solid #fed7aa !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 14px rgba(234, 88, 12, 0.55) !important;
        padding: 4px 6px !important;
        margin: 4px 0 0 6px !important;
        cursor: pointer !important;
        transition: all 0.2s ease-in-out !important;
    }}
    [data-testid="stExpandSidebarButton"]:hover,
    [data-testid="collapsedControl"]:hover {{
        transform: scale(1.1) !important;
        box-shadow: 0 6px 20px rgba(234, 88, 12, 0.8) !important;
        border-color: #ffffff !important;
    }}
    [data-testid="stExpandSidebarButton"] *,
    [data-testid="stExpandSidebarButton"] [data-testid="stIconMaterial"],
    [data-testid="collapsedControl"] *,
    [data-testid="collapsedControl"] svg {{
        color: #ffffff !important;
        fill: #ffffff !important;
        stroke: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        font-size: 22px !important;
        font-weight: 800 !important;
    }}

    /* The Collapse Arrow Button (When Sidebar is Open -> Top Right << Arrow) */
    [data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"],
    [data-testid="stSidebarHeader"] [data-testid="stSidebarCollapseButton"] {{
        display: flex !important;
        visibility: visible !important;
        opacity: 1 !important;
    }}
    [data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] button,
    [data-testid="stSidebarHeader"] [data-testid="stSidebarCollapseButton"] button {{
        display: inline-flex !important;
        visibility: visible !important;
        opacity: 1 !important;
        background: linear-gradient(135deg, #ea580c 0%, #f97316 100%) !important;
        border: 1.5px solid #fed7aa !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 14px rgba(234, 88, 12, 0.55) !important;
        padding: 4px 6px !important;
        cursor: pointer !important;
        transition: all 0.2s ease-in-out !important;
    }}
    [data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] button:hover,
    [data-testid="stSidebarHeader"] [data-testid="stSidebarCollapseButton"] button:hover {{
        transform: scale(1.1) !important;
        box-shadow: 0 6px 20px rgba(234, 88, 12, 0.8) !important;
        border-color: #ffffff !important;
    }}
    [data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] button *,
    [data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] button [data-testid="stIconMaterial"],
    [data-testid="stSidebarHeader"] [data-testid="stSidebarCollapseButton"] button *,
    [data-testid="stSidebarHeader"] [data-testid="stSidebarCollapseButton"] button [data-testid="stIconMaterial"] {{
        color: #ffffff !important;
        fill: #ffffff !important;
        stroke: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        font-size: 22px !important;
        font-weight: 800 !important;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------
# DATABASE INITIALIZATION
# ---------------------------
def get_conn():
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_conn()
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS users (
        username TEXT PRIMARY KEY,
        password BLOB NOT NULL,
        role TEXT NOT NULL,
        email TEXT,
        created_at TEXT
    )
    """)
    c.execute("""
    CREATE TABLE IF NOT EXISTS resumes_blob (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        candidate TEXT,
        filename TEXT,
        file_blob BLOB,
        file_text TEXT,
        uploaded_at TEXT,
        hr_decision TEXT DEFAULT 'Pending',
        ai_decision TEXT DEFAULT 'Pending',
        similarity REAL DEFAULT 0.0,
        probability REAL DEFAULT 0.0,
        reviewed_by TEXT,
        reviewed_at TEXT
    )
    """)
    c.execute("""
    CREATE TABLE IF NOT EXISTS active_job_spec (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT,
        jd_text TEXT,
        updated_at TEXT
    )
    """)
    c.execute("""
    CREATE TABLE IF NOT EXISTS job_vacancies (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        role_title TEXT NOT NULL,
        department TEXT,
        vacancies_count INTEGER DEFAULT 1,
        jd_text TEXT NOT NULL,
        required_skills TEXT,
        min_exp_years TEXT DEFAULT '2-4 Years',
        status TEXT DEFAULT 'Open',
        created_by TEXT,
        created_at TEXT
    )
    """)
    c.execute("""
    CREATE TABLE IF NOT EXISTS screening_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        candidate TEXT,
        filename TEXT,
        similarity REAL,
        probability REAL,
        status TEXT,
        stage TEXT,
        notes TEXT,
        jd_checksum TEXT,
        timestamp TEXT
    )
    """)
    
    # Auto-migration
    c.execute("PRAGMA table_info(screening_results)")
    res_cols = [r[1] for r in c.fetchall()]
    if "stage" not in res_cols:
        c.execute("ALTER TABLE screening_results ADD COLUMN stage TEXT")
    if "notes" not in res_cols:
        c.execute("ALTER TABLE screening_results ADD COLUMN notes TEXT")

    c.execute("PRAGMA table_info(resumes_blob)")
    rb_cols = [r[1] for r in c.fetchall()]
    if "hr_decision" not in rb_cols:
        c.execute("ALTER TABLE resumes_blob ADD COLUMN hr_decision TEXT DEFAULT 'Pending'")
    if "ai_decision" not in rb_cols:
        c.execute("ALTER TABLE resumes_blob ADD COLUMN ai_decision TEXT DEFAULT 'Pending'")
    if "similarity" not in rb_cols:
        c.execute("ALTER TABLE resumes_blob ADD COLUMN similarity REAL DEFAULT 0.0")
    if "probability" not in rb_cols:
        c.execute("ALTER TABLE resumes_blob ADD COLUMN probability REAL DEFAULT 0.0")
    if "reviewed_by" not in rb_cols:
        c.execute("ALTER TABLE resumes_blob ADD COLUMN reviewed_by TEXT")
    if "reviewed_at" not in rb_cols:
        c.execute("ALTER TABLE resumes_blob ADD COLUMN reviewed_at TEXT")

    # Seed Default Vacancies if missing
    c.execute("SELECT COUNT(*) FROM job_vacancies")
    if c.fetchone()[0] == 0:
        seed_vacancies = [
            ("Full-Stack Web Developer", "Core Engineering", 3, TECHNICAL_ROLE_TEMPLATES["Full-Stack Web Developer"]["text"], ", ".join(TECHNICAL_ROLE_TEMPLATES["Full-Stack Web Developer"]["skills"]), "2-4 Years (Mid-Level)"),
            ("Backend Software Engineer", "Platform Engineering", 2, TECHNICAL_ROLE_TEMPLATES["Backend Software Engineer"]["text"], ", ".join(TECHNICAL_ROLE_TEMPLATES["Backend Software Engineer"]["skills"]), "3-5 Years (Mid-Senior)"),
            ("Data Scientist & AI Specialist", "Data & Artificial Intelligence", 2, TECHNICAL_ROLE_TEMPLATES["Data Scientist & AI Specialist"]["text"], ", ".join(TECHNICAL_ROLE_TEMPLATES["Data Scientist & AI Specialist"]["skills"]), "2-5 Years (Mid-Level)"),
            ("DevOps & Cloud Architect", "Cloud Infrastructure", 1, TECHNICAL_ROLE_TEMPLATES["DevOps & Cloud Architect"]["text"], ", ".join(TECHNICAL_ROLE_TEMPLATES["DevOps & Cloud Architect"]["skills"]), "3-6 Years (Senior)"),
            ("Cybersecurity & SecOps Analyst", "Information Security", 2, TECHNICAL_ROLE_TEMPLATES["Cybersecurity & SecOps Analyst"]["text"], ", ".join(TECHNICAL_ROLE_TEMPLATES["Cybersecurity & SecOps Analyst"]["skills"]), "2-5 Years (Mid-Level)")
        ]
        now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        for title, dept, v_cnt, jdt, sk, exp in seed_vacancies:
            c.execute("""
                INSERT INTO job_vacancies (role_title, department, vacancies_count, jd_text, required_skills, min_exp_years, status, created_by, created_at)
                VALUES (?, ?, ?, ?, ?, ?, 'Open', 'HR Lead', ?)
            """, (title, dept, v_cnt, jdt, sk, exp, now_iso))

    # Seed Default Accounts if missing
    c.execute("SELECT COUNT(*) FROM users WHERE role='HR'")
    if c.fetchone()[0] == 0:
        pw_hash = bcrypt.hashpw("Admin@12345".encode("utf-8"), bcrypt.gensalt())
        c.execute(
            "INSERT INTO users (username, password, role, email, created_at) VALUES (?,?,?,?,?)",
            ("admin", pw_hash, "HR", "recruitment@smarthire.internal", datetime.now(timezone.utc).isoformat())
        )

    c.execute("SELECT COUNT(*) FROM users WHERE role='Candidate'")
    if c.fetchone()[0] == 0:
        cand_hash = bcrypt.hashpw("Candidate@12345".encode("utf-8"), bcrypt.gensalt())
        c.execute(
            "INSERT INTO users (username, password, role, email, created_at) VALUES (?,?,?,?,?)",
            ("candidate1", cand_hash, "Candidate", "applicant@smarthire.internal", datetime.now(timezone.utc).isoformat())
        )

    conn.commit()
    conn.close()

init_db()

# ---------------------------
# JOB VACANCY MANAGEMENT HELPERS
# ---------------------------
def get_vacancies(only_open=True):
    conn = get_conn()
    if only_open:
        df = pd.read_sql_query("SELECT * FROM job_vacancies WHERE status='Open' ORDER BY id DESC", conn)
    else:
        df = pd.read_sql_query("SELECT * FROM job_vacancies ORDER BY id DESC", conn)
    conn.close()
    return df

def announce_job_vacancy(role_title, department, vacancies_count, jd_text, required_skills, min_exp_years="2-4 Years", created_by="HR Lead"):
    conn = get_conn()
    c = conn.cursor()
    now_ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    c.execute("""
        INSERT INTO job_vacancies (role_title, department, vacancies_count, jd_text, required_skills, min_exp_years, status, created_by, created_at)
        VALUES (?, ?, ?, ?, ?, ?, 'Open', ?, ?)
    """, (role_title, department, vacancies_count, jd_text, required_skills, min_exp_years, created_by, now_ts))
    vid = c.lastrowid
    conn.commit()
    conn.close()
    save_active_jd_text(role_title, jd_text)
    return vid

def set_vacancy_status(vacancy_id, new_status):
    conn = get_conn()
    c = conn.cursor()
    c.execute("UPDATE job_vacancies SET status=? WHERE id=?", (new_status, vacancy_id))
    conn.commit()
    conn.close()

def delete_vacancy(vacancy_id):
    conn = get_conn()
    c = conn.cursor()
    c.execute("DELETE FROM job_vacancies WHERE id=?", (vacancy_id,))
    conn.commit()
    conn.close()

# ---------------------------
# AUTH FUNCTIONS
# ---------------------------
def hash_pw(password: str) -> bytes:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())

def check_pw(password: str, hashed: bytes) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), hashed)
    except Exception:
        return False

def create_user(username, password, role, email=None):
    conn = get_conn()
    c = conn.cursor()
    try:
        now_ts = datetime.now(timezone.utc).isoformat()
        c.execute(
            "INSERT INTO users(username, password, role, email, created_at) VALUES (?,?,?,?,?)",
            (username.strip(), hash_pw(password), role, email.strip() if email else None, now_ts)
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def get_user(username):
    conn = get_conn()
    c = conn.cursor()
    c.execute("SELECT username, password, role, email FROM users WHERE username=?", (username.strip(),))
    r = c.fetchone()
    conn.close()
    return r

def get_all_users_df():
    conn = get_conn()
    df = pd.read_sql_query("SELECT username, role, email, created_at FROM users ORDER BY created_at DESC", conn)
    conn.close()
    return df

def delete_user(username):
    conn = get_conn()
    c = conn.cursor()
    c.execute("DELETE FROM users WHERE username=?", (username,))
    conn.commit()
    conn.close()

# ---------------------------
# CANDIDATE NAME & LOGIN ID PARSING
# ---------------------------
def extract_candidate_name_from_resume(text: str, filename: str, fallback_cand: str = "") -> str:
    """
    Extracts the actual candidate full name from resume text or filename.
    """
    txt = text or ""
    fn = filename or ""
    fallback = fallback_cand or ""
    
    # 1. Try to extract from text first few lines
    if txt:
        lines = [l.strip() for l in txt.splitlines() if l.strip()]
        for line in lines[:5]:
            clean_line = re.sub(r'[,|—–-].*$', '', line).strip()
            clean_line = re.sub(r'\b(CISSP|OSCP|CSPO|PMP|AWS|GCP|BTech|BE|MS|PhD|MBA|CS|MD|CPA)\b', '', clean_line, flags=re.I).strip()
            clean_line = re.sub(r'[,]+', '', clean_line).strip()
            words = clean_line.split()
            if 1 < len(words) <= 4 and all(re.match(r'^[A-Za-z\.\'\-]+$', w) for w in words):
                lower_phrase = clean_line.lower()
                if not any(bad in lower_phrase for bad in ["job description", "summary", "curriculum", "vitae", "resume", "experience", "education", "skills"]):
                    return " ".join(w.capitalize() if not w.endswith('.') else w for w in words)
    
    # 2. Extract from filename
    if fn:
        base = os.path.splitext(fn)[0]
        cleaned = re.sub(r'^\d+[\s_-]*', '', base)
        cleaned = re.sub(r'^(HighMatch|MediumMatch|LowMatch|Fresher|Resume)[\s_-]+', '', cleaned, flags=re.I)
        cleaned = re.sub(r'^(HighMatch|MediumMatch|LowMatch|Fresher|Resume)[\s_-]+', '', cleaned, flags=re.I)
        parts = cleaned.replace('-', '_').split('_')
        role_words = {'lead', 'senior', 'secops', 'data', 'scientist', 'analyst', 'graphic', 'designer', 
                      'fullstack', 'frontend', 'web', 'dev', 'developer', 'hardware', 'tech', 'technician',
                      'principal', 'technical', 'pm', 'supply', 'chain', 'logistics', 'engineer', 'btech', 
                      'cs', 'screening', 'report', 'match', 'resume', 'my', 'updated', 'saas'}
        name_tokens = []
        for p in parts:
            if not p.strip():
                continue
            if p.lower() in role_words and len(name_tokens) >= 2:
                break
            if re.match(r'^\d+$', p) and len(name_tokens) >= 2:
                break
            name_tokens.append(p)
        if name_tokens:
            cand_name = " ".join(name_tokens)
            cand_name = re.sub(r'\s+(Resume|screening report|BTech CS)$', '', cand_name, flags=re.I).strip()
            if cand_name and cand_name.lower() not in ["my updated resume", "resume", "job description"]:
                return cand_name.title()
                
    # 3. Fallback from candidate field
    if fallback:
        if fallback.lower() not in ["candidate1", "candidate", "user", "admin", "job_description"]:
            clean_fallback = re.sub(r'^\d+[\s_-]*', '', fallback).replace('_', ' ')
            clean_fallback = re.sub(r'^(HighMatch|MediumMatch|LowMatch|Fresher|Resume)[\s_-]+', '', clean_fallback, flags=re.I)
            clean_fallback = re.sub(r'\s+(Resume|screening report|BTech CS)$', '', clean_fallback, flags=re.I).strip()
            if clean_fallback:
                return clean_fallback.title()
        elif fallback == "candidate1":
            return "Samantha Ray"

    return "Candidate Applicant"


def resolve_candidate_and_login_id(raw_cand: str, filename: str, file_text: str = "") -> tuple[str, str]:
    """
    Returns (candidate_name, login_id) ensuring clean candidate name from resume
    and distinct account Login ID.
    """
    cand_name = extract_candidate_name_from_resume(file_text, filename, raw_cand)
    raw = (raw_cand or "").strip()
    known_system_logins = {"candidate1", "candidate3", "cand_test_01", "candidate", "user", "demo", "test", "admin"}
    if raw in known_system_logins:
        login_id = raw
    else:
        slug_raw = re.sub(r'^(dr\.|mr\.|ms\.|mrs\.)\s*', '', cand_name.lower()).strip()
        slug_tokens = [re.sub(r'[^a-z0-9]', '', t) for t in slug_raw.split() if t]
        if len(slug_tokens) >= 2:
            login_id = f"{slug_tokens[0]}.{slug_tokens[-1]}"
        elif slug_tokens:
            login_id = slug_tokens[0]
        else:
            login_id = "candidate1"

    return cand_name, login_id


def extract_candidate_email(text: str, candidate_name: str = "", login_id: str = "", user_email_lookup: dict = None) -> str:
    """
    Extracts candidate email from resume text. If not present in resume,
    falls back to user profile email or clean standard candidate email.
    """
    if text:
        matches = re.findall(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', text)
        for m in matches:
            m_clean = m.strip('.,;()[]<>"\'')
            if m_clean and not m_clean.endswith(('.png', '.jpg', '.pdf')) and '@' in m_clean:
                return m_clean.lower()
    if user_email_lookup and login_id in user_email_lookup:
        u_mail = user_email_lookup[login_id]
        if u_mail and "@" in u_mail and not u_mail.startswith("admin@"):
            return u_mail.lower()
    clean_n = re.sub(r'^(dr\.|mr\.|ms\.|mrs\.)\s*', '', (candidate_name or "").lower()).strip()
    slug_tokens = [re.sub(r'[^a-z0-9]', '', t) for t in clean_n.split() if t]
    if len(slug_tokens) >= 2:
        return f"{slug_tokens[0]}.{slug_tokens[-1]}@candidate-mail.com"
    elif slug_tokens:
        return f"{slug_tokens[0]}@candidate-mail.com"
    clean_l = re.sub(r'[^a-z0-9.]', '', (login_id or "candidate").lower()).strip('.')
    return f"{clean_l or 'applicant'}@candidate-mail.com"


class SmartDataFrame(pd.DataFrame):
    """
    A DataFrame subclass that provides robust, case-insensitive and alias-tolerant
    column lookups (e.g. 'Resume File link' vs 'Resume File Link', 'Role Against Which Resumes Are Matched' vs 'role').
    """
    @property
    def _constructor(self):
        return SmartDataFrame

    def __getitem__(self, key):
        if isinstance(key, str) and key not in self.columns:
            k_clean = re.sub(r'[^a-z0-9]', '', key.lower())
            for col in self.columns:
                c_clean = re.sub(r'[^a-z0-9]', '', str(col).lower())
                if k_clean == c_clean:
                    return super().__getitem__(col)
                if 'role' in k_clean and 'role' in c_clean:
                    return super().__getitem__(col)
                if 'link' in k_clean and 'link' in c_clean:
                    return super().__getitem__(col)
        return super().__getitem__(key)


def resolve_matched_role_for_resume(filename: str = "", candidate: str = "", notes: str = "", conn=None) -> str:
    """
    Resolves the benchmark job role against which the resume was matched.
    Prioritizes explicit screening benchmark notes, then active job specification, session state, or default.
    """
    if notes:
        m = re.search(r'Benchmark:\s*([^|\n\r]+)', str(notes))
        if m and m.group(1).strip():
            return m.group(1).strip()

    close_conn = False
    try:
        if conn is None:
            conn = get_conn()
            close_conn = True
        cur = conn.cursor()
        if filename:
            cur.execute("SELECT notes FROM screening_results WHERE filename=? ORDER BY id DESC LIMIT 5", (filename,))
            for row in cur.fetchall():
                if row[0]:
                    m = re.search(r'Benchmark:\s*([^|\n\r]+)', str(row[0]))
                    if m and m.group(1).strip():
                        if close_conn: conn.close()
                        return m.group(1).strip()
        if candidate:
            cur.execute("SELECT notes FROM screening_results WHERE candidate=? ORDER BY id DESC LIMIT 5", (candidate,))
            for row in cur.fetchall():
                if row[0]:
                    m = re.search(r'Benchmark:\s*([^|\n\r]+)', str(row[0]))
                    if m and m.group(1).strip():
                        if close_conn: conn.close()
                        return m.group(1).strip()

        cur.execute("SELECT title FROM active_job_spec ORDER BY id DESC LIMIT 1")
        ajs = cur.fetchone()
        if ajs and ajs[0]:
            if close_conn: conn.close()
            return ajs[0].strip()
    except Exception:
        pass
    finally:
        if close_conn and conn:
            try:
                conn.close()
            except Exception:
                pass

    try:
        if hasattr(st, "session_state") and "target_role_title" in st.session_state and st.session_state.target_role_title:
            return st.session_state.target_role_title.strip()
    except Exception:
        pass

    return "Full-Stack Web Developer"


# ---------------------------
# TEXT EXTRACTION & STORAGE
# ---------------------------
def extract_text_from_file(uploaded_file) -> str:
    name = uploaded_file.name.lower()
    text = ""
    
    if name.endswith(".pdf"):
        if pdfplumber is not None:
            try:
                uploaded_file.seek(0)
                with pdfplumber.open(uploaded_file) as pdf:
                    for page in pdf.pages:
                        text += (page.extract_text() or "") + "\n"
                if text.strip():
                    uploaded_file.seek(0)
                    return text.strip()
            except Exception:
                pass
        if PdfReader is not None:
            try:
                uploaded_file.seek(0)
                reader = PdfReader(uploaded_file)
                for page in reader.pages:
                    text += (page.extract_text() or "") + "\n"
                if text.strip():
                    uploaded_file.seek(0)
                    return text.strip()
            except Exception:
                pass
        uploaded_file.seek(0)
        return text.strip()

    elif name.endswith(".docx"):
        if docx2txt is not None:
            tmp_path = os.path.join(UPLOAD_DIR, f"tmp_{datetime.now(timezone.utc).timestamp()}.docx")
            try:
                uploaded_file.seek(0)
                with open(tmp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                text = docx2txt.process(tmp_path)
            except Exception:
                text = ""
            finally:
                if os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass
                uploaded_file.seek(0)
            return (text or "").strip()
        uploaded_file.seek(0)
        return ""

    else:
        try:
            uploaded_file.seek(0)
            content = uploaded_file.read()
            uploaded_file.seek(0)
            return content.decode("utf-8", errors="ignore").strip()
        except Exception:
            uploaded_file.seek(0)
            return ""

def get_latest_jd_text():
    if "screening_jd_text" in st.session_state and st.session_state.screening_jd_text.strip():
        return st.session_state.screening_jd_text.strip()
    try:
        conn = get_conn()
        c = conn.cursor()
        c.execute("SELECT jd_text FROM active_job_spec ORDER BY id DESC LIMIT 1")
        row = c.fetchone()
        conn.close()
        if row and row["jd_text"]:
            return row["jd_text"].strip()
    except Exception:
        pass
    return "Software Engineer, Full-Stack Developer, Backend, Frontend, Python, JavaScript, React, SQL, Cloud, Docker, Git, REST APIs, Problem Solving, Agile"

def save_active_jd_text(title, jd_text):
    if not jd_text or not jd_text.strip():
        return
    st.session_state.screening_jd_text = jd_text.strip()
    try:
        conn = get_conn()
        c = conn.cursor()
        now_ts = datetime.now(timezone.utc).isoformat()
        c.execute("INSERT INTO active_job_spec(title, jd_text, updated_at) VALUES (?,?,?)",
                  (title or "Target Role", jd_text.strip(), now_ts))
        conn.commit()
        conn.close()
    except Exception:
        pass

# ---------------------------
# NLP & ML ENGINE
# ---------------------------
def compute_similarity_and_probs(jd_text, resume_texts):
    clean_jd = (jd_text or "").strip()
    clean_resumes = [(r or "").strip() for r in resume_texts]

    if not clean_jd:
        clean_jd = "general qualifications experience skills requirements"

    docs = [clean_jd]
    for r in clean_resumes:
        docs.append(r if len(r) > 2 else "candidate profile experience skills")

    try:
        vectorizer = TfidfVectorizer(
            stop_words="english",
            sublinear_tf=True,
            ngram_range=(1, 2),
            max_features=3000
        )
        tfidf = vectorizer.fit_transform(docs)
        sims = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()
    except Exception:
        sims = np.zeros(len(clean_resumes))

    alpha = 14.0
    center = 0.18
    ml_probs = 1.0 / (1.0 + np.exp(-alpha * (sims - center)))
    final_probs = np.where(sims >= 0.88, 0.99, ml_probs)
    return sims, final_probs

def compute_advanced_auto_threshold(sim_scores):
    """
    Advanced Automated AI Threshold Engine.
    Uses Otsu's variance-maximization clustering over applicant score distributions
    to dynamically determine optimal, strict, and broad qualification cutoffs.
    """
    if len(sim_scores) == 0:
        return 0.15, 0.22, 0.10, "Base Distribution"

    scores = np.sort(np.array(sim_scores))
    n = len(scores)

    if n == 1:
        optimal = float(scores[0] * 0.85)
        strict = float(scores[0])
        broad = float(scores[0] * 0.50)
        return round(optimal, 3), round(strict, 3), round(broad, 3), "Single Applicant Calibrated"

    best_thresh = float(np.mean(scores))
    max_var = -1.0
    total_var = float(np.var(scores))

    if total_var > 1e-5:
        for i in range(1, n):
            w0 = i / n
            w1 = (n - i) / n
            m0 = float(np.mean(scores[:i]))
            m1 = float(np.mean(scores[i:]))
            var_between = w0 * w1 * ((m0 - m1) ** 2)
            if var_between > max_var:
                max_var = var_between
                best_thresh = (scores[i - 1] + scores[i]) / 2.0

    optimal_cutoff = round(float(np.clip(best_thresh, min(0.06, float(scores.max()) * 0.75), 0.28)), 3)
    strict_cutoff = round(float(np.clip(np.percentile(scores, 75) if n >= 4 else (optimal_cutoff * 1.15), optimal_cutoff, 0.32)), 3)
    std_val = float(np.std(scores))
    broad_cutoff = round(float(np.clip(np.mean(scores) - 0.5 * std_val, min(0.04, float(scores.min())), optimal_cutoff)), 3)
    rationale = "Otsu Variance Clustering" if max_var > 0 else "Cohort Statistical Mean"
    return optimal_cutoff, strict_cutoff, broad_cutoff, rationale

def evaluate_fresher_candidate(cand_text, cand_skills, jd_skills, raw_similarity, raw_prob):
    """
    Automated Early-Career & Fresher Assessment Engine:
    Detects early-career signals (recent degree, internships, academic projects, lack of long senior titles)
    and evaluates foundational skill match density to ensure talented freshers are fairly qualified.
    """
    t = (cand_text or "").lower()

    fresher_cues = [
        "fresher", "entry level", "entry-level", "intern", "internship", "trainee",
        "b.tech", "btech", "b.e.", "bca", "mca", "b.sc", "b.s.", "bachelor",
        "academic project", "capstone", "coursework", "cgpa", "recent graduate", "class of 202"
    ]
    senior_cues = ["senior", "lead", "principal", "architect", "staff engineer", "head of", "director", "manager"]

    fresher_hits = sum(1 for cue in fresher_cues if cue in t)
    senior_hits = sum(1 for cue in senior_cues if cue in t)

    is_fresher = (fresher_hits >= 2 and senior_hits <= 1) or ("fresher" in t) or ("intern" in t and senior_hits == 0)

    jd_skills_lower = [s.lower() for s in jd_skills] if jd_skills else []
    matched_skills = [s for s in cand_skills if s.lower() in jd_skills_lower]
    skill_ratio = len(matched_skills) / max(1, len(jd_skills)) if jd_skills else 0.0

    if is_fresher:
        if len(matched_skills) >= 2 or skill_ratio >= 0.20 or raw_similarity >= 0.09:
            fresher_boosted_prob = max(raw_prob, min(0.85, 0.55 + 0.10 * len(matched_skills)))
            return {
                "is_fresher": True,
                "fresher_status": "Fresher Qualified",
                "adjusted_prob": round(float(fresher_boosted_prob), 4),
                "badge_label": "FRESHER QUALIFIED",
                "matched_count": len(matched_skills),
                "verdict": f"Early Career • {len(matched_skills)} Core Competencies Matched • High Trainability"
            }
        else:
            return {
                "is_fresher": True,
                "fresher_status": "Fresher (Needs Training)",
                "adjusted_prob": raw_prob,
                "badge_label": "FRESHER (NEEDS TRAINING)",
                "matched_count": len(matched_skills),
                "verdict": "Early Career • Foundational Skill Gaps"
            }
    else:
        return {
            "is_fresher": False,
            "fresher_status": "Experienced",
            "adjusted_prob": raw_prob,
            "badge_label": "EXPERIENCED",
            "matched_count": len(matched_skills),
            "verdict": "Industry Professional"
        }

def generate_interview_questions(candidate_name, skills, missing_skills, is_fresher):
    """
    Automated AI Technical Interview Cheat Sheet Generator (Phase 2):
    Creates 3 targeted, role-specific technical interview questions based on the candidate's
    verified competencies, project history, and identified gap areas.
    """
    top_skill = skills[0] if skills else "Software Engineering"
    second_skill = skills[1] if len(skills) > 1 else "System Design"
    missing = missing_skills[0] if missing_skills else "Cloud Architecture"

    if is_fresher:
        q1 = f"Can you walk us through the technical architecture of your project using {top_skill}, and how you handled data persistence and state?"
        q2 = f"How did you implement {second_skill} in your practical labs or capstone, and what was the most challenging technical bug you solved?"
        q3 = f"This target role utilizes {missing}. What is your current understanding of {missing}, and how would you approach ramping up during your first 2 weeks?"
    else:
        q1 = f"In your past production systems utilizing {top_skill}, what strategies did you employ to optimize throughput, latency, and system resilience?"
        q2 = f"Describe an architectural trade-off you evaluated between {top_skill} and {second_skill} when architecting critical modules."
        q3 = f"Our engineering environment leverages {missing}. How would you bridge your background in {top_skill} to guide best practices in {missing}?"

    return [q1, q2, q3]

def detect_is_fresher(cand_text):
    t = (cand_text or "").lower()
    fresher_cues = [
        "fresher", "entry level", "entry-level", "intern", "internship", "trainee",
        "b.tech", "btech", "b.e.", "bca", "mca", "b.sc", "b.s.", "bachelor",
        "academic project", "capstone", "coursework", "cgpa", "recent graduate", "class of 202"
    ]
    senior_cues = ["senior", "lead", "principal", "architect", "staff engineer", "head of", "director", "manager"]
    fresher_hits = sum(1 for cue in fresher_cues if cue in t)
    senior_hits = sum(1 for cue in senior_cues if cue in t)
    return (fresher_hits >= 2 and senior_hits <= 1) or ("fresher" in t) or ("intern" in t and senior_hits == 0)

def save_resume_blob(candidate, filename, buffer_bytes, text_content=""):
    conn = get_conn()
    c = conn.cursor()
    c.execute("SELECT id, hr_decision FROM resumes_blob WHERE candidate=? AND filename=?", (candidate, filename))
    existing = c.fetchone()
    now_ts = datetime.now(timezone.utc).isoformat()
    
    # Run immediate automated AI screening against active role benchmark
    jd_bench = get_latest_jd_text()
    sim_val = 0.0
    prob_val = 0.0
    ai_status = "Under Evaluation"
    try:
        s_arr, p_arr = compute_similarity_and_probs(jd_bench, [text_content or "candidate profile"])
        sim_val = round(float(s_arr[0]), 4)
        prob_val = round(float(p_arr[0]), 4)
        ai_status = "Selected" if prob_val >= 0.40 else "Rejected"
    except Exception:
        sim_val = 0.50
        prob_val = 0.50
        ai_status = "Selected"

    existing_hr_dec = existing["hr_decision"] if existing and existing["hr_decision"] else "Pending"

    if existing:
        c.execute("""
            UPDATE resumes_blob 
            SET file_blob=?, file_text=?, uploaded_at=?, similarity=?, probability=?, ai_decision=?, hr_decision='Pending'
            WHERE id=?
        """, (buffer_bytes, text_content, now_ts, sim_val, prob_val, ai_status, existing["id"]))
    else:
        c.execute("""
            INSERT INTO resumes_blob(candidate, filename, file_blob, file_text, uploaded_at, hr_decision, ai_decision, similarity, probability)
            VALUES (?,?,?,?,?,?,?,?,?)
        """, (candidate, filename, buffer_bytes, text_content, now_ts, "Pending", ai_status, sim_val, prob_val))
        
    if buffer_bytes and filename:
        try:
            dest_static = os.path.join(STATIC_RESUMES_DIR, filename)
            with open(dest_static, "wb") as f_st:
                f_st.write(buffer_bytes)
        except Exception:
            pass
        
    # Also sync into screening_results so candidate dashboard and pipeline metrics stay updated
    c.execute("SELECT id FROM screening_results WHERE candidate=? AND filename=? ORDER BY id DESC LIMIT 1", (candidate, filename))
    sr_existing = c.fetchone()
    status_for_sr = "Selected" if existing_hr_dec == "Accepted" else ("Rejected" if existing_hr_dec == "Rejected" else "Under Review")
    stage_for_sr = f"HR {existing_hr_dec}" if existing_hr_dec != "Pending" else "Inbound Review"
    
    if sr_existing:
        c.execute("""
            UPDATE screening_results
            SET similarity=?, probability=?, status=?, stage=?, notes=?, timestamp=?
            WHERE id=?
        """, (sim_val, prob_val, status_for_sr, stage_for_sr, f"AI Decision: {ai_status}", now_ts, sr_existing["id"]))
    else:
        c.execute("""
            INSERT INTO screening_results(candidate, filename, similarity, probability, status, stage, notes, jd_checksum, timestamp)
            VALUES (?,?,?,?,?,?,?,?,?)
        """, (candidate, filename, sim_val, prob_val, status_for_sr, stage_for_sr, f"AI Decision: {ai_status}", "v2.4", now_ts))

    conn.commit()
    conn.close()

def set_hr_application_decision(cand_id, candidate_name, filename, decision, reviewer_name):
    conn = get_conn()
    c = conn.cursor()
    cand_id = int(cand_id)
    now_ts = datetime.now(timezone.utc).isoformat()
    # Update resumes_blob
    c.execute("""
        UPDATE resumes_blob
        SET hr_decision=?, reviewed_by=?, reviewed_at=?
        WHERE id=?
    """, (decision, reviewer_name, now_ts, cand_id))
    
    status_mapped = "Selected" if decision == "Accepted" else ("Rejected" if decision == "Rejected" else "Under Review")
    stage_mapped = f"HR {decision}" if decision != "Pending" else "Inbound Review"
    
    c.execute("SELECT id, notes FROM screening_results WHERE candidate=? AND filename=? ORDER BY id DESC LIMIT 1", (candidate_name, filename))
    existing_sr = c.fetchone()
    if existing_sr:
        prev_note = existing_sr["notes"] or ""
        bench_match = re.search(r'Benchmark:\s*([^|\n\r]+)', str(prev_note))
        bench_suffix = f" | Benchmark: {bench_match.group(1).strip()}" if bench_match else ""
        c.execute("""
            UPDATE screening_results
            SET status=?, notes=?, stage=?, timestamp=?
            WHERE id=?
        """, (status_mapped, f"Recruiter Decision: {decision} (by {reviewer_name}){bench_suffix}", stage_mapped, now_ts, existing_sr["id"]))
    else:
        c.execute("SELECT similarity, probability FROM resumes_blob WHERE id=?", (cand_id,))
        rb_row = c.fetchone()
        sim_val = rb_row["similarity"] if rb_row and rb_row["similarity"] is not None else 0.70
        prob_val = rb_row["probability"] if rb_row and rb_row["probability"] is not None else 0.75
        c.execute("SELECT title FROM active_job_spec ORDER BY id DESC LIMIT 1")
        ajs_spec = c.fetchone()
        spec_title = ajs_spec[0].strip() if ajs_spec and ajs_spec[0] else "Full-Stack Web Developer"
        c.execute("""
            INSERT INTO screening_results(candidate, filename, similarity, probability, status, stage, notes, jd_checksum, timestamp)
            VALUES (?,?,?,?,?,?,?,?,?)
        """, (candidate_name, filename, sim_val, prob_val, status_mapped, stage_mapped, f"Recruiter Decision: {decision} (by {reviewer_name}) | Benchmark: {spec_title}", "v2.4", now_ts))
    conn.commit()
    conn.close()


def benchmark_resumes_against_role(target_jd_text, target_role_title, resume_files=None):
    """
    Benchmark all candidate resumes (both uploaded files and database vault candidates)
    against the target role specification, computing similarity, fresher evaluation,
    and rank ordering.
    """
    clean_jd = (target_jd_text or "").strip()
    if not clean_jd:
        return None

    jd_skills = extract_skills_from_text(clean_jd)
    c_names, c_files, c_texts = [], [], []

    # Uploaded files
    if resume_files:
        for rf in resume_files:
            txt = extract_text_from_file(rf)
            c_name = os.path.splitext(rf.name)[0]
            c_files.append(rf.name)
            c_names.append(c_name)
            c_texts.append(txt)
            rf.seek(0)
            buf = rf.read()
            save_resume_blob(c_name, rf.name, buf, txt)
            try:
                with open(os.path.join(UPLOAD_DIR, f"{datetime.now(timezone.utc).timestamp()}_{rf.name}"), "wb") as out:
                    out.write(buf)
            except Exception:
                pass

    # Database vault resumes
    try:
        conn = get_conn()
        c = conn.cursor()
        c.execute("SELECT candidate, filename, file_text FROM resumes_blob")
        db_records = c.fetchall()
        conn.close()
        for row in db_records:
            if row["filename"] not in c_files:
                c_names.append(row["candidate"])
                c_files.append(row["filename"])
                c_texts.append(row["file_text"] or "")
    except Exception:
        pass

    if not c_names:
        return None

    sims, probs = compute_similarity_and_probs(clean_jd, c_texts)
    cand_skills_list = [extract_skills_from_text(t) for t in c_texts]

    fresher_evals = [
        evaluate_fresher_candidate(c_texts[i], cand_skills_list[i], jd_skills, sims[i], probs[i])
        for i in range(len(c_texts))
    ]

    fresher_flags = [fe["is_fresher"] for fe in fresher_evals]
    fresher_labels = [fe["badge_label"] for fe in fresher_evals]
    adjusted_probs = [fe["adjusted_prob"] for fe in fresher_evals]
    verdicts = [fe["verdict"] for fe in fresher_evals]

    df_res = pd.DataFrame({
        "Candidate": c_names,
        "Filename": c_files,
        "Similarity": np.round(sims, 4),
        "Probability": np.round(adjusted_probs, 4),
        "Skills": cand_skills_list,
        "IsFresher": fresher_flags,
        "FresherBadge": fresher_labels,
        "Verdict": verdicts
    })
    df_res["Status"] = df_res["Probability"].apply(
        lambda p: "Selected" if p >= 0.50 else "Rejected"
    )
    df_res = df_res.sort_values(by=["Probability", "Similarity"], ascending=False).reset_index(drop=True)
    df_res.index = df_res.index + 1
    df_res.index.name = "Rank"

    st.session_state.screening_results_df = df_res
    st.session_state.screening_jd_text = clean_jd
    st.session_state.target_role_title = target_role_title
    st.session_state.screening_pdf_cache = {}
    for idx, name in enumerate(c_names):
        st.session_state.candidate_raw_texts[name] = c_texts[idx]

    # Synchronize results back to screening_results DB so all views stay in sync
    try:
        conn = get_conn()
        c = conn.cursor()
        now_ts = datetime.now(timezone.utc).isoformat()
        for idx, row in df_res.iterrows():
            c.execute("SELECT id FROM screening_results WHERE candidate=? AND filename=? ORDER BY id DESC LIMIT 1", (row["Candidate"], row["Filename"]))
            existing_sr = c.fetchone()
            if existing_sr:
                c.execute("""
                    UPDATE screening_results
                    SET similarity=?, probability=?, status=?, stage=?, notes=?, timestamp=?
                    WHERE id=?
                """, (float(row["Similarity"]), float(row["Probability"]), row["Status"], "Role Matched", f"Benchmark: {target_role_title}", now_ts, existing_sr["id"]))
            else:
                c.execute("""
                    INSERT INTO screening_results(candidate, filename, similarity, probability, status, stage, notes, jd_checksum, timestamp)
                    VALUES (?,?,?,?,?,?,?,?,?)
                """, (row["Candidate"], row["Filename"], float(row["Similarity"]), float(row["Probability"]), row["Status"], "Role Matched", f"Benchmark: {target_role_title}", "v2.4", now_ts))
        conn.commit()
        conn.close()
    except Exception:
        pass

    return df_res

def get_candidate_registered_email(cand_name):
    clean = re.sub(r'^\d+[\s_-]*', '', str(cand_name)).replace('_', ' ').strip()
    try:
        conn = get_conn()
        c = conn.cursor()
        c.execute("SELECT email FROM users WHERE username = ? COLLATE NOCASE", (cand_name,))
        row = c.fetchone()
        if row and row["email"]:
            conn.close()
            return row["email"]
        c.execute("SELECT email FROM users WHERE username = ? COLLATE NOCASE", (clean,))
        row = c.fetchone()
        if row and row["email"]:
            conn.close()
            return row["email"]
        c.execute("SELECT email FROM users WHERE username LIKE ? LIMIT 1", (f"%{clean}%",))
        row = c.fetchone()
        if row and row["email"]:
            conn.close()
            return row["email"]
        conn.close()
    except Exception:
        pass
    clean_addr = re.sub(r'[^a-zA-Z0-9_]', '', clean.lower().replace(' ', '.'))
    return f"{clean_addr or 'candidate'}@applicant.com"

def dispatch_formal_decision_report(cand_name, role_title, decision_status, reviewer_name, matched_skills=None, score_pct=None):
    cand_email = get_candidate_registered_email(cand_name)
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    clean_n = re.sub(r'^\d+[\s_-]*', '', str(cand_name)).replace('_', ' ').strip()
    
    if decision_status == "Accepted":
        status_banner = "APPLICATION ACCEPTED & SHORTLISTED"
        core_message = f"""Dear {clean_n},

Congratulations! We are delighted to formally notify you that following our comprehensive technical screening evaluation, your application for the {role_title} position has been officially ACCEPTED by our recruitment committee.

Your technical competencies and profile match have earned you a placement on our technical interview shortlist. Our talent acquisition coordination team will reach out with technical interview scheduling details shortly."""
    elif decision_status == "Rejected":
        status_banner = "APPLICATION NOT SELECTED"
        core_message = f"""Dear {clean_n},

Thank you for your interest in SmartHire and for submitting your qualifications for the {role_title} position.

Our technical evaluation team has reviewed your background against our immediate project requirements. While we were impressed by your background, we have decided to proceed with candidates whose immediate technical specializations more closely match our current architecture. Please review the tailored learning roadmap in your candidate dashboard to prepare for future openings."""
    else:
        status_banner = "APPLICATION UNDER ACTIVE REVIEW"
        core_message = f"""Dear {clean_n},

Thank you for applying for the {role_title} position at SmartHire.

Your technical application and resume are currently UNDER ACTIVE REVIEW by our recruitment committee ({reviewer_name}). We are conducting competency evaluations across all candidates and will communicate final interview decisions shortly."""

    report_content = f"""================================================================================
OFFICIAL RECRUITER DECISION & TECHNICAL EVALUATION REPORT
SmartHire Platform • Automated Talent Intelligence
================================================================================
Date: {now_utc}
Recipient: {clean_n} <{cand_email}>
Target Position: {role_title}
Reviewing Recruiter: {reviewer_name}
Official Recruiter Verdict: {status_banner}
{f'Evaluated Match Relevance: {score_pct}%' if score_pct is not None else ''}
{f'Verified Technical Competencies: {", ".join(matched_skills[:8])}' if matched_skills else ''}

--------------------------------------------------------------------------------
DECISION STATEMENT:
--------------------------------------------------------------------------------
{core_message}

--------------------------------------------------------------------------------
NEXT STEPS FOR CANDIDATE:
--------------------------------------------------------------------------------
- Log into your SmartHire Candidate Portal (http://localhost:8501) to track updates and skill diagnostics.
- Keep your contact information and availability current for upcoming coordination.

Warm regards,
{reviewer_name}
Talent Acquisition Lead
SmartHire Platform
================================================================================"""

    try:
        conn = get_conn()
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS candidate_notifications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                candidate_name TEXT,
                recipient_email TEXT,
                role_title TEXT,
                decision_status TEXT,
                report_text TEXT,
                sent_by TEXT,
                sent_at TIMESTAMP
            )
        """)
        c.execute("""
            INSERT INTO candidate_notifications 
            (candidate_name, recipient_email, role_title, decision_status, report_text, sent_by, sent_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (clean_n, cand_email, role_title, decision_status or "Under Review", report_content, reviewer_name, now_utc))
        conn.commit()
        conn.close()
    except Exception:
        pass

    return cand_email, report_content

def render_inbound_applications_list(parent_key="main"):
    reviewer_user = st.session_state.username if ("username" in st.session_state and st.session_state.username) else "HR Lead"
    is_dark_mode = st.session_state.get("theme", "dark") == "dark"

    st.markdown("### 📥 **Inbound Candidate Applications**")
    st.caption("Review candidate resumes submitted directly through the applicant portal. Review the AI Decision Marker, inspect detected competencies, and make recruiter hiring decisions (Accept or Reject). Live updating every second.")

    # Filter & Search Controls (outside auto-updating fragment so user input never loses focus)
    f_col1, f_col2 = st.columns([1.5, 2.5])
    with f_col1:
        st.selectbox(
            "Filter by Decision Status:",
            ["All Applications", "Pending Review Only", "Accepted Applications", "Rejected Applications"],
            key=f"{parent_key}_inbound_filter"
        )
    with f_col2:
        st.text_input("Search Inbound Applicants (Name or Filename):", "", key=f"{parent_key}_inbound_search")

    @st.fragment(run_every=1)
    def _inbound_live_feed():
        hr_filter = st.session_state.get(f"{parent_key}_inbound_filter", "All Applications")
        inbound_search = st.session_state.get(f"{parent_key}_inbound_search", "")

        conn = get_conn()
        df_inbound = pd.read_sql_query(
            """SELECT id, candidate, filename, uploaded_at, LENGTH(file_blob) as file_size, 
                      file_blob, file_text, hr_decision, ai_decision, similarity, probability, 
                      reviewed_by, reviewed_at 
               FROM resumes_blob ORDER BY uploaded_at DESC""",
            conn
        )
        conn.close()

        if df_inbound.empty:
            st.info("ℹ️ No candidate applications have been submitted through the portal yet. When applicants submit resumes in their portal, they will appear here with automated AI markers.")
            return

        # Top Metrics Bar for Inbound Submissions
        total_apps = len(df_inbound)
        pending_apps = len(df_inbound[df_inbound["hr_decision"].fillna("Pending") == "Pending"])
        accepted_apps = len(df_inbound[df_inbound["hr_decision"] == "Accepted"])
        rejected_apps = len(df_inbound[df_inbound["hr_decision"] == "Rejected"])

        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("Total Inbound Submissions", total_apps)
        m_col2.metric("Pending HR Review", pending_apps)
        m_col3.metric("Accepted by HR", accepted_apps)
        m_col4.metric("Rejected by HR", rejected_apps)

        filtered_df = df_inbound.copy()
        if hr_filter == "Pending Review Only":
            filtered_df = filtered_df[filtered_df["hr_decision"].fillna("Pending") == "Pending"]
        elif hr_filter == "Accepted Applications":
            filtered_df = filtered_df[filtered_df["hr_decision"] == "Accepted"]
        elif hr_filter == "Rejected Applications":
            filtered_df = filtered_df[filtered_df["hr_decision"] == "Rejected"]

        if inbound_search.strip():
            q = inbound_search.strip().lower()
            filtered_df = filtered_df[
                filtered_df["candidate"].str.lower().str.contains(q, na=False) |
                filtered_df["filename"].str.lower().str.contains(q, na=False)
            ]

        if filtered_df.empty:
            st.info("No applications match the selected filter criteria.")
            return

        # Theme-aware styling variables
        c_bg = "linear-gradient(135deg, rgba(38, 19, 8, 0.85) 0%, rgba(20, 14, 34, 0.90) 100%)" if is_dark_mode else "linear-gradient(145deg, #fffdfa 0%, #fff4e8 100%)"
        c_bd = "1.5px solid rgba(249, 115, 22, 0.35)" if is_dark_mode else "1.5px solid rgba(234, 88, 12, 0.28)"
        c_sh = "0 6px 20px rgba(0,0,0,0.3)" if is_dark_mode else "0 4px 18px rgba(234, 88, 12, 0.08)"
        tx_title = "#fef3c7" if is_dark_mode else "#261308"
        tx_sub = "#fed7aa" if is_dark_mode else "#7c2d12"
        file_pill_bg = "rgba(249, 115, 22, 0.18)" if is_dark_mode else "#ffedd5"
        file_pill_bd = "rgba(249, 115, 22, 0.5)" if is_dark_mode else "#ea580c"
        file_pill_tx = "#ffffff" if is_dark_mode else "#7c2d12"

        for idx, row in filtered_df.iterrows():
            c_id = row["id"]
            c_name = row["candidate"]
            c_file = row["filename"]
            c_txt = row["file_text"] or ""
            c_blob = row["file_blob"]
            c_time = row["uploaded_at"]
            c_hr_dec = row["hr_decision"] if row["hr_decision"] else "Pending"
            c_ai_dec = row["ai_decision"] if row["ai_decision"] else "Selected"
            c_sim = float(row["similarity"]) if row["similarity"] is not None else 0.65
            c_prob = float(row["probability"]) if row["probability"] is not None else 0.70
            c_reviewer = row["reviewed_by"] or "Recruiter"
            c_rev_time = row["reviewed_at"] or ""

            # Detected skills & fresher evaluation
            c_skills = extract_skills_from_text(c_txt)
            is_fresher = detect_is_fresher(c_txt)

            # Real candidate name from resume & distinct Login ID
            cand_real_name, cand_login_id = resolve_candidate_and_login_id(c_name, c_file, c_txt)

            # Avatar
            clean_n = cand_real_name
            name_parts = clean_n.split()
            initials = (name_parts[0][0] + (name_parts[1][0] if len(name_parts) > 1 else (name_parts[0][1:2] if len(name_parts[0]) > 1 else ''))).upper() if name_parts else 'AP'
            av_gradient = "linear-gradient(135deg, #ea580c 0%, #ec4899 50%, #8b5cf6 100%)" if not is_fresher else "linear-gradient(135deg, #0284c7 0%, #38bdf8 50%, #06b6d4 100%)"

            # AI Decision Marker HTML
            if c_ai_dec == "Selected":
                ai_badge_html = f"<span style='background: rgba(16, 185, 129, 0.18); border: 1.5px solid #10b981; color: #10b981; padding: 4px 12px; border-radius: 20px; font-size: 11.5px; font-weight: 800; letter-spacing: 0.5px;'>✨ {get_svg_icon('sparkles', 13, '#10b981')} AI Marker: RECOMMENDED</span>"
            else:
                ai_badge_html = f"<span style='background: rgba(239, 68, 68, 0.18); border: 1.5px solid #ef4444; color: #f87171; padding: 4px 12px; border-radius: 20px; font-size: 11.5px; font-weight: 800; letter-spacing: 0.5px;'>⚠️ {get_svg_icon('alert-triangle', 13, '#f87171')} AI Marker: BELOW ROLE BENCHMARK</span>"

            # HR Decision Marker HTML
            if c_hr_dec == "Accepted":
                hr_badge_html = f"<span style='background: rgba(16, 185, 129, 0.22); border: 1.5px solid #10b981; color: #10b981; padding: 5px 14px; border-radius: 20px; font-size: 12.5px; font-weight: 800;'>✅ {get_svg_icon('check-circle', 13, '#10b981')} HR DECISION: ACCEPTED ({c_reviewer})</span>"
            elif c_hr_dec == "Rejected":
                hr_badge_html = f"<span style='background: rgba(239, 68, 68, 0.22); border: 1.5px solid #ef4444; color: #f87171; padding: 5px 14px; border-radius: 20px; font-size: 12.5px; font-weight: 800;'>❌ {get_svg_icon('x-circle', 13, '#f87171')} HR DECISION: REJECTED ({c_reviewer})</span>"
            else:
                hr_badge_html = f"<span style='background: rgba(245, 158, 11, 0.22); border: 1.5px solid #f59e0b; color: #f59e0b; padding: 5px 14px; border-radius: 20px; font-size: 12.5px; font-weight: 800;'>🎯 {get_svg_icon('target', 13, '#f59e0b')} PENDING RECRUITER DECISION</span>"

            # Skills badges
            chips_html = "".join([f"<span class='skill-badge-item'>{s}</span>" for s in c_skills[:8]]) if c_skills else f"<span style='font-size:12px; color:{tx_sub};'>Standard foundational skills parsed</span>"
            fresher_badge = f"<span style='color: #0284c7; font-weight: 800; font-size: 12px;'>🏆 {get_svg_icon('award', 13, '#0284c7')} Early-Career Potential Boost Active</span>" if is_fresher else f"<span style='color: #10b981; font-weight: 800; font-size: 12px;'>💼 {get_svg_icon('briefcase', 13, '#10b981')} Experienced Professional</span>"

            st.markdown(
                f"""
                <div style="background: {c_bg}; border: {c_bd}; border-radius: 14px; padding: 20px 24px; margin-bottom: 16px; box-shadow: {c_sh};">
                    <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid rgba(234, 88, 12, 0.15); padding-bottom: 14px; flex-wrap: wrap; gap: 12px;">
                        <div style="display: flex; align-items: center; gap: 14px;">
                            <div style="width: 44px; height: 44px; border-radius: 12px; background: {av_gradient}; display: flex; align-items: center; justify-content: center; font-weight: 800; font-size: 16px; color: #fef3c7; flex-shrink: 0;">
                                {initials}
                            </div>
                            <div>
                                <div style="font-size: 18px; font-weight: 800; color: {tx_title}; display: flex; align-items: center; gap: 10px;">
                                    <span>{cand_real_name}</span>
                                    <span style="font-size: 12px; font-weight: 700; background: rgba(234, 88, 12, 0.18); border: 1px solid #ea580c; color: #f97316; padding: 2px 8px; border-radius: 6px;">Login ID: {cand_login_id}</span>
                                </div>
                                <div style="font-size: 12.5px; margin-top: 3px; display: flex; align-items: center; gap: 8px; flex-wrap: wrap;">
                                    <span style="color: #ea580c; font-weight: 800;">📄 {get_svg_icon('file-text', 13, '#ea580c')} Submitted Resume:</span>
                                    <span style="background: {file_pill_bg}; border: 1px solid {file_pill_bd}; color: {file_pill_tx}; padding: 2px 8px; border-radius: 6px; font-weight: 800; font-size: 12px; font-family: monospace;">{c_file}</span>
                                    <span style="color: {tx_sub}; font-size: 11.5px;">• Received: {c_time}</span>
                                </div>
                            </div>
                        </div>
                        <div style="display: flex; align-items: center; gap: 10px; flex-wrap: wrap;">
                            {ai_badge_html}
                            {hr_badge_html}
                        </div>
                    </div>
                    <div style="display: flex; gap: 28px; margin: 16px 0 12px 0; align-items: center; flex-wrap: wrap;">
                        <div>
                            <span style="font-size: 10.5px; text-transform: uppercase; font-weight: 800; color: #64748b; letter-spacing: 0.5px;">AI Role Relevance</span>
                            <div style="font-size: 16px; font-weight: 800; color: {'#10b981' if c_sim>=0.22 else '#ea580c'};">{c_sim*100:.1f}%</div>
                        </div>
                        <div>
                            <span style="font-size: 10.5px; text-transform: uppercase; font-weight: 800; color: #64748b; letter-spacing: 0.5px;">AI Shortlist Confidence</span>
                            <div style="font-size: 16px; font-weight: 800; color: {'#10b981' if c_prob>=0.50 else '#ef4444'};">{c_prob*100:.1f}%</div>
                        </div>
                        <div>
                            <span style="font-size: 10.5px; text-transform: uppercase; font-weight: 800; color: #64748b; letter-spacing: 0.5px;">Talent Profile</span>
                            <div style="margin-top: 2px;">{fresher_badge}</div>
                        </div>
                        <div style="flex: 1; min-width: 240px;">
                            <span style="font-size: 10.5px; text-transform: uppercase; font-weight: 800; color: #64748b; letter-spacing: 0.5px;">Detected Technical Competencies</span>
                            <div style="margin-top: 4px; display: flex; flex-wrap: wrap; gap: 4px;">{chips_html}</div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Interactive HR Decision Action Controls
            act_c1, act_c2, act_c3, act_c4, act_c5 = st.columns([1.3, 1.3, 1.1, 1.3, 1.0])
            with act_c1:
                acc_btn_label = "✅ ACCEPT APPLICATION" if c_hr_dec != "Accepted" else "✅ ACCEPTED (CURRENT)"
                if st.button(acc_btn_label, key=f"{parent_key}_btn_acc_{c_id}", type="primary" if c_hr_dec=="Accepted" else "secondary", use_container_width=True):
                    set_hr_application_decision(c_id, c_name, c_file, "Accepted", reviewer_user)
                    st.toast(f"Application for '{clean_n}' ACCEPTED by HR!")
                    st.rerun()
            with act_c2:
                rej_btn_label = "❌ REJECT APPLICATION" if c_hr_dec != "Rejected" else "❌ REJECTED (CURRENT)"
                if st.button(rej_btn_label, key=f"{parent_key}_btn_rej_{c_id}", use_container_width=True):
                    set_hr_application_decision(c_id, c_name, c_file, "Rejected", reviewer_user)
                    st.toast(f"Application for '{clean_n}' marked as REJECTED by HR.")
                    st.rerun()
            with act_c3:
                if st.button("🔄 Reset Status", key=f"{parent_key}_btn_rst_{c_id}", use_container_width=True):
                    set_hr_application_decision(c_id, c_name, c_file, "Pending", reviewer_user)
                    st.toast(f"Application for '{clean_n}' reset to Pending Review.")
                    st.rerun()
            with act_c4:
                if c_blob:
                    st.download_button(
                        label=f"📥 DOWNLOAD FILE",
                        data=c_blob,
                        file_name=c_file,
                        key=f"{parent_key}_inbound_dl_{c_id}",
                        use_container_width=True
                    )
            with act_c5:
                if st.button("✉️ Mail", key=f"{parent_key}_btn_direct_mail_{c_id}", use_container_width=True, help=f"Send formal recruiter decision report directly to {clean_n}'s registered email"):
                    cand_target_role = st.session_state.get("target_role_title", "Technical Position")
                    sent_to_email, report_body = dispatch_formal_decision_report(
                        cand_name=clean_n,
                        role_title=cand_target_role,
                        decision_status=c_hr_dec,
                        reviewer_name=reviewer_user,
                        matched_skills=c_skills,
                        score_pct=round(c_sim * 100, 1) if c_sim is not None else None
                    )
                    st.toast(f"Formal Decision Report emailed to {sent_to_email}!")
                    st.success(f"Formal Recruiter Decision Report successfully emailed to candidate's registered email ({sent_to_email})!")

            with st.expander(f"📄 Inspect Extracted Resume Text for {clean_n}"):
                st.text(c_txt or "No extracted document text available.")

    _inbound_live_feed()

# ---------------------------
# EXECUTIVE PDF REPORT BUILDER
# ---------------------------
def build_pdf_report(candidate, filename, similarity, probability, jd_excerpt=None, status=None, matched_skills=None):
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    w, h = A4

    def sanitize(text):
        if not text:
            return ""
        return str(text).encode("latin-1", "replace").decode("latin-1")

    # Header Banner - Deep Warm Saffron/Amber Twilight
    c.setFillColorRGB(0.12, 0.06, 0.02)
    c.rect(0, h - 38*mm, w, 38*mm, fill=1, stroke=0)

    # Logo Mark - Saffron Orange
    c.setFillColorRGB(0.96, 0.45, 0.08)
    c.roundRect(20*mm, h - 26*mm, 12*mm, 12*mm, 2*mm, fill=1, stroke=0)
    c.setFillColorRGB(1, 1, 1)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(22.5*mm, h - 22.5*mm, "SH")

    c.setFont("Helvetica-Bold", 18)
    c.drawString(36*mm, h - 20*mm, "SmartHire")
    c.setFont("Helvetica", 10)
    c.setFillColorRGB(0.98, 0.75, 0.5)
    c.drawString(36*mm, h - 26*mm, "Candidate Benchmark Evaluation Dossier")

    c.drawRightString(w - 20*mm, h - 20*mm, "CONFIDENTIAL ATS DOSSIER")
    c.setFont("Helvetica", 8)
    c.drawRightString(w - 20*mm, h - 25*mm, f"Timestamp: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")

    # Candidate Section
    c.setFillColorRGB(0.15, 0.1, 0.05)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(20*mm, h - 50*mm, "CANDIDATE ASSESSMENT OVERVIEW")
    
    c.setLineWidth(0.6)
    c.setStrokeColorRGB(0.9, 0.7, 0.5)
    c.line(20*mm, h - 53*mm, w - 20*mm, h - 53*mm)

    c.setFont("Helvetica-Bold", 10)
    c.setFillColorRGB(0.3, 0.2, 0.1)
    c.drawString(20*mm, h - 62*mm, "Applicant Name:")
    c.setFont("Helvetica", 10)
    c.setFillColorRGB(0.1, 0.1, 0.1)
    c.drawString(55*mm, h - 62*mm, sanitize(candidate))

    c.setFont("Helvetica-Bold", 10)
    c.setFillColorRGB(0.3, 0.2, 0.1)
    c.drawString(20*mm, h - 70*mm, "Source Document:")
    c.setFont("Helvetica", 10)
    c.setFillColorRGB(0.1, 0.1, 0.1)
    c.drawString(55*mm, h - 70*mm, sanitize(filename))

    # Metrics Card
    c.setFillColorRGB(0.99, 0.96, 0.92)
    c.roundRect(20*mm, h - 110*mm, w - 40*mm, 32*mm, 3*mm, fill=1, stroke=0)

    c.setFillColorRGB(0.92, 0.38, 0.05)
    c.setFont("Helvetica-Bold", 10)
    c.drawString(25*mm, h - 86*mm, "CORE RELEVANCE & FIT METRICS")

    c.setFillColorRGB(0.3, 0.2, 0.1)
    c.setFont("Helvetica", 9)
    c.drawString(25*mm, h - 96*mm, "Role Semantic Match:")
    c.setFont("Helvetica-Bold", 12)
    c.setFillColorRGB(0.1, 0.1, 0.1)
    c.drawString(25*mm, h - 104*mm, f"{similarity * 100:.1f}%")

    c.setFont("Helvetica", 9)
    c.setFillColorRGB(0.3, 0.2, 0.1)
    c.drawString(85*mm, h - 96*mm, "Shortlist Probability:")
    c.setFont("Helvetica-Bold", 12)
    c.setFillColorRGB(0.1, 0.1, 0.1)
    c.drawString(85*mm, h - 104*mm, f"{probability * 100:.1f}%")

    c.setFont("Helvetica", 9)
    c.setFillColorRGB(0.3, 0.2, 0.1)
    c.drawString(140*mm, h - 96*mm, "Recruiter Decision:")

    if status in ["Selected", "Shortlisted"]:
        c.setFillColorRGB(0.06, 0.65, 0.35)
    elif status == "Rejected":
        c.setFillColorRGB(0.85, 0.2, 0.2)
    else:
        c.setFillColorRGB(0.9, 0.55, 0.1)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(140*mm, h - 104*mm, sanitize(status or "Under Review"))

    # Skills Section
    c.setFillColorRGB(0.15, 0.1, 0.05)
    c.setFont("Helvetica-Bold", 11)
    c.drawString(20*mm, h - 124*mm, "DETECTED TECHNICAL SKILLS")
    c.line(20*mm, h - 127*mm, w - 20*mm, h - 127*mm)

    c.setFont("Helvetica", 9)
    c.setFillColorRGB(0.3, 0.2, 0.1)
    if matched_skills:
        skill_str = ", ".join(matched_skills[:16])
        c.drawString(20*mm, h - 135*mm, sanitize(skill_str))
    else:
        c.drawString(20*mm, h - 135*mm, "Standard candidate profile qualifications parsed.")

    # Job Excerpt
    c.setFillColorRGB(0.15, 0.1, 0.05)
    c.setFont("Helvetica-Bold", 11)
    c.drawString(20*mm, h - 150*mm, "TARGET ROLE BENCHMARK EXCERPT")
    c.line(20*mm, h - 153*mm, w - 20*mm, h - 153*mm)

    c.setFont("Helvetica", 8.5)
    c.setFillColorRGB(0.35, 0.25, 0.15)
    txt = c.beginText(20*mm, h - 162*mm)
    txt.setLeading(13)
    if jd_excerpt:
        for line in sanitize(jd_excerpt).splitlines()[:14]:
            txt.textLine(line[:96])
    else:
        txt.textLine("N/A")
    c.drawText(txt)

    # Footer
    c.setLineWidth(0.5)
    c.setStrokeColorRGB(0.9, 0.7, 0.5)
    c.line(20*mm, 18*mm, w - 20*mm, 18*mm)

    c.setFont("Helvetica", 8)
    c.setFillColorRGB(0.5, 0.4, 0.3)
    c.drawString(20*mm, 13*mm, "SmartHire Platform - Automated Talent Intelligence")
    c.drawRightString(w - 20*mm, 13*mm, "Page 1 of 1")

    c.showPage()
    c.save()
    buf.seek(0)
    return buf.read()

# ---------------------------
# SESSION STATE MANAGEMENT
# ---------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = None
    st.session_state.role = None
    st.session_state.email = None

if "screening_results_df" not in st.session_state:
    st.session_state.screening_results_df = None
if "screening_jd_text" not in st.session_state:
    st.session_state.screening_jd_text = ""
if "screening_pdf_cache" not in st.session_state:
    st.session_state.screening_pdf_cache = {}
if "candidate_raw_texts" not in st.session_state:
    st.session_state.candidate_raw_texts = {}
if "quick_jd_template" not in st.session_state:
    st.session_state.quick_jd_template = ""
if "target_role_title" not in st.session_state:
    st.session_state.target_role_title = "Full-Stack Web Developer"
if "target_dept" not in st.session_state:
    st.session_state.target_dept = "Core Engineering"
if "target_skills" not in st.session_state:
    st.session_state.target_skills = ""
if "target_exp" not in st.session_state:
    st.session_state.target_exp = "2-4 Years (Mid-Level)"
if "target_vacancies" not in st.session_state:
    st.session_state.target_vacancies = 3
if "selected_shortcut_role" not in st.session_state:
    st.session_state.selected_shortcut_role = "Full-Stack Web Developer"

# ---------------------------
# TOP DYNAMIC NAVBAR & THEME TOGGLE (RIGHT CORNER)
# ---------------------------
logged_user_display = st.session_state.username if st.session_state.logged_in else "Guest"
role_display = f"({st.session_state.role})" if st.session_state.logged_in else ""

nav_col1, nav_col2 = st.columns([8.2, 1.8], vertical_alignment="center")

with nav_col1:
    st.markdown(
        f"""
        <div class="smarthire-navbar">
            <div class="brand-group">
                <div class="brand-logo-mark">{get_svg_icon("zap", 18, "#fef3c7" if is_dark else "#fffaf5")}</div>
                <div>
                    <span class="brand-title-text">SmartHire</span>
                    <span class="brand-version-pill">ATS 2.4</span>
                </div>
            </div>
            <div class="status-badge-container">
                <span class="status-pulse-dot"></span>
                <span>Talent Engine Active</span>
                <span style="margin-left: 16px; font-weight: 800;">{logged_user_display} {role_display}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with nav_col2:
    theme_icon_svg = get_svg_icon("moon" if is_dark else "sun", 16, "#fed7aa" if is_dark else "#431407")
    theme_tag = "DARK MODE" if is_dark else "LIGHT MODE"
    st.markdown(
        f"""
        <div style="display: flex; justify-content: flex-end; align-items: center; margin-bottom: 4px;">
            <span style="font-size: 11px; font-weight: 800; color: {'#fed7aa' if is_dark else '#7c2d12'}; display: flex; align-items: center; gap: 4px;">
                {theme_icon_svg} {theme_tag}
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )
    switch_label = "☀️ SWITCH TO LIGHT" if is_dark else "🌙 SWITCH TO DARK"
    if st.button(switch_label, key="theme_toggle_btn", help="Switch between Dark and Light mode", use_container_width=True):
        st.session_state.theme = "light" if is_dark else "dark"
        st.rerun()

st.markdown("<div style='margin-bottom: 22px;'></div>", unsafe_allow_html=True)

# Real-time 1-second live data auto-update heartbeat across the website
@st.fragment(run_every=1)
def _website_live_heartbeat():
    pass

_website_live_heartbeat()

# ---------------------------
# SIDEBAR AUTHENTICATION
# ---------------------------
with st.sidebar:
    sb_head_color = "#fef3c7" if is_dark else "#431407"
    sb_sub_color = "#fed7aa" if is_dark else "#7c2d12"
    st.markdown(f"<h3 style='margin:0 0 12px 0; font-weight:800; font-size:18px; color:{sb_head_color};'>Workspace Access</h3>", unsafe_allow_html=True)
    if not st.session_state.logged_in:
        auth_tab1, auth_tab2 = st.tabs(["Sign In", "Register Account"])

        with auth_tab1:
            st.markdown(f"<div style='margin-bottom:12px; font-weight:700; font-size:13px; color:{sb_sub_color};'>Authenticate with your credentials</div>", unsafe_allow_html=True)
            lu = st.text_input("Username", key="login_username")
            lp = st.text_input("Password", type="password", key="login_password")
            
            if st.button("SIGN IN TO WORKSPACE", type="primary", use_container_width=True):
                user = get_user(lu)
                if user and check_pw(lp, user["password"]):
                    st.session_state.logged_in = True
                    st.session_state.username = user["username"]
                    st.session_state.role = user["role"]
                    st.session_state.email = user["email"]
                    st.rerun()
                else:
                    st.error("Invalid credentials provided.")

        with auth_tab2:
            st.markdown(f"<div style='margin-bottom:12px; font-weight:700; font-size:13px; color:{sb_sub_color};'>New User Provisioning</div>", unsafe_allow_html=True)
            su = st.text_input("Choose Username", key="signup_username")
            sp = st.text_input("Choose Password", type="password", key="signup_password")
            se = st.text_input("Email Address", key="signup_email")
            srole = st.selectbox("Assign Role", ["Candidate", "HR"], key="signup_role")
            st.markdown(f"<div style='margin-top:2px; margin-bottom:8px; font-size:12px; font-weight:800; color:{sb_head_color};'>Selected Role: <span style='color:#f97316;'>{srole}</span></div>", unsafe_allow_html=True)
            
            hr_sec_code = ""
            if srole == "HR":
                hr_sec_code = st.text_input("Company Security Code", type="password", key="signup_hr_sec_code", autocomplete="new-password")
            
            st.caption("Requires: 8+ characters, uppercase, lowercase, digit, symbol.")

            if st.button("REGISTER ACCOUNT", use_container_width=True):
                if not su or not sp:
                    st.warning("Please provide username and password.")
                elif srole == "HR" and hr_sec_code.strip().upper() != "HR-SECURE-2026":
                    st.error("Invalid Company Security Code. HR accounts require authorization.")
                elif get_user(su):
                    st.error("Username is already taken.")
                elif not is_strong_password(sp):
                    st.error("Password complexity requirements not satisfied.")
                else:
                    ok = create_user(su, sp, srole, se or None)
                    if ok:
                        st.success("Account provisioned successfully! You may now sign in.")
                    else:
                        st.error("Error creating user account.")

    else:
        # Dynamic Card for Workspace Access Login Details
        curr_user = st.session_state.get("username", "User")
        curr_role = st.session_state.get("role", "Candidate")
        curr_email = st.session_state.get("email") or "Registered Account"
        is_hr_role = (curr_role == "HR")
        role_accent = "#ea580c" if is_hr_role else "#10b981"
        role_badge_label = "HR Administrator" if is_hr_role else "Applicant / Candidate"
        initials = (curr_user[:2] if len(curr_user) >= 2 else curr_user).upper()
        
        card_bg = "linear-gradient(135deg, rgba(38, 18, 7, 0.95) 0%, rgba(22, 10, 4, 0.98) 100%)" if is_dark else "linear-gradient(135deg, #fff7ed 0%, #ffedd5 100%)"
        card_border = "1.5px solid rgba(249, 115, 22, 0.45)" if is_dark else "1.5px solid rgba(234, 88, 12, 0.35)"
        text_primary = "#fef3c7" if is_dark else "#261308"
        text_muted = "#fed7aa" if is_dark else "#7c2d12"
        badge_bg = "rgba(234, 88, 12, 0.22)" if is_hr_role else "rgba(16, 185, 129, 0.22)"
        badge_border = "#ea580c" if is_hr_role else "#10b981"
        badge_color = "#f97316" if is_hr_role else "#10b981"
        inner_box_bg = "rgba(0, 0, 0, 0.22)" if is_dark else "rgba(255, 255, 255, 0.65)"
        inner_box_bd = "1px solid rgba(249, 115, 22, 0.25)" if is_dark else "1px solid rgba(234, 88, 12, 0.20)"

        st.markdown(
            f"""
            <div style="background: {card_bg}; border: {card_border}; border-radius: 14px; padding: 14px 14px 12px 14px; margin-bottom: 12px; box-shadow: 0 4px 16px rgba(0,0,0,0.18);">
                <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 10px;">
                    <div style="display: flex; align-items: center; gap: 10px;">
                        <div style="width: 36px; height: 36px; border-radius: 50%; background: linear-gradient(135deg, {role_accent} 0%, #f59e0b 100%); display: flex; align-items: center; justify-content: center; font-weight: 800; font-size: 13px; color: #fff; box-shadow: 0 2px 8px rgba(0,0,0,0.25);">
                            {initials}
                        </div>
                        <div>
                            <div style="font-size: 14px; font-weight: 800; color: {text_primary}; line-height: 1.2;">{curr_user}</div>
                            <div style="font-size: 11px; font-weight: 600; color: {text_muted};">{curr_role} Workspace</div>
                        </div>
                    </div>
                    <span style="display: inline-flex; align-items: center; gap: 5px; font-size: 10.5px; font-weight: 700; color: #10b981;">
                        <span style="width: 7px; height: 7px; border-radius: 50%; background: #10b981; box-shadow: 0 0 6px #10b981;"></span>
                        Active
                    </span>
                </div>
                <div style="background: {inner_box_bg}; border: {inner_box_bd}; border-radius: 10px; padding: 9px 10px; margin-bottom: 9px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px;">
                        <span style="font-size: 11px; font-weight: 700; color: {text_muted};">Permissions</span>
                        <span style="background: {badge_bg}; border: 1px solid {badge_border}; color: {badge_color}; padding: 2px 8px; border-radius: 10px; font-size: 10.5px; font-weight: 800;">
                            {role_badge_label}
                        </span>
                    </div>
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-size: 11px; font-weight: 700; color: {text_muted};">Contact Email</span>
                        <span style="font-size: 11px; font-weight: 700; color: {text_primary}; max-width: 130px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;" title="{curr_email}">
                            {curr_email}
                        </span>
                    </div>
                </div>
                <div style="display: flex; align-items: center; justify-content: space-between; font-size: 10.5px; font-weight: 600; color: {text_muted}; padding-top: 1px;">
                    <span>Session: Authenticated</span>
                    <span style="color: {role_accent}; font-weight: 700;">Active 2026</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        st.markdown("---")
        if st.button("🚪 LOG OUT OF SMARTHIRE", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.username = None
            st.session_state.role = None
            st.session_state.email = None
            st.session_state.screening_results_df = None
            st.rerun()

# ---------------------------
# MAIN WORKSPACE
# ---------------------------
if not st.session_state.logged_in:
    hero_title_color = '#fef3c7' if is_dark else '#261308'
    hero_sub_color = '#fed7aa' if is_dark else '#431407'
    st.markdown(
        f"""
        <div class="glass-panel">
            <span class="skill-badge-item" style="font-size: 11px; margin-bottom: 8px;">ENTERPRISE TALENT ACQUISITION</span>
            <h1 style="font-size: 28px; font-weight: 800; letter-spacing: -0.5px; margin: 6px 0 10px 0; color: {hero_title_color};">
                Automated Candidate Screening & Technical Role Matching
            </h1>
            <p style="font-size: 14.5px; line-height: 1.5; margin-bottom: 16px; color: {hero_sub_color};">
                Instant semantic evaluation of applicant resumes against production technical specifications.
            </p>
            <div style="display: flex; gap: 14px; flex-wrap: wrap;">
                <div style="background: rgba(249, 115, 22, 0.12); border: 1px solid rgba(249, 115, 22, 0.4); padding: 12px 18px; border-radius: 12px; flex: 1; min-width: 250px;">
                    <span style="font-weight: 800; font-size: 13px; color: #ea580c;">RECRUITING MANAGERS</span>
                    <div style="font-size: 12.5px; margin-top: 4px; color: {hero_sub_color};">Role specifications, vacancy broadcast, resume evaluation & candidate management.</div>
                </div>
                <div style="background: rgba(16, 185, 129, 0.12); border: 1px solid rgba(16, 185, 129, 0.4); padding: 12px 18px; border-radius: 12px; flex: 1; min-width: 250px;">
                    <span style="font-weight: 800; font-size: 13px; color: #10b981;">JOB CANDIDATES</span>
                    <div style="font-size: 12.5px; margin-top: 4px; color: {hero_sub_color};">Explore vacancies, test eligibility, get diagnostic guidance & track status.</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

else:
    role = st.session_state.role
    user = st.session_state.username

    # ========================================================
    # HR RECRUITER WORKSPACE
    # ========================================================
    if role == "HR":
        reviewer_user = st.session_state.username if ("username" in st.session_state and st.session_state.username) else "HR Lead"
        hr_email_disp = st.session_state.get("email") or "admin@company.com"
        hr_init = (reviewer_user[:2] if len(reviewer_user) >= 2 else reviewer_user).upper()
        hr_card_bg = "linear-gradient(135deg, rgba(38, 18, 7, 0.90) 0%, rgba(22, 10, 4, 0.95) 100%)" if is_dark else "linear-gradient(135deg, #fffaf5 0%, #ffedd5 100%)"
        hr_card_bd = "1.5px solid rgba(249, 115, 22, 0.40)" if is_dark else "1.5px solid rgba(234, 88, 12, 0.35)"

        st.markdown(
            f"""
            <div style="background: {hr_card_bg}; border: {hr_card_bd}; border-radius: 14px; padding: 12px 18px; margin: 4px 0 14px 0; box-shadow: 0 4px 18px rgba(0,0,0,0.14);">
                <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 10px;">
                    <div style="display: flex; align-items: center; gap: 12px;">
                        <div style="width: 38px; height: 38px; border-radius: 50%; background: linear-gradient(135deg, #ea580c 0%, #f59e0b 100%); display: flex; align-items: center; justify-content: center; font-weight: 800; font-size: 14px; color: #fff; box-shadow: 0 2px 8px rgba(234, 88, 12, 0.35);">
                            {hr_init}
                        </div>
                        <div>
                            <div style="font-size: 15px; font-weight: 800; color: {'#fef3c7' if is_dark else '#261308'}; display: flex; align-items: center; gap: 8px;">
                                {reviewer_user}
                                <span style="background: rgba(234, 88, 12, 0.2); border: 1px solid #ea580c; color: #f97316; padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: 800;">HR Administrator</span>
                            </div>
                            <div style="font-size: 12px; color: {'#fed7aa' if is_dark else '#7c2d12'}; margin-top: 1px;">
                                Contact: <span style="font-weight: 700; color: {'#fef3c7' if is_dark else '#261308'};">{hr_email_disp}</span> • Access: <span style="color: #10b981; font-weight: 700;">Full Recruiter Suite</span>
                            </div>
                        </div>
                    </div>
                    <span style="background: rgba(16, 185, 129, 0.15); border: 1px solid rgba(16, 185, 129, 0.4); color: #10b981; padding: 4px 12px; border-radius: 12px; font-size: 11px; font-weight: 800; display: inline-flex; align-items: center; gap: 6px;">
                        <span style="width: 7px; height: 7px; border-radius: 50%; background: #10b981; box-shadow: 0 0 6px #10b981;"></span>
                        Workspace Active
                    </span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        hr_nav1, hr_nav_vacancies, hr_nav_inbound, hr_nav2, hr_nav3, hr_nav4 = st.tabs([
            "🎯 Screening Studio",
            "📢 Vacancy Announcements",
            "📥 Inbound Applications",
            "📈 Talent Pipeline",
            "📁 Document Vault",
            "🛡️ Access Control"
        ])

        # ----------------------------------------------------
        # TAB 1: ACTIVE SCREENING STUDIO
        # ----------------------------------------------------
        with hr_nav1:
            ann_bg = "linear-gradient(135deg, rgba(38, 19, 8, 0.90) 0%, rgba(24, 12, 5, 0.90) 100%)" if is_dark else "linear-gradient(135deg, #fff7ed 0%, #ffedd5 100%)"
            ann_bd = "1.5px solid rgba(249, 115, 22, 0.45)" if is_dark else "1.5px solid rgba(234, 88, 12, 0.35)"
            reviewer_user = st.session_state.username if ("username" in st.session_state and st.session_state.username) else "HR Lead"
            st.markdown("### 🎯 **Talent Screening Studio**")
            st.caption("Benchmark candidate resumes against technical role requirements in real time.")

            # Quick Role Shortcuts & Instant Match Panel
            st.markdown(
                f"""
                <div style="background: {ann_bg}; border: {ann_bd}; border-radius: 12px; padding: 14px 18px; margin: 10px 0 14px 0;">
                    <div style="display: flex; align-items: center; justify-content: space-between; gap: 12px; flex-wrap: wrap;">
                        <div style="display: flex; align-items: center; gap: 10px;">
                            <span>{get_svg_icon('target', 20, '#ea580c')}</span>
                            <div>
                                <div style="font-size: 15px; font-weight: 800; color: {'#fef3c7' if is_dark else '#261308'};">Select Technical Role & Match Resumes</div>
                                <div style="font-size: 12px; color: {'#fed7aa' if is_dark else '#7c2d12'};">Select a job role shortcut to instantly benchmark applicant resumes against it.</div>
                            </div>
                        </div>
                        <span style="background: rgba(234, 88, 12, 0.2); border: 1px solid #ea580c; color: #ea580c; padding: 4px 12px; border-radius: 12px; font-size: 11px; font-weight: 800;">Role Matching Engine</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Domain Categories
            role_domains = [
                "All Roles",
                "Software Engineering",
                "Data & AI",
                "Infrastructure & Security",
                "Quality & Systems",
                "Mobile & App"
            ]

            if "selected_tech_domain" not in st.session_state or st.session_state.selected_tech_domain not in role_domains:
                st.session_state.selected_tech_domain = "All Roles"
            elif st.session_state.selected_tech_domain == "All Roles (in Tech World)":
                st.session_state.selected_tech_domain = "All Roles"

            current_domain = st.session_state.get("selected_tech_domain", "All Roles")
            if current_domain == "All Roles (in Tech World)":
                current_domain = "All Roles"
                st.session_state.selected_tech_domain = "All Roles"

            # Interactive Dropdowns: Domain Dropdown & Tech World Role Dropdown
            drp_col1, drp_col2 = st.columns([1.1, 2.1])
            with drp_col1:
                dom_sel_idx = role_domains.index(current_domain) if current_domain in role_domains else 0
                chosen_domain_dd = st.selectbox(
                    "Domain Category:",
                    role_domains,
                    index=dom_sel_idx,
                    key="hr_domain_category_selectbox"
                )
                if chosen_domain_dd != current_domain:
                    st.session_state.selected_tech_domain = chosen_domain_dd
                    current_domain = chosen_domain_dd

            # All Roles vs Domain Roles
            if current_domain in ["All Roles", "All Roles (in Tech World)"]:
                domain_roles_list = list(TECHNICAL_ROLE_TEMPLATES.keys())
                role_select_label = "Select Role:"
            else:
                domain_roles_list = [k for k, v in TECHNICAL_ROLE_TEMPLATES.items() if v["category"] == current_domain]
                if not domain_roles_list:
                    domain_roles_list = list(TECHNICAL_ROLE_TEMPLATES.keys())
                role_select_label = f"Select Role - {current_domain}:"

            with drp_col2:
                default_role = st.session_state.get("selected_shortcut_role", domain_roles_list[0])
                r_sel_idx = domain_roles_list.index(default_role) if default_role in domain_roles_list else 0
                chosen_role_title = st.selectbox(
                    role_select_label,
                    domain_roles_list,
                    index=r_sel_idx,
                    key=f"hr_role_selectbox_{current_domain.replace(' ', '_').replace('&', 'and')}"
                )
                st.session_state.selected_shortcut_role = chosen_role_title

            # Selected Role Summary Card
            active_spec = TECHNICAL_ROLE_TEMPLATES[chosen_role_title]
            card_tx_col = "#fef3c7" if is_dark else "#261308"
            role_icon_svg = get_svg_icon(active_spec.get("icon", "code"), 22, "#ea580c")
            skills_badges_html = "".join([f"<span class='skill-badge-item' style='margin-right:6px; margin-bottom:5px; font-size:11px;'>{s}</span>" for s in active_spec["skills"][:10]])
            card_box_bg = "linear-gradient(135deg, rgba(38, 18, 7, 0.92) 0%, rgba(26, 12, 4, 0.95) 100%)" if is_dark else "linear-gradient(135deg, #fffaf5 0%, #ffedd5 100%)"
            card_box_bd = "1.5px solid rgba(251, 146, 60, 0.45)" if is_dark else "1.5px solid rgba(234, 88, 12, 0.40)"

            st.markdown(
                f"""
                <div style="background: {card_box_bg}; border: {card_box_bd}; border-radius: 12px; padding: 14px 18px; margin: 10px 0 14px 0; box-shadow: 0 4px 18px rgba(0,0,0,0.18);">
                    <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 10px;">
                        <div style="display: flex; align-items: center; gap: 12px;">
                            <span>{role_icon_svg}</span>
                            <div>
                                <div style="font-size: 16.5px; font-weight: 800; color: {card_tx_col};">{chosen_role_title}</div>
                                <div style="font-size: 11.5px; color: {'#fed7aa' if is_dark else '#7c2d12'}; margin-top: 2px;">
                                    <b>Domain:</b> {active_spec['category']} &nbsp;•&nbsp; <b>Department:</b> {active_spec['department']} &nbsp;•&nbsp; <b>Experience:</b> {active_spec['exp']} &nbsp;•&nbsp; <b>Openings:</b> {active_spec['default_vacancies']}
                                </div>
                            </div>
                        </div>
                        <span style="background: rgba(234, 88, 12, 0.2); border: 1px solid #ea580c; color: #ea580c; padding: 3px 10px; border-radius: 10px; font-size: 10.5px; font-weight: 800;">Target Role Selected</span>
                    </div>
                    <div style="margin-top: 10px; display: flex; flex-wrap: wrap; gap: 5px;">
                        {skills_badges_html}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # ONLY A SINGLE BUTTON TO CONFIRM SELECTION AT THE END OF JOB ROLE / DOMAINS
            if st.button("🎯 CONFIRM ROLE SELECTION & MATCH RESUMES", type="primary", use_container_width=True, key="btn_confirm_role_selection"):
                st.session_state.quick_jd_template = active_spec["text"]
                st.session_state.target_role_title = chosen_role_title
                st.session_state.target_dept = active_spec["department"]
                st.session_state.target_skills = ", ".join(active_spec["skills"])
                st.session_state.target_exp = active_spec["exp"]
                st.session_state.target_vacancies = active_spec["default_vacancies"]
                st.session_state.selected_shortcut_role = chosen_role_title
                
                # Auto-sync announcement broadcast panel
                st.session_state["hr_inp_ann_title"] = chosen_role_title
                st.session_state["hr_inp_ann_dept"] = active_spec["department"]
                st.session_state["hr_inp_ann_dept_synced_role"] = chosen_role_title
                st.session_state["hr_inp_ann_skills"] = ", ".join(active_spec["skills"])
                st.session_state["hr_inp_ann_skills_synced_role"] = chosen_role_title

                save_active_jd_text(chosen_role_title, active_spec["text"])
                matched_df = benchmark_resumes_against_role(active_spec["text"], chosen_role_title)
                if matched_df is not None:
                    st.toast(f"Confirmed '{chosen_role_title}' and matched {len(matched_df)} candidate resumes!")
                st.rerun()

            # Section 1: Role Specification
            st.markdown("##### 🎯 **1. Target Role & Requirements**")
            jd_box1, jd_box2 = st.columns([1, 1])

            with jd_box1:
                jd_upload = st.file_uploader(
                    "Upload Role Specification (.txt, .pdf, .docx)",
                    type=["txt", "pdf", "docx"],
                    key="sh_jd_file"
                )

            with jd_box2:
                jd_text_manual = st.text_area(
                    "Or paste / customize role requirements directly:",
                    value=st.session_state.get("quick_jd_template", ""),
                    height=130,
                    placeholder="e.g. Senior Backend Engineer with Python, FastAPI, Docker, PostgreSQL, Scikit-Learn..."
                )

            active_jd_text = ""
            if jd_upload is not None:
                active_jd_text = extract_text_from_file(jd_upload)
            elif jd_text_manual.strip():
                active_jd_text = jd_text_manual.strip()

            if active_jd_text:
                st.session_state.screening_jd_text = active_jd_text
                current_role_name = st.session_state.get("target_role_title", "Target Role Specification")
                save_active_jd_text(current_role_name, active_jd_text)

            jd_skills = extract_skills_from_text(active_jd_text)
            if jd_skills:
                st.markdown("**Role Key Competencies Detected:**")
                skills_html = "".join([f"<span class='skill-badge-item'>{s}</span>" for s in jd_skills[:14]])
                st.markdown(f"<div>{skills_html}</div>", unsafe_allow_html=True)

            st.markdown("---")

            # Section 2: Resumes
            st.markdown("##### 📄 **2. Applicant Resumes**")
            resume_files = st.file_uploader(
                "Upload Candidate Resumes (.pdf, .docx, .txt)",
                type=["pdf", "docx", "txt"],
                accept_multiple_files=True,
                key="sh_resume_files"
            )

            include_vault_candidates = st.checkbox(
                "Include applicant resumes already stored in candidate vault",
                value=False
            )

            # Benchmark Action Button
            if st.button("⚡ BENCHMARK CANDIDATES AGAINST ROLE", type="primary", use_container_width=True):
                if not active_jd_text:
                    st.error("Please define the role specification first (upload file or paste text).")
                elif not resume_files and not include_vault_candidates:
                    st.error("Please upload resumes or enable the candidate vault option.")
                else:
                    with st.spinner("Extracting NLP features, matching competencies, and computing shortlist metrics..."):
                        c_names, c_files, c_texts = [], [], []

                        # Uploaded files
                        if resume_files:
                            for rf in resume_files:
                                txt = extract_text_from_file(rf)
                                c_name = os.path.splitext(rf.name)[0]
                                c_files.append(rf.name)
                                c_names.append(c_name)
                                c_texts.append(txt)

                                rf.seek(0)
                                buf = rf.read()
                                save_resume_blob(c_name, rf.name, buf, txt)
                                with open(os.path.join(UPLOAD_DIR, f"{datetime.now(timezone.utc).timestamp()}_{rf.name}"), "wb") as out:
                                    out.write(buf)

                        # Vault files
                        if include_vault_candidates:
                            conn = get_conn()
                            c = conn.cursor()
                            c.execute("SELECT candidate, filename, file_text FROM resumes_blob")
                            db_records = c.fetchall()
                            conn.close()

                            for row in db_records:
                                if row["filename"] not in c_files:
                                    c_names.append(row["candidate"])
                                    c_files.append(row["filename"])
                                    c_texts.append(row["file_text"] or "")

                        sims, probs = compute_similarity_and_probs(active_jd_text, c_texts)
                        cand_skills_list = [extract_skills_from_text(t) for t in c_texts]

                        # Automated Fresher & Early-Career Potential Assessment Engine
                        fresher_evals = [
                            evaluate_fresher_candidate(c_texts[i], cand_skills_list[i], jd_skills, sims[i], probs[i])
                            for i in range(len(c_texts))
                        ]

                        fresher_flags = [fe["is_fresher"] for fe in fresher_evals]
                        fresher_labels = [fe["badge_label"] for fe in fresher_evals]
                        adjusted_probs = [fe["adjusted_prob"] for fe in fresher_evals]
                        verdicts = [fe["verdict"] for fe in fresher_evals]

                        df_res = pd.DataFrame({
                            "Candidate": c_names,
                            "Filename": c_files,
                            "Similarity": np.round(sims, 4),
                            "Probability": np.round(adjusted_probs, 4),
                            "Skills": cand_skills_list,
                            "IsFresher": fresher_flags,
                            "FresherBadge": fresher_labels,
                            "Verdict": verdicts
                        })
                        df_res["Status"] = df_res["Probability"].apply(
                            lambda p: "Selected" if p >= 0.50 else "Rejected"
                        )
                        df_res = df_res.sort_values(by=["Probability", "Similarity"], ascending=False).reset_index(drop=True)
                        df_res.index = df_res.index + 1
                        df_res.index.name = "Rank"

                        st.session_state.screening_results_df = df_res
                        st.session_state.screening_jd_text = active_jd_text
                        st.session_state.screening_pdf_cache = {}
                        for idx, name in enumerate(c_names):
                            st.session_state.candidate_raw_texts[name] = c_texts[idx]

                        st.success(f"Benchmarking completed for {len(df_res)} candidate(s).")

            # DISPLAY SCREENING RESULTS
            if st.session_state.screening_results_df is not None:
                res_df = st.session_state.screening_results_df
                jd_text_curr = st.session_state.screening_jd_text

                st.markdown("---")
                st.markdown("### 📊 **Evaluation Analytics & Benchmarks**")

                # Metrics Overview Cards with Saffron, Emerald, Sky Blue and Amber Accents
                k1, k2, k3, k4 = st.columns(4)
                tot = len(res_df)
                sel_exp = len(res_df[(res_df["Status"] == "Selected") & (~res_df["IsFresher"])])
                sel_fresher = len(res_df[(res_df["Status"] == "Selected") & (res_df["IsFresher"])])
                avg_score = res_df["Similarity"].mean()

                # Dynamic Theme Accents for Dashboard Elements
                c_k1_col = "#f97316" if is_dark else "#ea580c"
                c_k2_col = "#10b981" if is_dark else "#059669"
                c_k3_col = "#38bdf8" if is_dark else "#0284c7"
                c_k4_col = "#facc15" if is_dark else "#d97706"

                k1.markdown(f"<div class='metric-grid-card'><div class='metric-header-title'>Total Evaluated</div><div class='metric-hero-val' style='color:{c_k1_col};'>{tot}</div></div>", unsafe_allow_html=True)
                k2.markdown(f"<div class='metric-grid-card'><div class='metric-header-title'>Experienced Shortlist</div><div class='metric-hero-val' style='color:{c_k2_col};'>{sel_exp}</div></div>", unsafe_allow_html=True)
                k3.markdown(f"<div class='metric-grid-card'><div class='metric-header-title'>Early Career Qualified</div><div class='metric-hero-val' style='color:{c_k3_col};'>{sel_fresher}</div></div>", unsafe_allow_html=True)
                k4.markdown(f"<div class='metric-grid-card'><div class='metric-header-title'>Average Match</div><div class='metric-hero-val' style='color:{c_k4_col};'>{avg_score*100:.1f}%</div></div>", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # Compute advanced dynamic thresholds
                sim_values = res_df["Similarity"].values
                optimal_cutoff, strict_cutoff, broad_cutoff, rationale = compute_advanced_auto_threshold(sim_values)

                # Streamlined Candidate Pool View
                ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1.5, 1.8, 1.2])
                with ctrl_col1:
                    filter_name = st.text_input("Filter Candidate Name:", "")
                with ctrl_col2:
                    view_filter = st.selectbox(
                        "Candidate Pool View:",
                        [
                            "⭐ All Shortlisted Candidates (Recommended & Freshers)",
                            "Freshers Only (Early Career Fast-Track)",
                            "Experienced Talent Only",
                            "Complete Candidate Pool (Unfiltered)"
                        ],
                        index=0,
                        help="AI automatically evaluates and shortlists candidates using dynamic cohort thresholding and early-career potential assessment."
                    )
                with ctrl_col3:
                    view_mode = st.radio("Presentation Mode", ["Dynamic Profile Cards", "Candidate Comparison Matrix", "Tabular Dataset"], horizontal=True)

                display_df = res_df.copy()
                if filter_name.strip():
                    display_df = display_df[display_df["Candidate"].str.contains(filter_name.strip(), case=False, na=False)]

                if "All Shortlisted" in view_filter:
                    display_df = display_df[display_df["Status"] == "Selected"]
                elif "Freshers Only" in view_filter:
                    display_df = display_df[(display_df["IsFresher"]) & (display_df["Status"] == "Selected")]
                elif "Experienced Talent" in view_filter:
                    display_df = display_df[(~display_df["IsFresher"]) & (display_df["Status"] == "Selected")]

                # 100% Automated Threshold Diagnostic Strip
                diag_bg = "rgba(234, 88, 12, 0.08)" if is_dark else "linear-gradient(135deg, #fff7ed 0%, #ffedd5 100%)"
                diag_border = "rgba(234, 88, 12, 0.28)" if is_dark else "rgba(234, 88, 12, 0.35)"
                diag_shadow = "none" if is_dark else "0 2px 10px rgba(234, 88, 12, 0.06)"
                diag_text = "#fef3c7" if is_dark else "#261308"
                diag_subtext = "rgba(255, 255, 255, 0.85)" if is_dark else "#7c2d12"

                st.markdown(
                    f"""
                    <div style="background: {diag_bg}; border: 1.5px solid {diag_border}; border-radius: 10px; padding: 12px 18px; margin: 4px 0 16px 0; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px; box-shadow: {diag_shadow};">
                        <div style="display: flex; align-items: center; gap: 10px; flex-wrap: wrap;">
                            <span style="background: linear-gradient(135deg, #ea580c, #f97316); color: #fef3c7; padding: 4px 10px; border-radius: 6px; font-size: 11px; font-weight: 800; letter-spacing: 0.5px; text-transform: uppercase;">100% Automated ATS Engine</span>
                            <span style="font-size: 13px; font-weight: 700; color: {diag_text};">Dynamic Threshold: <span style="color: #ea580c; font-weight: 800;">{optimal_cutoff*100:.1f}%</span></span>
                            <span style="font-size: 12px; font-weight: 600; color: {diag_subtext};">• Early-Career Potential Boost: <b style="color: {'#38bdf8' if is_dark else '#0284c7'}; font-weight: 800;">Active</b></span>
                        </div>
                        <div style="font-size: 13px; font-weight: 700; color: {diag_text};">
                            Displaying <span style="color: {'#10b981' if is_dark else '#059669'}; font-weight: 800;">{len(display_df)}</span> of {len(res_df)} Candidates
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                if display_df.empty:
                    st.info("ℹ️ No candidates match the current filter. Switch to 'Complete Candidate Pool (Unfiltered)' to view everyone.")

                # VIEW 1: DYNAMIC PROFILE CARDS
                if view_mode == "Dynamic Profile Cards":
                    jd_excerpt = " ".join((jd_text_curr or "").split()[:100])
                    jd_skills_lower = [s.lower() for s in jd_skills] if jd_skills else []

                    # Theme-aware styling variables for candidate cards
                    card_bg = "linear-gradient(135deg, rgba(38, 19, 8, 0.75) 0%, rgba(20, 14, 34, 0.85) 100%)" if is_dark else "linear-gradient(145deg, #fffdfa 0%, #fff4e8 100%)"
                    card_border = "rgba(249, 115, 22, 0.35)" if is_dark else "rgba(234, 88, 12, 0.28)"
                    card_shadow = "0 6px 24px rgba(0,0,0,0.35)" if is_dark else "0 4px 20px rgba(234, 88, 12, 0.08), 0 1px 3px rgba(0,0,0,0.03)"
                    card_divider = "rgba(255, 255, 255, 0.08)" if is_dark else "rgba(234, 88, 12, 0.15)"
                    c_title_col = "#fef3c7" if is_dark else "#261308"
                    c_sub_col = "#fed7aa" if is_dark else "#7c2d12"
                    c_lbl_col = "#cbd5e1" if is_dark else "#64748b"
                    gauge_track_col = "rgba(255, 255, 255, 0.1)" if is_dark else "rgba(234, 88, 12, 0.12)"
                    gauge_text_col = "#fef3c7" if is_dark else "#261308"
                    verdict_box_bg = "rgba(255, 255, 255, 0.04)" if is_dark else "linear-gradient(135deg, rgba(255, 247, 237, 0.95) 0%, rgba(254, 243, 199, 0.75) 100%)"
                    verdict_border = "none" if is_dark else "1px solid rgba(234, 88, 12, 0.3)"
                    verdict_text_col = "#fed7aa" if is_dark else "#1e293b"
                    file_pill_bg = "rgba(249, 115, 22, 0.18)" if is_dark else "#ffedd5"
                    file_pill_bd = "rgba(249, 115, 22, 0.5)" if is_dark else "#ea580c"
                    file_pill_tx = "#ffffff" if is_dark else "#7c2d12"

                    for rank_idx, row in display_df.iterrows():
                        c_name = row["Candidate"]
                        c_file = row["Filename"]
                        c_sim = float(row["Similarity"])
                        c_prob = float(row["Probability"])
                        c_status = row["Status"]
                        c_skills = row["Skills"] if isinstance(row["Skills"], list) else []
                        is_fresher_cand = bool(row.get("IsFresher", False))
                        cand_verdict = row.get("Verdict", "General qualifications parsed")

                        # Phase 1: Modern Avatar Initials with Vibrant Gradient
                        clean_n = re.sub(r'^\d+[\s_-]*', '', c_name).replace('_', ' ')
                        name_parts = clean_n.split()
                        initials = (name_parts[0][0] + (name_parts[1][0] if len(name_parts) > 1 else (name_parts[0][1:2] if len(name_parts[0]) > 1 else ''))).upper() if name_parts else 'CA'
                        
                        avatar_gradient = "linear-gradient(135deg, #ea580c 0%, #ec4899 50%, #8b5cf6 100%)" if not is_fresher_cand else "linear-gradient(135deg, #0284c7 0%, #38bdf8 50%, #06b6d4 100%)"

                        # Status & Tier Badges
                        if is_fresher_cand and c_status == "Selected":
                            status_badge_html = "<span style='background: rgba(56, 189, 248, 0.18); border: 1.5px solid #38bdf8; color: #38bdf8; padding: 5px 12px; border-radius: 20px; font-size: 12px; font-weight: 800; letter-spacing: 0.6px; box-shadow: 0 2px 10px rgba(56, 189, 248, 0.25);'>FRESHER QUALIFIED</span>"
                            type_badge_html = f"<span style='color: {'#38bdf8' if is_dark else '#0284c7'}; font-weight: 800; font-size: 13px;'>{get_svg_icon('award', 14, '#38bdf8')} High-Potential Early Career</span>"
                        elif c_status == "Selected":
                            status_badge_html = "<span style='background: rgba(16, 185, 129, 0.18); border: 1.5px solid #10b981; color: #10b981; padding: 5px 12px; border-radius: 20px; font-size: 12px; font-weight: 800; letter-spacing: 0.6px; box-shadow: 0 2px 10px rgba(16, 185, 129, 0.25);'>RECOMMENDED</span>"
                            type_badge_html = f"<span style='color: {'#10b981' if is_dark else '#059669'}; font-weight: 800; font-size: 13px;'>{get_svg_icon('briefcase', 13, '#10b981')} Experienced Professional</span>"
                        else:
                            status_badge_html = "<span style='background: rgba(239, 68, 68, 0.15); border: 1.5px solid #ef4444; color: #f87171; padding: 5px 12px; border-radius: 20px; font-size: 12px; font-weight: 800; letter-spacing: 0.6px;'>DECLINED</span>"
                            type_badge_html = f"<span style='color: {'#94a3b8' if is_dark else '#64748b'}; font-weight: 700; font-size: 13px;'>Below Role Benchmark</span>"

                        # Phase 1: Skill Badge Tiers (Core Must-Have vs Supplementary)
                        matched_chips = []
                        other_chips = []
                        chip_m_bg = "rgba(234, 88, 12, 0.22)" if is_dark else "rgba(234, 88, 12, 0.10)"
                        chip_m_bd = "#ea580c" if is_dark else "rgba(234, 88, 12, 0.4)"
                        chip_m_tx = "#fed7aa" if is_dark else "#9a3412"

                        chip_o_bg = "rgba(255, 255, 255, 0.06)" if is_dark else "#f1f5f9"
                        chip_o_bd = "rgba(255, 255, 255, 0.14)" if is_dark else "#cbd5e1"
                        chip_o_tx = "#cbd5e1" if is_dark else "#334155"

                        for s in c_skills[:10]:
                            if s.lower() in jd_skills_lower:
                                matched_chips.append(f"<span style='background: {chip_m_bg}; border: 1px solid {chip_m_bd}; color: {chip_m_tx}; padding: 3px 9px; border-radius: 6px; font-size: 11px; font-weight: 800; margin: 2px;'>{get_svg_icon('check-circle', 11, chip_m_tx)} {s}</span>")
                            else:
                                other_chips.append(f"<span style='background: {chip_o_bg}; border: 1px solid {chip_o_bd}; color: {chip_o_tx}; padding: 3px 8px; border-radius: 6px; font-size: 11px; font-weight: 600; margin: 2px;'>{s}</span>")
                        
                        all_chips_html = "".join(matched_chips + other_chips) if (matched_chips or other_chips) else f"<span style='font-size:12px; font-weight:700; color:{c_sub_col}; opacity:0.8;'>Foundational competencies parsed</span>"

                        # Phase 1: Radial Visual SVG Gauges
                        sim_pct = max(0, min(100, float(c_sim * 100)))
                        prob_pct = max(0, min(100, float(c_prob * 100)))
                        sim_offset = 113.1 * (1.0 - (sim_pct / 100.0))
                        prob_offset = 113.1 * (1.0 - (prob_pct / 100.0))
                        sim_color = ("#10b981" if is_dark else "#059669") if sim_pct >= 24 else ("#ea580c" if sim_pct >= 14 else ("#f59e0b" if is_dark else "#d97706"))
                        prob_color = ("#10b981" if is_dark else "#059669") if prob_pct >= 60 else (("#facc15" if is_dark else "#d97706") if prob_pct >= 40 else "#ef4444")

                        rank_medal = f"#{rank_idx} "

                        # Phase 2: Missing Competency Gap Analysis
                        missing_skills = [s for s in jd_skills if s.lower() not in [cs.lower() for cs in c_skills]] if jd_skills else []
                        gap_chip_bg = "rgba(239, 68, 68, 0.12)" if is_dark else "rgba(239, 68, 68, 0.08)"
                        gap_chip_bd = "rgba(239, 68, 68, 0.35)" if is_dark else "rgba(239, 68, 68, 0.3)"
                        gap_chip_tx = "#fca5a5" if is_dark else "#b91c1c"
                        missing_chips_html = "".join([f"<span style='background: {gap_chip_bg}; border: 1px solid {gap_chip_bd}; color: {gap_chip_tx}; padding: 2px 7px; border-radius: 5px; font-size: 11px; font-weight: 700; margin: 2px;'>{get_svg_icon('x-circle', 11, gap_chip_tx)} {ms}</span>" for ms in missing_skills[:4]]) if missing_skills else f"<span style='color: {'#10b981' if is_dark else '#059669'}; font-size: 11px; font-weight: 700;'>{get_svg_icon('check-circle', 11, '#10b981')} Full Key Skill Coverage</span>"

                        # Phase 2: Early-Career Potential Radar & Agility Metrics (for Freshers)
                        fresher_radar_html = ""
                        if is_fresher_cand:
                            overlap_pct = min(100, int(len(matched_chips) / max(1, len(jd_skills)) * 100)) if jd_skills else 65
                            trainability_val = min(98, 70 + len(matched_chips) * 4)
                            fr_bg = "rgba(56, 189, 248, 0.08)" if is_dark else "linear-gradient(135deg, rgba(240, 249, 255, 0.95) 0%, rgba(224, 242, 254, 0.6) 100%)"
                            fr_bd = "rgba(56, 189, 248, 0.25)" if is_dark else "rgba(2, 132, 199, 0.28)"
                            fr_title_col = "#38bdf8" if is_dark else "#0284c7"
                            fr_txt_col = "#e2e8f0" if is_dark else "#0f172a"
                            fr_success_col = "#10b981" if is_dark else "#059669"
                            fresher_radar_html = f"""
                            <div style="background: {fr_bg}; border: 1px solid {fr_bd}; border-radius: 8px; padding: 10px 14px; margin-top: 10px; display: flex; gap: 20px; align-items: center; flex-wrap: wrap;">
                                <div style="font-size: 11px; font-weight: 800; color: {fr_title_col}; text-transform: uppercase;">{get_svg_icon("award", 13, "#38bdf8")} Early-Career Potential Index:</div>
                                <div style="font-size: 12px; color: {fr_txt_col};">Foundational Overlap: <b>{overlap_pct}%</b></div>
                                <div style="font-size: 12px; color: {fr_txt_col};">Practical Evidence: <b>Verified Projects</b></div>
                                <div style="font-size: 12px; color: {fr_txt_col};">Trainability & Agility: <b style="color: {fr_success_col};">{trainability_val}/100</b></div>
                            </div>
                            """

                        st.markdown(
                            f"""
                            <div class="profile-card" style="background: {card_bg}; border: 1px solid {card_border}; border-radius: 14px; padding: 22px 24px; margin-bottom: 14px; backdrop-filter: blur(14px); box-shadow: {card_shadow};">
                                <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid {card_divider}; padding-bottom: 14px; flex-wrap: wrap; gap: 12px;">
                                    <div style="display: flex; align-items: center; gap: 14px;">
                                        <div style="width: 46px; height: 46px; border-radius: 12px; background: {avatar_gradient}; display: flex; align-items: center; justify-content: center; font-weight: 800; font-size: 17px; color: #fef3c7; box-shadow: 0 4px 14px rgba(0,0,0,0.25); flex-shrink: 0;">
                                            {initials}
                                        </div>
                                        <div>
                                            <div style="font-size: 18px; font-weight: 800; color: {c_title_col}; letter-spacing: -0.2px;">{rank_medal}{clean_n}</div>
                                            <div style="font-size: 12.5px; margin-top: 4px; display: flex; align-items: center; gap: 6px;">
                                                <span style="color: #ea580c; font-weight: 800;">{get_svg_icon('file-text', 13, '#ea580c')} Resume File:</span>
                                                <span style="background: {file_pill_bg}; border: 1px solid {file_pill_bd}; color: {file_pill_tx}; padding: 2px 8px; border-radius: 6px; font-weight: 800; font-size: 12px; font-family: monospace;">{c_file}</span>
                                            </div>
                                        </div>
                                    </div>
                                    <div>{status_badge_html}</div>
                                </div>
                                <div style="display: flex; gap: 32px; margin: 16px 0 14px 0; align-items: center; flex-wrap: wrap;">
                                    <div style="display: flex; align-items: center; gap: 10px;">
                                        <div style="position: relative; width: 50px; height: 50px; display: inline-flex; align-items: center; justify-content: center;">
                                            <svg width="50" height="50" viewBox="0 0 44 44">
                                                <circle cx="22" cy="22" r="18" fill="none" stroke="{gauge_track_col}" stroke-width="4"/>
                                                <circle cx="22" cy="22" r="18" fill="none" stroke="{sim_color}" stroke-width="4"
                                                        stroke-dasharray="113.1" stroke-dashoffset="{sim_offset:.1f}"
                                                        stroke-linecap="round" transform="rotate(-90 22 22)"/>
                                            </svg>
                                            <span style="position: absolute; font-size: 11px; font-weight: 800; color: {gauge_text_col};">{sim_pct:.0f}%</span>
                                        </div>
                                        <div>
                                            <span style="font-size: 10px; text-transform: uppercase; font-weight: 800; color: {c_lbl_col}; letter-spacing: 0.5px;">Role Match</span>
                                            <div style="font-size: 14px; font-weight: 800; color: {sim_color};">{sim_pct:.1f}%</div>
                                        </div>
                                    </div>
                                    <div style="display: flex; align-items: center; gap: 10px;">
                                        <div style="position: relative; width: 50px; height: 50px; display: inline-flex; align-items: center; justify-content: center;">
                                            <svg width="50" height="50" viewBox="0 0 44 44">
                                                <circle cx="22" cy="22" r="18" fill="none" stroke="{gauge_track_col}" stroke-width="4"/>
                                                <circle cx="22" cy="22" r="18" fill="none" stroke="{prob_color}" stroke-width="4"
                                                        stroke-dasharray="113.1" stroke-dashoffset="{prob_offset:.1f}"
                                                        stroke-linecap="round" transform="rotate(-90 22 22)"/>
                                            </svg>
                                            <span style="position: absolute; font-size: 11px; font-weight: 800; color: {gauge_text_col};">{prob_pct:.0f}%</span>
                                        </div>
                                        <div>
                                            <span style="font-size: 10px; text-transform: uppercase; font-weight: 800; color: {c_lbl_col}; letter-spacing: 0.5px;">Fit Confidence</span>
                                            <div style="font-size: 14px; font-weight: 800; color: {prob_color};">{prob_pct:.1f}%</div>
                                        </div>
                                    </div>
                                    <div>
                                        <span style="font-size: 10px; text-transform: uppercase; font-weight: 800; color: {c_lbl_col}; letter-spacing: 0.5px;">Talent Tier</span>
                                        <div style="margin-top: 2px;">{type_badge_html}</div>
                                    </div>
                                    <div style="flex: 1; min-width: 240px;">
                                        <span style="font-size: 10px; text-transform: uppercase; font-weight: 800; color: {c_lbl_col}; letter-spacing: 0.5px;">Competencies (Core Matches)</span>
                                        <div style="margin-top: 4px; display: flex; flex-wrap: wrap; gap: 4px;">{all_chips_html}</div>
                                    </div>
                                </div>
                                <div style="display: flex; justify-content: space-between; align-items: center; gap: 14px; flex-wrap: wrap; margin-bottom: 8px;">
                                    <div style="font-size: 11px; color: {c_lbl_col};"><b>Competency Gaps:</b> {missing_chips_html}</div>
                                </div>
                                <div style="background: {verdict_box_bg}; border-left: 4px solid #ea580c; border: {verdict_border}; border-radius: 6px; padding: 10px 14px; font-size: 12.5px; color: {verdict_text_col};">
                                    <b>AI Assessment:</b> {cand_verdict}
                                </div>
                                {fresher_radar_html}
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                        # Phase 2: Automated AI Technical Interview Prep Cheat Sheet (3 Questions)
                        with st.expander(f"AI Technical Interview Questions ({clean_n})", expanded=False):
                            qs = generate_interview_questions(clean_n, c_skills, missing_skills, is_fresher_cand)
                            prep_bg = "rgba(46, 23, 9, 0.75)" if is_dark else "linear-gradient(135deg, #fffaf5 0%, #fff1e0 100%)"
                            prep_border = "rgba(234, 88, 12, 0.35)" if is_dark else "rgba(234, 88, 12, 0.28)"
                            prep_shadow = "none" if is_dark else "0 2px 10px rgba(234, 88, 12, 0.06)"
                            prep_q1_c = "#ea580c" if is_dark else "#c2410c"
                            prep_q2_c = "#facc15" if is_dark else "#b45309"
                            prep_q3_c = "#38bdf8" if is_dark else "#0284c7"
                            prep_tx_c = "#e2e8f0" if is_dark else "#1e293b"
                            st.markdown(
                                f"""
                                <div style="background: {prep_bg}; border: 1.5px solid {prep_border}; border-radius: 8px; padding: 14px 18px; margin: 4px 0 8px 0; box-shadow: {prep_shadow};">
                                    <div style="font-size: 11.5px; font-weight: 800; color: #ea580c; text-transform: uppercase; margin-bottom: 10px; letter-spacing: 0.5px;">Targeted Technical Screening Questions:</div>
                                    <div style="margin-bottom: 10px; font-size: 13px; line-height: 1.5; color: {prep_tx_c};"><span style="color: {prep_q1_c}; font-weight: 800;">Q1 (Architecture & Mastery):</span> {qs[0]}</div>
                                    <div style="margin-bottom: 10px; font-size: 13px; line-height: 1.5; color: {prep_tx_c};"><span style="color: {prep_q2_c}; font-weight: 800;">Q2 (Practical Projects & Debugging):</span> {qs[1]}</div>
                                    <div style="font-size: 13px; line-height: 1.5; color: {prep_tx_c};"><span style="color: {prep_q3_c}; font-weight: 800;">Q3 (Skill Gap & Ramp-Up Agility):</span> {qs[2]}</div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                        # Phase 3: 1-Click Recruiter Triage Action Bar & Direct Dossier Download
                        pdf_cache_key = f"{c_name}_{c_file}"
                        if pdf_cache_key not in st.session_state.screening_pdf_cache:
                            st.session_state.screening_pdf_cache[pdf_cache_key] = build_pdf_report(
                                c_name, c_file, c_sim, c_prob, jd_excerpt, c_status, c_skills
                            )
                        pdf_data = st.session_state.screening_pdf_cache[pdf_cache_key]

                        triage_c1, triage_c2, triage_c3, triage_c4 = st.columns([1.1, 1.1, 1.3, 1.3])
                        with triage_c1:
                            if st.button("✅ Accept Application", key=f"btn_shortlist_{rank_idx}_{c_name}", use_container_width=True):
                                st.toast(f"{clean_n} marked as Accepted!")
                        with triage_c2:
                            if st.button("❌ Reject Application", key=f"btn_star_{rank_idx}_{c_name}", use_container_width=True):
                                st.toast(f"{clean_n} marked as Declined.")
                        with triage_c3:
                            if st.button("✉️ Mail", key=f"btn_screening_mail_{rank_idx}_{c_name}", use_container_width=True, help=f"Send formal recruiter decision report directly to {clean_n}'s registered email"):
                                cand_target_role = st.session_state.get("target_role_title", "Technical Position")
                                sent_to_email, report_body = dispatch_formal_decision_report(
                                    cand_name=clean_n,
                                    role_title=cand_target_role,
                                    decision_status="Accepted" if c_status == "Strong Match" else "Under Review",
                                    reviewer_name=reviewer_user,
                                    matched_skills=c_skills,
                                    score_pct=round(c_sim * 100, 1) if c_sim is not None else None
                                )
                                st.toast(f"Formal Recruiter's Decision Report emailed to {sent_to_email}!")
                                st.success(f"Formal Decision Report successfully emailed to candidate's registered email ({sent_to_email})!")
                        with triage_c4:
                                invite_email_txt = f"""Subject: Interview Invitation: Technical Round with SmartHire Team

Dear {clean_n},

Congratulations! We were impressed by your technical qualifications and profile match for our open position.

Based on our automated competency evaluation, we would like to invite you to a 45-minute technical screening interview to discuss your experience, technical projects, and career aspirations.

Please let us know your availability over the next 3 business days, or select a slot via our recruitment portal.

Best regards,
Talent Acquisition Team
SmartHire ATS Platform"""
                                st.text_area("Ready-to-Send Email Invite:", value=invite_email_txt, height=180, key=f"email_invite_{rank_idx}_{c_name}")
                                st.caption("Copy and send directly to the candidate.")
                        with triage_c4:
                            st.download_button(
                                label="📥 DOSSIER (PDF)",
                                data=pdf_data,
                                file_name=f"{c_name}_SmartHire_dossier.pdf",
                                mime="application/pdf",
                                key=f"dl_dossier_{rank_idx}_{c_name}",
                                use_container_width=True
                            )

                # VIEW 2: CANDIDATE FACE-OFF COMPARE (Phase 3)
                elif view_mode == "Candidate Comparison Matrix":
                    st.markdown("#### ⚖️ **Side-by-Side Candidate Comparison Matrix**")
                    st.caption("Perform an in-depth head-to-head comparison of technical overlap, competency depth, and hiring trade-offs.")

                    cand_names_list = list(res_df["Candidate"].values)
                    if len(cand_names_list) < 2:
                        st.info("At least 2 evaluated candidates are required for side-by-side comparison.")
                    else:
                        fo_col1, fo_col2 = st.columns(2)
                        with fo_col1:
                            cand_a_name = st.selectbox("Select Candidate A (Primary):", cand_names_list, index=0, key="fo_cand_a")
                        with fo_col2:
                            cand_b_name = st.selectbox("Select Candidate B (Challenger):", cand_names_list, index=1 if len(cand_names_list) > 1 else 0, key="fo_cand_b")

                        matches_a = res_df[res_df["Candidate"] == cand_a_name]
                        matches_b = res_df[res_df["Candidate"] == cand_b_name]
                        if matches_a.empty or matches_b.empty:
                            st.warning("Selected candidates are not found in the current evaluation dataset.")
                            st.stop()
                        row_a = matches_a.iloc[0]
                        row_b = matches_b.iloc[0]

                        fo_a_bg = "rgba(234, 88, 12, 0.08)" if is_dark else "linear-gradient(135deg, #fffaf5 0%, #fff7ed 100%)"
                        fo_a_bd = "2px solid #ea580c"
                        fo_a_sh = "0 4px 18px rgba(0,0,0,0.3)" if is_dark else "0 6px 20px rgba(234, 88, 12, 0.1)"
                        fo_a_title = "#fef3c7" if is_dark else "#261308"
                        fo_a_sub = "#fed7aa" if is_dark else "#c2410c"
                        fo_a_lbl = "#cbd5e1" if is_dark else "#64748b"
                        fo_a_body = "#fed7aa" if is_dark else "#334155"

                        fo_b_bg = "rgba(56, 189, 248, 0.08)" if is_dark else "linear-gradient(135deg, #f8fafc 0%, #f0f9ff 100%)"
                        fo_b_bd = "2px solid #38bdf8" if is_dark else "2px solid #0284c7"
                        fo_b_sh = "0 4px 18px rgba(0,0,0,0.3)" if is_dark else "0 6px 20px rgba(2, 132, 199, 0.1)"
                        fo_b_title = "#fef3c7" if is_dark else "#261308"
                        fo_b_sub = "#bae6fd" if is_dark else "#0284c7"
                        fo_b_lbl = "#cbd5e1" if is_dark else "#64748b"
                        fo_b_body = "#bae6fd" if is_dark else "#334155"
                        fo_b_sim_c = "#38bdf8" if is_dark else "#0284c7"
                        fo_conf_c = "#facc15" if is_dark else "#d97706"

                        card_a_col, card_b_col = st.columns(2)
                        with card_a_col:
                            sim_a = float(row_a["Similarity"]) * 100
                            prob_a = float(row_a["Probability"]) * 100
                            skills_a = row_a["Skills"] if isinstance(row_a["Skills"], list) else []
                            is_fresher_a = bool(row_a.get("IsFresher", False))
                            tag_a = "Fresher" if is_fresher_a else "Experienced"

                            st.markdown(f"""
                            <div style="background: {fo_a_bg}; border: {fo_a_bd}; border-radius: 12px; padding: 18px 20px; box-shadow: {fo_a_sh};">
                                <div style="font-size: 18px; font-weight: 800; color: {fo_a_title};">{cand_a_name} <span style="font-size: 12px; color: {fo_a_sub};">({tag_a})</span></div>
                                <div style="display: flex; gap: 20px; margin: 12px 0;">
                                    <div><span style="font-size: 11px; text-transform: uppercase; font-weight: 800; color: {fo_a_lbl};">Role Match</span><div style="font-size: 22px; font-weight: 800; color: #ea580c;">{sim_a:.1f}%</div></div>
                                    <div><span style="font-size: 11px; text-transform: uppercase; font-weight: 800; color: {fo_a_lbl};">Fit Confidence</span><div style="font-size: 22px; font-weight: 800; color: {fo_conf_c};">{prob_a:.1f}%</div></div>
                                    <div><span style="font-size: 11px; text-transform: uppercase; font-weight: 800; color: {fo_a_lbl};">Status</span><div style="font-size: 16px; font-weight: 800; color: {'#10b981' if row_a['Status']=='Selected' else '#ef4444'};">{row_a['Status']}</div></div>
                                </div>
                                <div style="font-size: 12px; color: {fo_a_body}; margin-top: 6px;"><b>Key Competencies ({len(skills_a)}):</b> {', '.join(skills_a[:8]) if skills_a else 'None parsed'}</div>
                                <div style="font-size: 12px; opacity: 0.95; color: {fo_a_body}; margin-top: 6px;"><b>Assessment:</b> {row_a.get('Verdict', 'N/A')}</div>
                            </div>
                            """, unsafe_allow_html=True)

                        with card_b_col:
                            sim_b = float(row_b["Similarity"]) * 100
                            prob_b = float(row_b["Probability"]) * 100
                            skills_b = row_b["Skills"] if isinstance(row_b["Skills"], list) else []
                            is_fresher_b = bool(row_b.get("IsFresher", False))
                            tag_b = "Fresher" if is_fresher_b else "Experienced"

                            st.markdown(f"""
                            <div style="background: {fo_b_bg}; border: {fo_b_bd}; border-radius: 12px; padding: 18px 20px; box-shadow: {fo_b_sh};">
                                <div style="font-size: 18px; font-weight: 800; color: {fo_b_title};">{cand_b_name} <span style="font-size: 12px; color: {fo_b_sub};">({tag_b})</span></div>
                                <div style="display: flex; gap: 20px; margin: 12px 0;">
                                    <div><span style="font-size: 11px; text-transform: uppercase; font-weight: 800; color: {fo_b_lbl};">Role Match</span><div style="font-size: 22px; font-weight: 800; color: {fo_b_sim_c};">{sim_b:.1f}%</div></div>
                                    <div><span style="font-size: 11px; text-transform: uppercase; font-weight: 800; color: {fo_b_lbl};">Fit Confidence</span><div style="font-size: 22px; font-weight: 800; color: {fo_conf_c};">{prob_b:.1f}%</div></div>
                                    <div><span style="font-size: 11px; text-transform: uppercase; font-weight: 800; color: {fo_b_lbl};">Status</span><div style="font-size: 16px; font-weight: 800; color: {'#10b981' if row_b['Status']=='Selected' else '#ef4444'};">{row_b['Status']}</div></div>
                                </div>
                                <div style="font-size: 12px; color: {fo_b_body}; margin-top: 6px;"><b>Key Competencies ({len(skills_b)}):</b> {', '.join(skills_b[:8]) if skills_b else 'None parsed'}</div>
                                <div style="font-size: 12px; opacity: 0.95; color: {fo_b_body}; margin-top: 6px;"><b>Assessment:</b> {row_b.get('Verdict', 'N/A')}</div>
                            </div>
                            """, unsafe_allow_html=True)

                        # Head-to-Head Competency Matrix
                        st.markdown("<br>", unsafe_allow_html=True)
                        st.markdown("##### 🔍 **Head-to-Head Skill Overlap Matrix**")
                        set_a = set(s.lower() for s in skills_a)
                        set_b = set(s.lower() for s in skills_b)
                        all_compared_skills = sorted(list(set(s.lower() for s in (skills_a + skills_b + jd_skills))))

                        matrix_rows = []
                        for s_low in all_compared_skills[:15]:
                            disp_s = s_low.title()
                            has_a = "Yes" if s_low in set_a else "No"
                            has_b = "Yes" if s_low in set_b else "No"
                            in_jd = "Core Role Skill" if (jd_skills and s_low in [j.lower() for j in jd_skills]) else "Supplementary"
                            matrix_rows.append({"Skill / Competency": disp_s, f"{cand_a_name}": has_a, f"{cand_b_name}": has_b, "Role Requirement": in_jd})

                        st.dataframe(pd.DataFrame(matrix_rows), use_container_width=True)

                # VIEW 3: TABULAR DATASET
                else:
                    res_cands = []
                    res_logins = []
                    for _, r in display_df.iterrows():
                        f_txt = st.session_state.candidate_raw_texts.get(r["Candidate"], "")
                        c_name, l_id = resolve_candidate_and_login_id(r["Candidate"], r["Filename"], f_txt)
                        res_cands.append(c_name)
                        res_logins.append(l_id)

                    table_view_df = pd.DataFrame({
                        "Sl. No.": range(1, len(display_df) + 1),
                        "Candidate Name": res_cands,
                        "Login ID": res_logins,
                        "Resume Filename": display_df["Filename"],
                        "Similarity Score": display_df["Similarity"],
                        "Probability Score": display_df["Probability"],
                        "Screening Status": display_df["Status"],
                        "Category": display_df["IsFresher"].apply(lambda f: "Fresher" if f else "Experienced"),
                        "AI Assessment": display_df["Verdict"]
                    })
                    st.dataframe(table_view_df, use_container_width=True, hide_index=True)

                st.markdown("---")

                # Action Bar
                bar_col1, bar_col2 = st.columns([1, 1])
                with bar_col1:
                    if st.button("💾 SAVE EVALUATIONS TO PIPELINE", use_container_width=True):
                        conn = get_conn()
                        c = conn.cursor()
                        now_ts = datetime.now(timezone.utc).isoformat()
                        for _, row in res_df.iterrows():
                            c.execute("""
                                INSERT INTO screening_results(candidate, filename, similarity, probability, status, stage, jd_checksum, timestamp)
                                VALUES (?,?,?,?,?,?,?,?)
                            """, (
                                str(row["Candidate"]),
                                str(row["Filename"]),
                                float(row["Similarity"]),
                                float(row["Probability"]),
                                str(row["Status"]),
                                "Screened",
                                "v2.4",
                                now_ts
                            ))
                        conn.commit()
                        conn.close()
                        st.success("Evaluations recorded into talent pipeline ledger.")

                with bar_col2:
                    csv_export = display_df[["Candidate", "Filename", "Similarity", "Probability", "Status"]].to_csv(index=True).encode("utf-8")
                    st.download_button(
                        label="📥 EXPORT TALENT DATASET (CSV)",
                        data=csv_export,
                        file_name=f"SmartHire_export_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )


        # ----------------------------------------------------
        # TAB 2: VACANCY ANNOUNCEMENTS
        # ----------------------------------------------------
        with hr_nav_vacancies:
            st.markdown("### 📢 **Job Vacancy Announcements**")
            st.caption("Broadcast open technical vacancies to the candidate portal and manage job requisitions.")

            ann_bg = "linear-gradient(135deg, rgba(38, 19, 8, 0.90) 0%, rgba(24, 12, 5, 0.90) 100%)" if is_dark else "linear-gradient(135deg, #fff7ed 0%, #ffedd5 100%)"
            ann_bd = "1.5px solid rgba(249, 115, 22, 0.45)" if is_dark else "1.5px solid rgba(234, 88, 12, 0.35)"
            st.markdown(
                f"""
                <div style="background: {ann_bg}; border: {ann_bd}; border-radius: 12px; padding: 14px 18px; margin: 10px 0 16px 0; box-shadow: 0 4px 18px rgba(0,0,0,0.12);">
                    <div style="display: flex; align-items: center; justify-content: space-between; gap: 12px; flex-wrap: wrap;">
                        <div style="display: flex; align-items: center; gap: 10px;">
                            <span>{get_svg_icon('megaphone', 22, '#ea580c')}</span>
                            <div>
                                <div style="font-size: 15.5px; font-weight: 800; color: {'#fef3c7' if is_dark else '#261308'};">Announce Job Vacancies for Technical Roles</div>
                                <div style="font-size: 12px; color: {'#fed7aa' if is_dark else '#7c2d12'}; margin-top: 2px;">Broadcast active role openings to the candidate portal for real-time eligibility evaluation.</div>
                            </div>
                        </div>
                        <span style="background: rgba(234, 88, 12, 0.2); border: 1px solid #ea580c; color: #ea580c; padding: 4px 12px; border-radius: 12px; font-size: 11.5px; font-weight: 800;">HR Broadcast</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            def on_vacancy_role_change_vac_tab():
                sel_r = st.session_state.get("hr_inp_ann_title")
                if sel_r in TECHNICAL_ROLE_TEMPLATES:
                    matched_d = TECHNICAL_ROLE_TEMPLATES[sel_r]["department"]
                    matched_skills = ", ".join(TECHNICAL_ROLE_TEMPLATES[sel_r]["skills"])
                    st.session_state["hr_inp_ann_dept"] = matched_d
                    st.session_state["hr_inp_ann_dept_synced_role"] = sel_r
                    st.session_state["target_dept"] = matched_d
                    st.session_state["target_role_title"] = sel_r
                    st.session_state["hr_inp_ann_skills"] = matched_skills
                    st.session_state["hr_inp_ann_skills_synced_role"] = sel_r
                    st.session_state["target_skills"] = matched_skills
                    st.session_state["hr_vac_ann_jd_spec"] = TECHNICAL_ROLE_TEMPLATES[sel_r]["text"]

            v_col1, v_col2, v_col3 = st.columns([1.5, 1.2, 1.3])
            with v_col1:
                all_role_options = list(TECHNICAL_ROLE_TEMPLATES.keys())
                default_ann_title = st.session_state.get("target_role_title", all_role_options[0])
                r_sel_idx = all_role_options.index(default_ann_title) if default_ann_title in all_role_options else 0
                ann_title_val = st.selectbox("Vacancy Role Title:", all_role_options, index=r_sel_idx, key="hr_inp_ann_title", on_change=on_vacancy_role_change_vac_tab)

                # Auto-sync department to selected role
                expected_dept = TECHNICAL_ROLE_TEMPLATES[ann_title_val]["department"]
                if "hr_inp_ann_dept" not in st.session_state or st.session_state.get("hr_inp_ann_dept_synced_role") != ann_title_val:
                    st.session_state["hr_inp_ann_dept"] = expected_dept
                    st.session_state["hr_inp_ann_dept_synced_role"] = ann_title_val
                
                curr_dept = st.session_state.get("hr_inp_ann_dept", expected_dept)
                d_sel_idx = ALL_DEPARTMENTS.index(curr_dept) if curr_dept in ALL_DEPARTMENTS else (
                    ALL_DEPARTMENTS.index(expected_dept) if expected_dept in ALL_DEPARTMENTS else 0
                )
                ann_dept_val = st.selectbox("Department:", ALL_DEPARTMENTS, index=d_sel_idx, key="hr_inp_ann_dept")
            with v_col2:
                default_ann_cnt = int(st.session_state.get("target_vacancies", 3))
                ann_cnt_val = st.number_input("Open Vacancies Count:", min_value=1, max_value=50, value=default_ann_cnt, key="hr_inp_ann_cnt")
                exp_options = ["0-2 Years (Associate / Fresher)", "2-4 Years (Mid-Level)", "3-5 Years (Mid-Senior)", "5+ Years (Senior / Lead)"]
                default_exp_idx = 1
                if st.session_state.get("target_exp") in exp_options:
                    default_exp_idx = exp_options.index(st.session_state.get("target_exp"))
                ann_exp_val = st.selectbox("Experience Required:", exp_options, index=default_exp_idx, key="hr_inp_ann_exp")
            with v_col3:
                # Automated Core Skills Sync
                if ann_title_val in TECHNICAL_ROLE_TEMPLATES:
                    expected_skills = ", ".join(TECHNICAL_ROLE_TEMPLATES[ann_title_val]["skills"])
                else:
                    expected_skills = st.session_state.get("target_skills", "React, Node.js, TypeScript, PostgreSQL, Docker, REST APIs, Git")

                if "hr_inp_ann_skills" not in st.session_state or st.session_state.get("hr_inp_ann_skills_synced_role") != ann_title_val:
                    st.session_state["hr_inp_ann_skills"] = expected_skills
                    st.session_state["hr_inp_ann_skills_synced_role"] = ann_title_val
                    st.session_state["target_skills"] = expected_skills

                ann_skills_val = st.text_area(
                    "Required Core Skills:",
                    value=st.session_state.get("hr_inp_ann_skills", expected_skills),
                    height=110,
                    key="hr_inp_ann_skills"
                )

            # Role Specification Text Area (Pre-loaded from role template or customized)
            default_jd_spec = TECHNICAL_ROLE_TEMPLATES.get(ann_title_val, {}).get("text", st.session_state.get("quick_jd_template", ""))
            if "hr_vac_ann_jd_spec" not in st.session_state or st.session_state.get("hr_vac_ann_jd_synced_role") != ann_title_val:
                st.session_state["hr_vac_ann_jd_spec"] = default_jd_spec
                st.session_state["hr_vac_ann_jd_synced_role"] = ann_title_val

            ann_jd_spec = st.text_area(
                "Role Requirements Specification / Job Description:",
                value=st.session_state.get("hr_vac_ann_jd_spec", default_jd_spec),
                height=130,
                key="hr_vac_ann_jd_spec"
            )

            if st.button("📢 PUBLISH VACANCY ANNOUNCEMENT", type="primary", use_container_width=True, key="btn_publish_announcement"):
                final_jd = ann_jd_spec.strip() if ann_jd_spec.strip() else default_jd_spec.strip()
                if not ann_title_val.strip():
                    st.error("Please enter a vacancy role title.")
                elif not final_jd:
                    st.error("Please provide role requirements specification before publishing.")
                else:
                    new_vid = announce_job_vacancy(
                        role_title=ann_title_val.strip(),
                        department=ann_dept_val.strip(),
                        vacancies_count=int(ann_cnt_val),
                        jd_text=final_jd,
                        required_skills=ann_skills_val.strip(),
                        min_exp_years=ann_exp_val,
                        created_by=reviewer_user
                    )
                    st.toast(f"Vacancy for '{ann_title_val}' ({ann_cnt_val} Openings) published!")
                    st.success(f"Vacancy for '{ann_title_val}' ({ann_cnt_val} Openings) officially announced and published to Candidate Portal!")
                    st.rerun()

            # Manage existing vacancies expander
            df_curr_vacancies = get_vacancies(only_open=False)
            if not df_curr_vacancies.empty:
                with st.expander(f"Manage Announced Job Vacancies ({len(df_curr_vacancies)} Active/Archived Roles)", expanded=True):
                    for _, v_r in df_curr_vacancies.iterrows():
                        v_row_id = v_r["id"]
                        v_row_t = v_r["role_title"]
                        v_row_d = v_r["department"]
                        v_row_c = v_r["vacancies_count"]
                        v_row_s = v_r["status"]
                        v_row_dt = v_r["created_at"]
                        
                        m_c1, m_c2, m_c3 = st.columns([3.5, 1.2, 1.0])
                        with m_c1:
                            status_badge = f"<span style='color: #10b981; font-weight:800;'>{get_svg_icon('check-circle', 14, '#10b981')} OPEN</span>" if v_row_s == "Open" else f"<span style='color: #ef4444; font-weight:800;'>{get_svg_icon('x-circle', 14, '#ef4444')} CLOSED</span>"
                            st.markdown(f"**{v_row_t}** ({v_row_c} Openings) — `{v_row_d}` • {status_badge} • *Announced: {v_row_dt}*", unsafe_allow_html=True)
                        with m_c2:
                            toggle_lbl = "🔒 Close Vacancy" if v_row_s == "Open" else "🔓 Reopen Vacancy"
                            if st.button(toggle_lbl, key=f"btn_v_tog_{v_row_id}", use_container_width=True):
                                set_vacancy_status(v_row_id, "Closed" if v_row_s == "Open" else "Open")
                                st.rerun()
                        with m_c3:
                            if st.button("🗑️ Delete", key=f"btn_v_remove_{v_row_id}", use_container_width=True):
                                delete_vacancy(v_row_id)
                                st.rerun()

        # ----------------------------------------------------
        # TAB 3: INBOUND CANDIDATE APPLICATIONS (DIRECT PORTAL SUBMISSIONS)
        # ----------------------------------------------------
        with hr_nav_inbound:
            render_inbound_applications_list(parent_key="inbound_tab")

        # ----------------------------------------------------
        # TAB 3: TALENT PIPELINE & HISTORY
        # ----------------------------------------------------
        # TAB 4: TALENT PIPELINE & HISTORY
        # ----------------------------------------------------
        with hr_nav2:
            st.markdown("### 📈 **Talent Pipeline History**")
            st.caption("Comprehensive evaluation ledger of all candidate screening benchmarks. Live updating every second.")

            p_col1, p_col2, p_col3 = st.columns([1.5, 1.1, 1.4])
            with p_col1:
                st.text_input("Search Candidate Records:", "", key="pipeline_search_inp", placeholder="Search candidate name, email, login ID, or file...")
            with p_col2:
                st.selectbox("Status Filter:", ["All", "Selected", "Rejected"], key="pipeline_status_filter")
            with p_col3:
                st.selectbox(
                    "Sort Candidates:",
                    [
                        "Latest Screened (Newest First)",
                        "Oldest Screened (Oldest First)",
                        "Match Score: High to Low",
                        "Match Score: Low to High",
                        "Candidate Name: A-Z",
                        "Candidate Name: Z-A",
                        "Candidate Email: A-Z",
                        "Candidate Email: Z-A",
                        "Screening Status: Selected First",
                        "Screening Status: Rejected First"
                    ],
                    key="pipeline_sort_selectbox"
                )

            @st.fragment(run_every=1)
            def _pipeline_live_feed():
                p_search = st.session_state.get("pipeline_search_inp", "")
                p_status = st.session_state.get("pipeline_status_filter", "All")
                p_sort = st.session_state.get("pipeline_sort_selectbox", "Latest Screened (Newest First)")

                conn = get_conn()
                df_pipeline = pd.read_sql_query(
                    "SELECT candidate, filename, similarity, probability, status, timestamp FROM screening_results ORDER BY timestamp DESC",
                    conn
                )
                # Fetch text lookup map and blobs from resumes_blob
                c_txt_map = {}
                c_blob_map = {}
                try:
                    cur = conn.cursor()
                    cur.execute("SELECT filename, file_blob, file_text FROM resumes_blob")
                    for row in cur.fetchall():
                        c_txt_map[row["filename"]] = row["file_text"] or ""
                        c_blob_map[row["filename"]] = row["file_blob"]
                except Exception:
                    pass

                # Fetch user email lookup
                user_email_map = {}
                try:
                    cur = conn.cursor()
                    cur.execute("SELECT username, email FROM users")
                    user_email_map = {row["username"]: (row["email"] or "") for row in cur.fetchall()}
                except Exception:
                    pass
                conn.close()

                # Ensure static resumes directory is synced
                for fn, blob in c_blob_map.items():
                    if blob and fn:
                        st_p = os.path.join(STATIC_RESUMES_DIR, fn)
                        if not os.path.exists(st_p):
                            try:
                                with open(st_p, "wb") as f_out:
                                    f_out.write(blob)
                            except Exception:
                                pass

                if df_pipeline.empty:
                    st.info("Pipeline records are empty. Screen candidates in the Active Screening Studio to populate records.")
                else:
                    # Resolve Candidate Names from resumes, Emails, Login IDs and Resume Links
                    p_cand_names = []
                    p_login_ids = []
                    p_emails = []
                    p_resume_links = []
                    for _, r in df_pipeline.iterrows():
                        fn = r["filename"]
                        f_txt = c_txt_map.get(fn, "")
                        c_name, l_id = resolve_candidate_and_login_id(r["candidate"], fn, f_txt)
                        email = extract_candidate_email(f_txt, c_name, l_id, user_email_map)
                        resume_link = f"/app/static/resumes/{fn}"
                        p_cand_names.append(c_name)
                        p_login_ids.append(l_id)
                        p_emails.append(email)
                        p_resume_links.append(resume_link)
                    df_pipeline["candidate_name"] = p_cand_names
                    df_pipeline["login_id"] = p_login_ids
                    df_pipeline["email"] = p_emails
                    df_pipeline["resume_link"] = p_resume_links

                    p_view = df_pipeline.copy()
                    if p_search.strip():
                        q = p_search.strip().lower()
                        p_view = p_view[
                            p_view["candidate_name"].str.lower().str.contains(q, na=False) |
                            p_view["email"].str.lower().str.contains(q, na=False) |
                            p_view["login_id"].str.lower().str.contains(q, na=False) |
                            p_view["filename"].str.lower().str.contains(q, na=False)
                        ]
                    if p_status != "All":
                        p_view = p_view[p_view["status"] == p_status]

                    # Apply sorting dropdown
                    if p_sort == "Latest Screened (Newest First)":
                        p_view = p_view.sort_values(by="timestamp", ascending=False, na_position="last")
                    elif p_sort == "Oldest Screened (Oldest First)":
                        p_view = p_view.sort_values(by="timestamp", ascending=True, na_position="last")
                    elif p_sort == "Match Score: High to Low":
                        p_view = p_view.sort_values(by="similarity", ascending=False, na_position="last")
                    elif p_sort == "Match Score: Low to High":
                        p_view = p_view.sort_values(by="similarity", ascending=True, na_position="last")
                    elif p_sort == "Candidate Name: A-Z":
                        p_view = p_view.sort_values(by="candidate_name", ascending=True, na_position="last")
                    elif p_sort == "Candidate Name: Z-A":
                        p_view = p_view.sort_values(by="candidate_name", ascending=False, na_position="last")
                    elif p_sort == "Candidate Email: A-Z":
                        p_view = p_view.sort_values(by="email", ascending=True, na_position="last")
                    elif p_sort == "Candidate Email: Z-A":
                        p_view = p_view.sort_values(by="email", ascending=False, na_position="last")
                    elif p_sort == "Screening Status: Selected First":
                        p_view = p_view.sort_values(by="status", ascending=False, na_position="last")
                    elif p_sort == "Screening Status: Rejected First":
                        p_view = p_view.sort_values(by="status", ascending=True, na_position="last")

                    if p_view.empty:
                        st.info("No candidates match the specified search or filter criteria.")
                    else:
                        st.markdown("#### 📊 **Candidate Screening & Benchmark Evaluations Table**")
                        display_pipeline = SmartDataFrame({
                            "Sl. No.": range(1, len(p_view) + 1),
                            "Candidate Name": p_view["candidate_name"],
                            "Email": p_view["email"],
                            "Resume File Link": p_view["resume_link"],
                            "Similarity Score": p_view["similarity"],
                            "Probability Score": p_view["probability"],
                            "Screening Status": p_view["status"],
                            "Evaluation Timestamp": p_view["timestamp"]
                        })
                        st.dataframe(
                            display_pipeline,
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                "Resume File Link": st.column_config.LinkColumn(
                                    "Resume File Link",
                                    help="Click to open applicant resume PDF in a new browser tab",
                                    display_text=r".*/([^/]+)"
                                ),
                                "Resume File link": st.column_config.LinkColumn(
                                    "Resume File link",
                                    help="Click to open applicant resume PDF in a new browser tab",
                                    display_text=r".*/([^/]+)"
                                ),
                                "Email": st.column_config.TextColumn(
                                    "Candidate Email",
                                    help="Applicant contact email address"
                                )
                            }
                        )

                        st.download_button(
                            label="📥 DOWNLOAD PIPELINE LEDGER (CSV)",
                            data=display_pipeline.to_csv(index=False).encode("utf-8"),
                            file_name="SmartHire_pipeline_ledger.csv",
                            mime="text/csv",
                            key="btn_dl_pipeline_csv"
                        )

                        st.markdown("##### 📄 **Document Retrieval & Inspection**")
                        selected_pipe_doc = st.selectbox(
                            "Select candidate to inspect or download resume file:",
                            p_view["candidate_name"] + " (" + p_view["filename"] + ")",
                            key="pipe_doc_select"
                        )
                        if selected_pipe_doc:
                            chosen_fn = p_view.iloc[
                                (p_view["candidate_name"] + " (" + p_view["filename"] + ")" == selected_pipe_doc).argmax()
                            ]["filename"]
                            conn = get_conn()
                            cur = conn.cursor()
                            cur.execute("SELECT file_blob, file_text FROM resumes_blob WHERE filename=?", (chosen_fn,))
                            rb_row = cur.fetchone()
                            conn.close()

                            f_blob = rb_row["file_blob"] if rb_row else None
                            f_text = rb_row["file_text"] if rb_row else ""
                            if not f_blob:
                                st_f = os.path.join(STATIC_RESUMES_DIR, chosen_fn)
                                if os.path.exists(st_f):
                                    with open(st_f, "rb") as f_in:
                                        f_blob = f_in.read()

                            p_dl_col1, p_dl_col2 = st.columns([1.5, 2.5])
                            with p_dl_col1:
                                if f_blob:
                                    st.download_button(
                                        label=f"📥 DOWNLOAD {chosen_fn}",
                                        data=f_blob,
                                        file_name=chosen_fn,
                                        key=f"pipe_dl_{chosen_fn}",
                                        use_container_width=True
                                    )
                                else:
                                    st.info(f"File '{chosen_fn}' available in ledger.")
                            with p_dl_col2:
                                with st.expander("Inspect Extracted Resume Text"):
                                    st.text(f_text or "No extracted text preview available.")

            _pipeline_live_feed()

        # ----------------------------------------------------
        # TAB 5: CANDIDATE VAULT
        # ----------------------------------------------------
        with hr_nav3:
            st.markdown("### 📁 **Candidate Document Vault**")
            st.caption("Secure document repository of applicant resumes uploaded into SmartHire. Live updating every second.")

            vf_col1, vf_col2 = st.columns([1.6, 1.4])
            with vf_col1:
                st.text_input("Search Stored Resumes:", "", key="vault_search_inp", placeholder="Search candidate name, role, filename, or document ID...")
            with vf_col2:
                st.selectbox(
                    "Sort Documents:",
                    [
                        "Upload Date: Newest First",
                        "Upload Date: Oldest First",
                        "Candidate Name: A-Z",
                        "Candidate Name: Z-A",
                        "Role Matched: A-Z",
                        "Role Matched: Z-A",
                        "Document ID: Low to High",
                        "Document ID: High to Low",
                        "Login ID: A-Z",
                        "Login ID: Z-A",
                        "File Size: Largest First",
                        "File Size: Smallest First"
                    ],
                    key="vault_sort_selectbox"
                )

            @st.fragment(run_every=1)
            def _vault_live_feed():
                v_search = st.session_state.get("vault_search_inp", "")
                v_sort = st.session_state.get("vault_sort_selectbox", "Upload Date: Newest First")

                conn = get_conn()
                df_vault = pd.read_sql_query(
                    "SELECT id, candidate, filename, file_text, uploaded_at, file_blob, LENGTH(file_blob) as file_size FROM resumes_blob ORDER BY uploaded_at DESC",
                    conn
                )
                # Fetch screening results notes to resolve benchmark matched roles
                cur = conn.cursor()
                cur.execute("SELECT filename, notes FROM screening_results ORDER BY id ASC")
                sr_notes_map = {}
                for sr_fn, sr_n in cur.fetchall():
                    if sr_fn and sr_n:
                        m = re.search(r'Benchmark:\s*([^|\n\r]+)', str(sr_n))
                        if m and m.group(1).strip():
                            sr_notes_map[sr_fn] = m.group(1).strip()

                cur.execute("SELECT title FROM active_job_spec ORDER BY id DESC LIMIT 1")
                ajs_row = cur.fetchone()
                active_spec_role = ajs_row[0].strip() if ajs_row and ajs_row[0] else "Full-Stack Web Developer"
                conn.close()

                # Ensure static resumes directory is synced
                for _, r_b in df_vault.iterrows():
                    fn_b = r_b["filename"]
                    blob_b = r_b.get("file_blob")
                    if blob_b and fn_b:
                        st_p = os.path.join(STATIC_RESUMES_DIR, fn_b)
                        if not os.path.exists(st_p):
                            try:
                                with open(st_p, "wb") as f_out:
                                    f_out.write(blob_b)
                            except Exception:
                                pass

                if df_vault.empty:
                    st.info("No documents found in vault.")
                else:
                    # Resolve Candidate Name, Login ID, Resume Link, and Matched Benchmark Role
                    v_cand_names = []
                    v_login_ids = []
                    v_resume_links = []
                    v_matched_roles = []
                    current_bench_role = st.session_state.get("target_role_title", active_spec_role)

                    for _, r in df_vault.iterrows():
                        c_name, l_id = resolve_candidate_and_login_id(r["candidate"], r["filename"], r.get("file_text", ""))
                        r_link = f"/app/static/resumes/{r['filename']}"
                        m_role = sr_notes_map.get(r["filename"]) or current_bench_role or "Full-Stack Web Developer"
                        v_cand_names.append(c_name)
                        v_login_ids.append(l_id)
                        v_resume_links.append(r_link)
                        v_matched_roles.append(m_role)
                    df_vault["candidate_name"] = v_cand_names
                    df_vault["login_id"] = v_login_ids
                    df_vault["resume_link"] = v_resume_links
                    df_vault["matched_role"] = v_matched_roles

                    v_view = df_vault.copy()
                    if v_search.strip():
                        v_q = v_search.strip().lower()
                        v_view = v_view[
                            v_view["candidate_name"].str.lower().str.contains(v_q, na=False) |
                            v_view["matched_role"].str.lower().str.contains(v_q, na=False) |
                            v_view["filename"].str.lower().str.contains(v_q, na=False) |
                            v_view["login_id"].str.lower().str.contains(v_q, na=False) |
                            v_view["id"].astype(str).str.contains(v_q, na=False)
                        ]

                    # Apply sorting dropdown
                    if v_sort == "Upload Date: Newest First":
                        v_view = v_view.sort_values(by="uploaded_at", ascending=False, na_position="last")
                    elif v_sort == "Upload Date: Oldest First":
                        v_view = v_view.sort_values(by="uploaded_at", ascending=True, na_position="last")
                    elif v_sort == "Candidate Name: A-Z":
                        v_view = v_view.sort_values(by="candidate_name", ascending=True, na_position="last")
                    elif v_sort == "Candidate Name: Z-A":
                        v_view = v_view.sort_values(by="candidate_name", ascending=False, na_position="last")
                    elif v_sort == "Role Matched: A-Z":
                        v_view = v_view.sort_values(by="matched_role", ascending=True, na_position="last")
                    elif v_sort == "Role Matched: Z-A":
                        v_view = v_view.sort_values(by="matched_role", ascending=False, na_position="last")
                    elif v_sort == "Document ID: Low to High":
                        v_view = v_view.sort_values(by="id", ascending=True, na_position="last")
                    elif v_sort == "Document ID: High to Low":
                        v_view = v_view.sort_values(by="id", ascending=False, na_position="last")
                    elif v_sort == "Login ID: A-Z":
                        v_view = v_view.sort_values(by="login_id", ascending=True, na_position="last")
                    elif v_sort == "Login ID: Z-A":
                        v_view = v_view.sort_values(by="login_id", ascending=False, na_position="last")
                    elif v_sort == "File Size: Largest First":
                        v_view = v_view.sort_values(by="file_size", ascending=False, na_position="last")
                    elif v_sort == "File Size: Smallest First":
                        v_view = v_view.sort_values(by="file_size", ascending=True, na_position="last")

                    if v_view.empty:
                        st.info("No documents match the specified search criteria.")
                    else:
                        st.markdown("#### 📁 **Candidate Document Repository & Vault Records Table**")
                        display_vault = SmartDataFrame({
                            "Sl. No.": range(1, len(v_view) + 1),
                            "Document ID": v_view["id"],
                            "Candidate Name": v_view["candidate_name"],
                            "Resume File link": v_view["resume_link"],
                            "Upload Timestamp": v_view["uploaded_at"],
                            "Role Against Which Resumes Are Matched": v_view["matched_role"]
                        })
                        st.dataframe(
                            display_vault,
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                "Resume File link": st.column_config.LinkColumn(
                                    "Resume File link",
                                    help="Click to open applicant resume PDF in a new browser tab",
                                    display_text=r".*/([^/]+)"
                                ),
                                "Resume File Link": st.column_config.LinkColumn(
                                    "Resume File Link",
                                    help="Click to open applicant resume PDF in a new browser tab",
                                    display_text=r".*/([^/]+)"
                                ),
                                "Document ID": st.column_config.NumberColumn(
                                    "Document ID",
                                    help="Unique vault record identifier",
                                    format="%d"
                                ),
                                "Role Against Which Resumes Are Matched": st.column_config.TextColumn(
                                    "Role Against Which Resumes Are Matched",
                                    help="Target job benchmark role against which applicant credentials were evaluated"
                                )
                            }
                        )

                        st.download_button(
                            label="📥 DOWNLOAD VAULT LEDGER (CSV)",
                            data=display_vault.to_csv(index=False).encode("utf-8"),
                            file_name="SmartHire_candidate_vault_ledger.csv",
                            mime="text/csv",
                            key="btn_dl_vault_csv"
                        )

                        st.markdown("##### 📄 **Document Retrieval & Inspection**")
                        selected_vault_doc = st.selectbox(
                            "Select candidate file to retrieve:",
                            v_view["candidate_name"] + " [" + v_view["login_id"] + "] (" + v_view["filename"] + ")"
                        )

                        if selected_vault_doc:
                            chosen_id = v_view.iloc[
                                (v_view["candidate_name"] + " [" + v_view["login_id"] + "] (" + v_view["filename"] + ")" == selected_vault_doc).argmax()
                            ]["id"]

                            conn = get_conn()
                            c = conn.cursor()
                            c.execute("SELECT filename, file_blob, file_text FROM resumes_blob WHERE id=?", (int(chosen_id),))
                            row_doc = c.fetchone()
                            conn.close()

                            if row_doc:
                                dl_col1, dl_col2, dl_col3 = st.columns([1.2, 1.2, 2.6])
                                with dl_col1:
                                    if row_doc["file_blob"]:
                                        st.download_button(
                                            label=f"📥 DOWNLOAD {row_doc['filename']}",
                                            data=row_doc["file_blob"],
                                            file_name=row_doc["filename"],
                                            key=f"vault_dl_{chosen_id}",
                                            use_container_width=True
                                        )
                                with dl_col2:
                                    st.markdown("<div class='delete-btn-box'>", unsafe_allow_html=True)
                                    if st.button("🗑️ DELETE DOCUMENT", key=f"vault_del_{chosen_id}", use_container_width=True):
                                        conn = get_conn()
                                        c = conn.cursor()
                                        c.execute("DELETE FROM resumes_blob WHERE id=?", (int(chosen_id),))
                                        conn.commit()
                                        conn.close()
                                        st.success(f"Document '{row_doc['filename']}' deleted from vault.")
                                        st.rerun()
                                    st.markdown("</div>", unsafe_allow_html=True)
                                with dl_col3:
                                    with st.expander("Inspect Extracted Document Text"):
                                        st.text(row_doc["file_text"] or "No extracted text available.")

            _vault_live_feed()

        # ----------------------------------------------------
        # TAB 4: TEAM & ACCESS
        # ----------------------------------------------------
        with hr_nav4:
            st.markdown("### 🛡️ **Team & Access Control**")
            users_df = get_all_users_df()

            # Map Candidate Name & Login ID for every account
            user_real_names = []
            user_name_lookup = {
                "admin": "HR Administrator",
                "candidate1": "Samantha Ray",
                "candidate3": "Candidate Account 3",
                "hr_recruiter": "HR Recruitment Lead",
                "cand_test_01": "Candidate Testing User",
                "hr_agent_sync": "HR Automated Sync Agent",
                "hr_test_eval": "HR Evaluation Officer",
                "vikram.malhotra": "Vikram Malhotra",
                "elena.rostova": "Dr. Elena Rostova",
                "alex.chen": "Alex Chen",
                "marcus.brody": "Marcus Brody",
                "olivia.taylor": "Olivia Taylor",
                "emily.watson": "Emily Watson",
                "david.miller": "David Miller",
                "rahul.sharma": "Rahul Sharma",
            }
            for _, u_r in users_df.iterrows():
                u_name = u_r["username"]
                r_name = user_name_lookup.get(u_name, u_name.replace('.', ' ').replace('_', ' ').title())
                user_real_names.append(r_name)
            users_df["candidate_name"] = user_real_names
            users_df["login_id"] = users_df["username"]

            u_col1, u_col2 = st.columns([1.6, 1.4])
            with u_col1:
                u_search = st.text_input("Search Users:", "", key="users_search_inp", placeholder="Search candidate name, login ID, role, email...")
            with u_col2:
                u_sort = st.selectbox(
                    "Sort Users:",
                    [
                        "Created: Newest First",
                        "Created: Oldest First",
                        "Candidate Name: A-Z",
                        "Candidate Name: Z-A",
                        "Login ID: A-Z",
                        "Login ID: Z-A",
                        "Role: A-Z",
                        "Role: Z-A"
                    ],
                    key="users_sort_selectbox"
                )

            u_view = users_df.copy()
            if u_search.strip():
                u_q = u_search.strip().lower()
                u_view = u_view[
                    u_view["candidate_name"].str.lower().str.contains(u_q, na=False) |
                    u_view["login_id"].str.lower().str.contains(u_q, na=False) |
                    u_view["role"].str.lower().str.contains(u_q, na=False) |
                    u_view["email"].str.lower().str.contains(u_q, na=False)
                ]

            if u_sort == "Created: Newest First":
                u_view = u_view.sort_values(by="created_at", ascending=False, na_position="last")
            elif u_sort == "Created: Oldest First":
                u_view = u_view.sort_values(by="created_at", ascending=True, na_position="last")
            elif u_sort == "Candidate Name: A-Z":
                u_view = u_view.sort_values(by="candidate_name", ascending=True, na_position="last")
            elif u_sort == "Candidate Name: Z-A":
                u_view = u_view.sort_values(by="candidate_name", ascending=False, na_position="last")
            elif u_sort == "Login ID: A-Z":
                u_view = u_view.sort_values(by="login_id", ascending=True, na_position="last")
            elif u_sort == "Login ID: Z-A":
                u_view = u_view.sort_values(by="login_id", ascending=False, na_position="last")
            elif u_sort == "Role: A-Z":
                u_view = u_view.sort_values(by="role", ascending=True, na_position="last")
            elif u_sort == "Role: Z-A":
                u_view = u_view.sort_values(by="role", ascending=False, na_position="last")

            if u_view.empty:
                st.info("No user accounts match the search criteria.")
            else:
                st.markdown("#### 🛡️ **User Accounts & Access Directory Table**")
                display_users = pd.DataFrame({
                    "Sl. No.": range(1, len(u_view) + 1),
                    "Candidate Name": u_view["candidate_name"],
                    "Login ID": u_view["login_id"],
                    "Account Role": u_view["role"],
                    "Corporate Email": u_view["email"],
                    "Account Created Timestamp": u_view["created_at"]
                })
                st.dataframe(display_users, use_container_width=True, hide_index=True)

            acc_col1, acc_col2 = st.columns([1, 1])

            with acc_col1:
                st.markdown("##### 🗑️ **Remove User Account**")
                usernames_list = users_df["username"].tolist()
                user_to_remove = st.selectbox("Select account to delete:", usernames_list, key="sel_user_to_delete")

                if st.button("🗑️ DELETE USER ACCOUNT", key="btn_del_user_acc", use_container_width=True):
                    if user_to_remove == st.session_state.username:
                        st.error("Cannot delete your own active session account.")
                    else:
                        delete_user(user_to_remove)
                        st.success(f"Account '{user_to_remove}' deleted.")
                        st.rerun()

            with acc_col2:
                st.markdown("##### ➕ **Provision Team Account**")
                with st.expander("➕ Create User Account", expanded=True):
                    nu_name = st.text_input("Username", key="prov_u")
                    nu_pass = st.text_input("Password", type="password", key="prov_p")
                    nu_role = st.selectbox("Assign Role", ["Candidate", "HR"], key="prov_r")
                    nu_email = st.text_input("Corporate Email", key="prov_e")
                    
                    prov_sec_code = ""
                    if nu_role == "HR":
                        prov_sec_code = st.text_input("Company Security Code", type="password", key="prov_sec_code", autocomplete="new-password")

                    if st.button("➕ PROVISION ACCOUNT", key="prov_btn"):
                        if not is_strong_password(nu_pass):
                            st.error("Password must be 8+ chars and contain uppercase, lowercase, digit, and symbol.")
                        elif nu_role == "HR" and prov_sec_code.strip().upper() != "HR-SECURE-2026":
                            st.error("Invalid Company Security Code. HR accounts require authorization.")
                        elif create_user(nu_name, nu_pass, nu_role, nu_email or None):
                            st.success(f"Account '{nu_name}' provisioned successfully.")
                            st.rerun()
                        else:
                            st.error("Username already registered.")

    # ========================================================
    # CANDIDATE APPLICANT PORTAL
    # ========================================================
    elif role == "Candidate":
        st.markdown(f"### 🎓 **Applicant Assessment Portal**")
        st.caption(f"Welcome back, {user}. Track your hiring progress and evaluate ATS fit.")

        # Dynamic Candidate Workspace Access Card
        cand_email_disp = st.session_state.get("email") or f"{user}@example.com"
        cand_init = (user[:2] if len(user) >= 2 else user).upper()
        cand_card_bg = "linear-gradient(135deg, rgba(38, 18, 7, 0.90) 0%, rgba(22, 10, 4, 0.95) 100%)" if is_dark else "linear-gradient(135deg, #fffaf5 0%, #ffedd5 100%)"
        cand_card_bd = "1.5px solid rgba(16, 185, 129, 0.40)" if is_dark else "1.5px solid rgba(16, 185, 129, 0.35)"

        st.markdown(
            f"""
            <div style="background: {cand_card_bg}; border: {cand_card_bd}; border-radius: 14px; padding: 12px 18px; margin: 6px 0 16px 0; box-shadow: 0 4px 18px rgba(0,0,0,0.14);">
                <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 10px;">
                    <div style="display: flex; align-items: center; gap: 12px;">
                        <div style="width: 38px; height: 38px; border-radius: 50%; background: linear-gradient(135deg, #10b981 0%, #f59e0b 100%); display: flex; align-items: center; justify-content: center; font-weight: 800; font-size: 14px; color: #fff; box-shadow: 0 2px 8px rgba(16, 185, 129, 0.35);">
                            {cand_init}
                        </div>
                        <div>
                            <div style="font-size: 15px; font-weight: 800; color: {'#fef3c7' if is_dark else '#261308'}; display: flex; align-items: center; gap: 8px;">
                                {user}
                                <span style="background: rgba(16, 185, 129, 0.2); border: 1px solid #10b981; color: #10b981; padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: 800;">Candidate Portal</span>
                            </div>
                            <div style="font-size: 12px; color: {'#fed7aa' if is_dark else '#7c2d12'}; margin-top: 1px;">
                                Contact: <span style="font-weight: 700; color: {'#fef3c7' if is_dark else '#261308'};">{cand_email_disp}</span> • Access: <span style="color: #10b981; font-weight: 700;">Verified Applicant</span>
                            </div>
                        </div>
                    </div>
                    <span style="background: rgba(16, 185, 129, 0.15); border: 1px solid rgba(16, 185, 129, 0.4); color: #10b981; padding: 4px 12px; border-radius: 12px; font-size: 11px; font-weight: 800; display: inline-flex; align-items: center; gap: 6px;">
                        <span style="width: 7px; height: 7px; border-radius: 50%; background: #10b981; box-shadow: 0 0 6px #10b981;"></span>
                        Portal Session Active
                    </span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        conn = get_conn()
        df_cand_doc = pd.read_sql_query(
            """SELECT id, filename, uploaded_at, file_blob, file_text, 
                      hr_decision, ai_decision, similarity, probability,
                      reviewed_by, reviewed_at
               FROM resumes_blob WHERE candidate=? ORDER BY uploaded_at DESC LIMIT 1""",
            conn,
            params=(user,)
        )
        df_cand_eval = pd.read_sql_query(
            "SELECT filename, similarity, probability, status, timestamp FROM screening_results WHERE candidate=? ORDER BY timestamp DESC LIMIT 1",
            conn,
            params=(user,)
        )
        conn.close()
        has_resume = not df_cand_doc.empty
        has_ev = not df_cand_eval.empty

        cand_hr_dec = "Pending"
        cand_ai_dec = "Selected"
        cand_sim = 0.65
        cand_prob = 0.70

        if has_resume:
            row_doc = df_cand_doc.iloc[0]
            cand_hr_dec = str(row_doc["hr_decision"]) if pd.notna(row_doc.get("hr_decision")) and row_doc.get("hr_decision") else "Pending"
            cand_ai_dec = str(row_doc["ai_decision"]) if pd.notna(row_doc.get("ai_decision")) and row_doc.get("ai_decision") else "Selected"
            cand_sim = float(row_doc["similarity"]) if pd.notna(row_doc.get("similarity")) else 0.65
            cand_prob = float(row_doc["probability"]) if pd.notna(row_doc.get("probability")) else 0.70
        elif has_ev:
            row_ev = df_cand_eval.iloc[0]
            cand_sim = float(row_ev["similarity"])
            cand_prob = float(row_ev["probability"])
            cand_ai_dec = str(row_ev["status"])

        @st.fragment(run_every=1)
        def _candidate_live_stepper_and_status():
            conn = get_conn()
            df_cand_doc = pd.read_sql_query(
                """SELECT id, filename, uploaded_at, file_blob, file_text, 
                          hr_decision, ai_decision, similarity, probability 
                   FROM resumes_blob WHERE candidate=? ORDER BY uploaded_at DESC LIMIT 1""",
                conn,
                params=(user,)
            )
            df_cand_eval = pd.read_sql_query(
                "SELECT filename, similarity, probability, status, timestamp FROM screening_results WHERE candidate=? ORDER BY timestamp DESC LIMIT 1",
                conn,
                params=(user,)
            )
            conn.close()

            has_res = not df_cand_doc.empty
            has_ev = not df_cand_eval.empty

            cand_hr_dec = "Pending"
            cand_ai_dec = "Selected"
            cand_sim = 0.65
            cand_prob = 0.70

            if has_res:
                row_doc = df_cand_doc.iloc[0]
                cand_hr_dec = str(row_doc["hr_decision"]) if pd.notna(row_doc.get("hr_decision")) and row_doc.get("hr_decision") else "Pending"
                cand_ai_dec = str(row_doc["ai_decision"]) if pd.notna(row_doc.get("ai_decision")) and row_doc.get("ai_decision") else "Selected"
                cand_sim = float(row_doc["similarity"]) if pd.notna(row_doc.get("similarity")) else 0.65
                cand_prob = float(row_doc["probability"]) if pd.notna(row_doc.get("probability")) else 0.70
            elif has_ev:
                row_ev = df_cand_eval.iloc[0]
                cand_sim = float(row_ev["similarity"])
                cand_prob = float(row_ev["probability"])
                cand_ai_dec = str(row_ev["status"])

            # 4-Stage Stepper
            step1_class = "token-complete" if has_res else "token-dormant"
            step1_tok = "1"

            step2_class = "token-complete" if (has_res or has_ev) else "token-dormant"
            step2_tok = "2"

            if cand_hr_dec == "Accepted":
                step3_class = "token-complete"
                step3_tok = "3"
                step4_class = "token-active"
                step4_tok = "4"
            elif cand_hr_dec == "Rejected":
                step3_class = "token-dormant"
                step3_tok = "3"
                step4_class = "token-dormant"
                step4_tok = "4"
            else:  # Pending
                step3_class = "token-active" if has_res else "token-dormant"
                step3_tok = "3"
                step4_class = "token-dormant"
                step4_tok = "4"

            st.markdown(
                f"""
                <div class="stepper-deck">
                    <div class="step-unit">
                        <div class="step-token {step1_class}">{step1_tok}</div>
                        <div class="step-descriptor">Application Submitted</div>
                    </div>
                    <div class="step-unit">
                        <div class="step-token {step2_class}">{step2_tok}</div>
                        <div class="step-descriptor">AI Benchmark Screened</div>
                    </div>
                    <div class="step-unit">
                        <div class="step-token {step3_class}">{step3_tok}</div>
                        <div class="step-descriptor">Recruiter Hiring Review</div>
                    </div>
                    <div class="step-unit">
                        <div class="step-token {step4_class}">{step4_tok}</div>
                        <div class="step-descriptor">Interview & Offer Stage</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        _candidate_live_stepper_and_status()

        app_col1, app_col2 = st.columns([1, 1])

        with app_col1:
            @st.fragment(run_every=1)
            def _candidate_active_doc_card():
                conn = get_conn()
                df_cand_doc = pd.read_sql_query(
                    """SELECT id, filename, uploaded_at, file_blob, file_text, 
                              hr_decision, ai_decision, similarity, probability 
                       FROM resumes_blob WHERE candidate=? ORDER BY uploaded_at DESC LIMIT 1""",
                    conn,
                    params=(user,)
                )
                conn.close()

                st.markdown("#### 📄 **Active Resume on File**")
                if not df_cand_doc.empty:
                    c_filename = df_cand_doc.iloc[0]["filename"]
                    c_uploaded_at = df_cand_doc.iloc[0]["uploaded_at"]
                    c_blob = df_cand_doc.iloc[0]["file_blob"]
                    c_text = df_cand_doc.iloc[0]["file_text"] or ""

                    doc_card_bg = "linear-gradient(135deg, rgba(38, 19, 8, 0.95) 0%, rgba(24, 12, 5, 0.95) 100%)" if is_dark else "linear-gradient(135deg, #fff7ed 0%, #ffedd5 100%)"
                    doc_card_bd = "1.5px solid rgba(249, 115, 22, 0.5)" if is_dark else "1.5px solid rgba(234, 88, 12, 0.45)"
                    doc_title_col = "#fef3c7" if is_dark else "#261308"
                    doc_sub_col = "#fed7aa" if is_dark else "#7c2d12"

                    st.markdown(
                        f"""
                        <div style="background: {doc_card_bg}; border: {doc_card_bd}; border-radius: 12px; padding: 16px 18px; margin-bottom: 14px; box-shadow: 0 4px 18px rgba(0,0,0,0.12); max-width: 100%; box-sizing: border-box; overflow: hidden;">
                            <div style="display: flex; align-items: flex-start; justify-content: space-between; gap: 12px; flex-wrap: wrap; max-width: 100%; min-width: 0;">
                                <div style="display: flex; align-items: flex-start; gap: 12px; min-width: 0; flex: 1 1 200px; max-width: 100%; overflow: hidden;">
                                    <div style="width: 42px; height: 42px; min-width: 42px; flex-shrink: 0; border-radius: 10px; background: linear-gradient(135deg, #ea580c, #f59e0b); display: flex; align-items: center; justify-content: center; font-size: 20px; color: #fef3c7; margin-top: 2px;">{get_svg_icon("file-text", 20, "#fef3c7")}</div>
                                    <div style="min-width: 0; flex: 1 1 auto; overflow: hidden; max-width: calc(100% - 54px);">
                                        <div style="font-size: 11px; text-transform: uppercase; font-weight: 800; color: #ea580c; letter-spacing: 0.5px; white-space: nowrap;">Active Resume Document</div>
                                        <div style="font-size: 14px; font-weight: 800; color: {doc_title_col}; margin-top: 3px; font-family: monospace; overflow-wrap: anywhere; word-break: break-all; white-space: normal; line-height: 1.35; max-width: 100%;" title="{c_filename}">{c_filename}</div>
                                    </div>
                                </div>
                                <span style="background: #ea580c; color: #fef3c7; padding: 4px 12px; border-radius: 20px; font-size: 11.5px; font-weight: 800; box-shadow: 0 2px 8px rgba(234, 88, 12, 0.35); flex-shrink: 0; white-space: nowrap; margin-top: 2px;">{get_svg_icon("check-circle", 12, "#fef3c7")} Synced & Parsed</span>
                            </div>
                            <div style="font-size: 12px; color: {doc_sub_col}; margin-top: 12px; font-weight: 700; overflow-wrap: anywhere; word-break: break-word;">
                                <b>Submission Timestamp:</b> {c_uploaded_at}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    skills_detected = extract_skills_from_text(c_text)
                    if skills_detected:
                        st.markdown("🎯 **Parsed Competencies:**")
                        chips = "".join([f"<span class='skill-badge-item'>{s}</span>" for s in skills_detected[:10]])
                        st.markdown(f"<div>{chips}</div>", unsafe_allow_html=True)

                    if c_blob:
                        st.download_button(
                            label="📥 DOWNLOAD SUBMITTED FILE",
                            data=c_blob,
                            file_name=c_filename,
                            key="cand_dl_btn"
                        )
                else:
                    st.info("No active resume found. Submit your resume to begin evaluation.")

            _candidate_active_doc_card()

        with app_col2:
            st.markdown("#### **" + ("📤 Update Document" if has_resume else "📤 Submit Resume") + "**")
            cand_new_file = st.file_uploader(
                "Upload Resume Document (.pdf, .docx, .txt)",
                type=["pdf", "docx", "txt"],
                key="cand_file_up"
            )

            if cand_new_file:
                cand_file_token = f"{cand_new_file.name}_{cand_new_file.size}"
                btn_submit = st.button("🚀 SUBMIT APPLICATION", type="primary", key="cand_btn_submit_app")
                if btn_submit or st.session_state.get(f"synced_file_{user}") != cand_file_token:
                    cand_new_file.seek(0)
                    extracted_txt = extract_text_from_file(cand_new_file)
                    cand_new_file.seek(0)
                    buf_bytes = cand_new_file.read()

                    save_resume_blob(user, cand_new_file.name, buf_bytes, extracted_txt)
                    with open(os.path.join(UPLOAD_DIR, f"{datetime.now(timezone.utc).timestamp()}_{cand_new_file.name}"), "wb") as out:
                        out.write(buf_bytes)
                    st.session_state[f"synced_file_{user}"] = cand_file_token
                    if btn_submit:
                        st.success("Resume updated and transmitted to recruiting pipeline.")
                        st.rerun()
                    else:
                        st.toast(f"Resume '{cand_new_file.name}' synced to Inbound Applications pipeline.")

        # ----------------------------------------------------
        # ANNOUNCED JOB VACANCIES & ELIGIBILITY ASSESSMENT
        # ----------------------------------------------------
        st.markdown("---")
        st.markdown("### 📢 **Active Job Vacancies & Role Eligibility**")
        st.caption("Review active vacancies announced by the recruiting team. Check your real-time eligibility for any open position, view rejection diagnostics, and access tailored learning roadmaps to improve your candidacy.")

        df_open_vacancies = get_vacancies(only_open=True)

        if df_open_vacancies.empty:
            st.info("No open job vacancies have been announced yet. Please check back soon.")
        else:
            v_options = [
                f"{row['role_title']} ({row['vacancies_count']} Open Positions) — {row['department']}"
                for _, row in df_open_vacancies.iterrows()
            ]
            selected_v_idx = st.selectbox(
                "Choose an Announced Job Vacancy to Evaluate Eligibility:",
                range(len(v_options)),
                format_func=lambda i: v_options[i],
                key="cand_vac_picker"
            )
            target_vacancy = df_open_vacancies.iloc[selected_v_idx].to_dict()

            # Vacancy Overview Card
            v_title = target_vacancy["role_title"]
            v_dept = target_vacancy["department"]
            v_count = target_vacancy["vacancies_count"]
            v_exp = target_vacancy["min_exp_years"]
            v_skills_raw = target_vacancy.get("required_skills", "")
            v_skills_list = [s.strip() for s in v_skills_raw.split(",") if s.strip()] if v_skills_raw else extract_skills_from_text(target_vacancy.get("jd_text", ""))

            vac_card_bg = "linear-gradient(135deg, rgba(38, 19, 8, 0.85) 0%, rgba(26, 12, 4, 0.90) 100%)" if is_dark else "linear-gradient(135deg, #fff7ed 0%, #ffedd5 100%)"
            vac_card_bd = "1.5px solid rgba(249, 115, 22, 0.45)" if is_dark else "1.5px solid rgba(234, 88, 12, 0.35)"
            vac_tx_title = "#fef3c7" if is_dark else "#261308"
            vac_tx_sub = "#fed7aa" if is_dark else "#7c2d12"

            v_chips_html = "".join([f"<span class='skill-badge-item'>{s}</span>" for s in v_skills_list[:12]])

            st.markdown(
                f"""
                <div style="background: {vac_card_bg}; border: {vac_card_bd}; border-radius: 12px; padding: 18px 22px; margin: 12px 0 16px 0; box-shadow: 0 4px 18px rgba(0,0,0,0.12);">
                    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
                        <div>
                            <span style="font-size: 11px; text-transform: uppercase; font-weight: 800; color: #ea580c; letter-spacing: 0.5px;">Announced Role Vacancy</span>
                            <div style="font-size: 18px; font-weight: 800; color: {vac_tx_title}; margin-top: 2px;">{v_title}</div>
                            <div style="font-size: 12.5px; color: {vac_tx_sub}; margin-top: 4px; font-weight: 600;">
                                {get_svg_icon('briefcase', 13, '#ea580c')} <b>Dept:</b> {v_dept} &nbsp;•&nbsp; {get_svg_icon('users', 13, '#ea580c')} <b>Open Vacancies:</b> <span style="color: #ea580c; font-weight: 800;">{v_count} Positions</span> &nbsp;•&nbsp; {get_svg_icon('target', 13, '#ea580c')} <b>Experience:</b> {v_exp}
                            </div>
                        </div>
                        <span style="background: #10b981; color: #fef3c7; padding: 5px 14px; border-radius: 20px; font-size: 12px; font-weight: 800; box-shadow: 0 2px 8px rgba(16, 185, 129, 0.35);">
                            {get_svg_icon('check-circle', 13, '#fffaf5')} Active Hiring Open
                        </span>
                    </div>
                    <div style="margin-top: 14px;">
                        <span style="font-size: 11px; text-transform: uppercase; font-weight: 800; color: {vac_tx_sub}; letter-spacing: 0.5px;">Required Core Competencies:</span>
                        <div style="margin-top: 6px; display: flex; flex-wrap: wrap; gap: 4px;">{v_chips_html}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Eligibility Evaluation
            if not has_resume:
                st.info("Please submit your resume above to check your real-time eligibility for this announced vacancy.")
            else:
                c_cand_text = df_cand_doc.iloc[0]["file_text"] or ""
                elig_report = generate_eligibility_report(c_cand_text, target_vacancy)

                is_cand_eligible = elig_report["is_eligible"]
                match_cov = elig_report["coverage"]
                sim_res = elig_report["sim_score"]
                prob_res = elig_report["prob_score"]
                matched_s = elig_report["matched_skills"]
                missing_s = elig_report["missing_skills"]

                if is_cand_eligible:
                    st.markdown(
                        f"""
                        <div style="background: rgba(16, 185, 129, 0.16); border: 2px solid #10b981; border-radius: 12px; padding: 20px 24px; margin: 16px 0; box-shadow: 0 4px 18px rgba(16, 185, 129, 0.15);">
                            <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 14px;">
                                <div style="display: flex; align-items: center; gap: 14px;">
                                    <span>{get_svg_icon("check-circle", 36, "#10b981")}</span>
                                    <div>
                                        <div style="font-size: 18px; font-weight: 800; color: #10b981;">YOU ARE ELIGIBLE FOR THIS ROLE!</div>
                                        <div style="font-size: 13.5px; color: {'#a7f3d0' if is_dark else '#047857'}; margin-top: 3px;">
                                            Congratulations! Your technical qualifications and background meet the qualification benchmark for <b>{v_title}</b> ({v_count} Openings). Your profile has been flagged as <b>Eligible</b> for recruiter review.
                                        </div>
                                    </div>
                                </div>
                                <span style="background: #10b981; color: #fef3c7; padding: 6px 16px; border-radius: 20px; font-size: 12.5px; font-weight: 800;">
                                    {get_svg_icon("check-circle", 12, "#fef3c7")} Shortlist Qualified
                                </span>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"""
                        <div style="background: rgba(239, 68, 68, 0.16); border: 2px solid #ef4444; border-radius: 12px; padding: 20px 24px; margin: 16px 0; box-shadow: 0 4px 18px rgba(239, 68, 68, 0.15);">
                            <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 14px;">
                                <div style="display: flex; align-items: center; gap: 14px;">
                                    <span>{get_svg_icon("alert-triangle", 36, "#ef4444")}</span>
                                    <div>
                                        <div style="font-size: 18px; font-weight: 800; color: #f87171;">NOT CURRENTLY ELIGIBLE FOR THIS ROLE</div>
                                        <div style="font-size: 13.5px; color: {'#fca5a5' if is_dark else '#991b1b'}; margin-top: 3px;">
                                            Based on automated ATS benchmarking against the announced <b>{v_title}</b> vacancy requirements, your profile does not meet the qualifying threshold for this position at this time.
                                        </div>
                                    </div>
                                </div>
                                <span style="background: #ef4444; color: #fef3c7; padding: 6px 16px; border-radius: 20px; font-size: 12.5px; font-weight: 800;">
                                    Below Benchmark Threshold
                                </span>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                # 4 Metrics for Vacancy Evaluation
                em_col1, em_col2, em_col3, em_col4 = st.columns(4)
                em_col1.metric("Eligibility Result", "Eligible" if is_cand_eligible else "Not Eligible")
                em_col2.metric("Technical Match Coverage", f"{match_cov * 100:.1f}%", help="Percentage of required vacancy skills matched")
                em_col3.metric("Role Relevance Score", f"{sim_res * 100:.1f}%")
                em_col4.metric("Shortlist Fit Probability", f"{prob_res * 100:.1f}%")

                if is_cand_eligible:
                    st.success(f"Strong alignment detected in: {', '.join(matched_s[:8])}. To maximize your interview success for '{v_title}', be prepared to discuss deep architectural trade-offs and real-world system resilience.")

        # ----------------------------------------------------
        # OFFICIAL EVALUATION DOSSIER & SUMMARY
        # ----------------------------------------------------
        st.markdown("---")
        st.markdown("### 📋 **Evaluation Dossier & Summary**")

        if not has_resume:
            st.info("Please submit your resume to receive automated ATS benchmark metrics and recruiter review.")
        else:
            if cand_hr_dec == "Accepted":
                st.markdown(
                    """
                    <div style="background: rgba(16, 185, 129, 0.15); border: 1.5px solid #10b981; border-radius: 12px; padding: 14px 18px; margin-bottom: 16px; display: flex; align-items: center; gap: 12px;">
                        <span style="font-size: 26px;">✅</span>
                        <div>
                            <div style="font-weight: 800; font-size: 15px; color: #10b981;">Application Accepted by Hiring Recruiter</div>
                            <div style="font-size: 13px; color: #a7f3d0; margin-top: 2px;">Congratulations! Your application has been approved by the hiring team and moved to the interview shortlist. Our talent acquisition specialist will be in touch regarding interview schedules.</div>
                        </div>
                    </div>
                    """ if is_dark else """
                    <div style="background: #ecfdf5; border: 1.5px solid #059669; border-radius: 12px; padding: 14px 18px; margin-bottom: 16px; display: flex; align-items: center; gap: 12px;">
                        <span style="font-size: 26px;">✅</span>
                        <div>
                            <div style="font-weight: 800; font-size: 15px; color: #065f46;">Application Accepted by Hiring Recruiter</div>
                            <div style="font-size: 13px; color: #047857; margin-top: 2px;">Congratulations! Your application has been approved by the hiring team and moved to the interview shortlist. Our talent acquisition specialist will be in touch regarding interview schedules.</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            elif cand_hr_dec == "Rejected":
                st.markdown(
                    """
                    <div style="background: rgba(239, 68, 68, 0.15); border: 1.5px solid #ef4444; border-radius: 12px; padding: 14px 18px; margin-bottom: 16px; display: flex; align-items: center; gap: 12px;">
                        <span style="font-size: 26px;">❌</span>
                        <div>
                            <div style="font-weight: 800; font-size: 15px; color: #f87171;">Application Status: Not Selected at This Time</div>
                            <div style="font-size: 13px; color: #fca5a5; margin-top: 2px;">Thank you for your interest in our team. After thorough evaluation against active benchmark requirements, the recruitment lead has decided to proceed with other applicants. Your resume remains archived in our talent base for upcoming roles.</div>
                        </div>
                    </div>
                    """ if is_dark else """
                    <div style="background: #fef2f2; border: 1.5px solid #dc2626; border-radius: 12px; padding: 14px 18px; margin-bottom: 16px; display: flex; align-items: center; gap: 12px;">
                        <span style="font-size: 26px;">❌</span>
                        <div>
                            <div style="font-weight: 800; font-size: 15px; color: #991b1b;">Application Status: Not Selected at This Time</div>
                            <div style="font-size: 13px; color: #b91c1c; margin-top: 2px;">Thank you for your interest in our team. After thorough evaluation against active benchmark requirements, the recruitment lead has decided to proceed with other applicants. Your resume remains archived in our talent base for upcoming roles.</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:  # Pending
                st.markdown(
                    """
                    <div style="background: rgba(245, 158, 11, 0.15); border: 1.5px solid #f59e0b; border-radius: 12px; padding: 14px 18px; margin-bottom: 16px; display: flex; align-items: center; gap: 12px;">
                        <span style="font-size: 24px;">⏳</span>
                        <div>
                            <div style="font-weight: 800; font-size: 15px; color: #f59e0b;">Recruiter Hiring Decision: Pending Review</div>
                            <div style="font-size: 13px; color: #fde68a; margin-top: 2px;">Your application has been received and screened by the ATS AI engine. The recruitment lead is reviewing your profile and will update your status soon.</div>
                        </div>
                    </div>
                    """ if is_dark else """
                    <div style="background: #fffbeb; border: 1.5px solid #d97706; border-radius: 12px; padding: 14px 18px; margin-bottom: 16px; display: flex; align-items: center; gap: 12px;">
                        <span style="font-size: 24px;">⏳</span>
                        <div>
                            <div style="font-weight: 800; font-size: 15px; color: #92400e;">Recruiter Hiring Decision: Pending Review</div>
                            <div style="font-size: 13px; color: #b45309; margin-top: 2px;">Your application has been received and screened by the ATS AI engine. The recruitment lead is reviewing your profile and will update your status soon.</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st_col1, st_col2, st_col3, st_col4 = st.columns(4)

            hr_label = "Accepted" if cand_hr_dec == "Accepted" else ("Declined" if cand_hr_dec == "Rejected" else "Pending Review")
            ai_label = "Recommended" if cand_ai_dec == "Selected" else ("Under Evaluation" if cand_ai_dec == "Under Evaluation" else "Below Benchmark")

            st_col1.metric("Recruiter Decision", hr_label)
            st_col2.metric("AI Decision Marker", ai_label)
            st_col3.metric("Role Relevance Score", f"{cand_sim * 100:.1f}%")
            st_col4.metric("Shortlist Fit", f"{cand_prob * 100:.1f}%")

            if has_resume and not df_cand_doc.empty:
                c_filename = df_cand_doc.iloc[0]["filename"]
                cand_skills = extract_skills_from_text(df_cand_doc.iloc[0]["file_text"] or "")
            elif has_ev and not df_cand_eval.empty:
                c_filename = df_cand_eval.iloc[0]["filename"]
                cand_skills = []
            else:
                c_filename = "resume.pdf"
                cand_skills = []

            cand_pdf = build_pdf_report(
                user,
                c_filename,
                cand_sim,
                cand_prob,
                status="Selected" if cand_hr_dec == "Accepted" else ("Rejected" if cand_hr_dec == "Rejected" else cand_ai_dec),
                matched_skills=cand_skills
            )

            st.download_button(
                label="📥 DOWNLOAD OFFICIAL EVALUATION DOSSIER (PDF)",
                data=cand_pdf,
                file_name=f"{user}_SmartHire_dossier.pdf",
                mime="application/pdf",
                key=f"cand_dossier_dl_{user}",
                use_container_width=True
            )

        with st.expander("💡 ATS Optimization Recommendations"):
            st.markdown("""
            - **Include Core Keywords:** Integrate exact technology terms from the target role specification.
            - **Clean Layout:** Use single-column standard formatting to ensure high-fidelity document parsing.
            - **Measurable Metrics:** Quantify your work impact with percentages, throughput figures, or scale metrics.
            """)

        # ----------------------------------------------------
        # SPECIFIC REASONS FOR INELIGIBILITY & SUGGESTIONS SECTION (BOTTOM OF CANDIDATE DASHBOARD)
        # ----------------------------------------------------
        if has_resume and 'is_cand_eligible' in locals() and not is_cand_eligible:
            diag_box_bg = "linear-gradient(135deg, rgba(38, 19, 8, 0.95) 0%, rgba(20, 10, 4, 0.95) 100%)" if is_dark else "linear-gradient(135deg, #fffaf5 0%, #fff1e0 100%)"
            diag_box_bd = "1.5px solid rgba(239, 68, 68, 0.45)" if is_dark else "1.5px solid rgba(220, 38, 38, 0.35)"
            diag_title_c = "#fef3c7" if is_dark else "#261308"
            diag_sub_c = "#fed7aa" if is_dark else "#7c2d12"

            st.markdown(
                f"""
                <div style="background: {diag_box_bg}; border: {diag_box_bd}; border-radius: 12px; padding: 22px 24px; margin: 18px 0; box-shadow: 0 4px 18px rgba(0,0,0,0.12);">
                    <div style="font-size: 16px; font-weight: 800; color: #ef4444; margin-bottom: 12px; display: flex; align-items: center; gap: 8px;">
                        <span>{get_svg_icon("file-text", 18, "#ef4444")}</span> Specific Reasons for Ineligibility / Rejection:
                    </div>
                """,
                unsafe_allow_html=True
            )

            for reason_item in elig_report["rejection_reasons"]:
                st.markdown(f"- **{reason_item}**")

            if missing_s:
                st.markdown("<div style='margin-top: 10px; font-size: 12px; font-weight: 800; color: #f87171;'>Missing Technical Skills Identified:</div>", unsafe_allow_html=True)
                missing_chips = "".join([f"<span style='background: rgba(239, 68, 68, 0.2); border: 1px solid #ef4444; color: #f87171; padding: 3px 9px; border-radius: 6px; font-size: 11.5px; font-weight: 800; margin-right: 5px; display: inline-block; margin-bottom: 5px;'>{get_svg_icon('x-circle', 11, '#f87171')} {s}</span>" for s in missing_s])
                st.markdown(f"<div>{missing_chips}</div>", unsafe_allow_html=True)

            if matched_s:
                st.markdown("<div style='margin-top: 8px; font-size: 12px; font-weight: 800; color: #10b981;'>Skills Successfully Matched:</div>", unsafe_allow_html=True)
                matched_chips = "".join([f"<span style='background: rgba(16, 185, 129, 0.2); border: 1px solid #10b981; color: #10b981; padding: 3px 9px; border-radius: 6px; font-size: 11.5px; font-weight: 800; margin-right: 5px; display: inline-block; margin-bottom: 5px;'>{get_svg_icon('check-circle', 11, '#10b981')} {s}</span>" for s in matched_s])
                st.markdown(f"<div>{matched_chips}</div>", unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

            # DYNAMIC CARD FOR RECRUITER'S DECISION STATUS BAR (PLACED JUST ABOVE SUGGESTION PANEL)
            if has_resume and not df_cand_doc.empty:
                cand_hr_dec = df_cand_doc.iloc[0]["hr_decision"] if ("hr_decision" in df_cand_doc.columns and df_cand_doc.iloc[0]["hr_decision"]) else "Pending"
                cand_reviewer = df_cand_doc.iloc[0]["reviewed_by"] if ("reviewed_by" in df_cand_doc.columns and df_cand_doc.iloc[0]["reviewed_by"]) else "Talent Acquisition Team"
                cand_rev_time = df_cand_doc.iloc[0]["reviewed_at"] if ("reviewed_at" in df_cand_doc.columns and df_cand_doc.iloc[0]["reviewed_at"]) else "Recently Evaluated"
            else:
                cand_hr_dec = "Pending"
                cand_reviewer = "Talent Acquisition Team"
                cand_rev_time = "Recently Evaluated"

            if cand_hr_dec == "Accepted":
                dec_card_bg = "linear-gradient(135deg, rgba(6, 78, 59, 0.35) 0%, rgba(4, 120, 87, 0.20) 100%)" if is_dark else "linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%)"
                dec_card_bd = "2px solid #10b981"
                dec_badge_html = "<span style='background: #10b981; color: #fffaf5; padding: 6px 16px; border-radius: 20px; font-weight: 800; font-size: 13px;'>APPLICATION ACCEPTED & SHORTLISTED</span>"
                dec_icon = get_svg_icon("check-circle", 26, "#10b981")
                dec_headline = "Official Recruiter Decision: Application Accepted"
                dec_body = f"Congratulations! Your application has been officially accepted by {cand_reviewer}. You have been moved to the interview shortlist."
            elif cand_hr_dec == "Rejected":
                dec_card_bg = "linear-gradient(135deg, rgba(127, 29, 29, 0.35) 0%, rgba(185, 28, 28, 0.20) 100%)" if is_dark else "linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%)"
                dec_card_bd = "2px solid #ef4444"
                dec_badge_html = "<span style='background: #ef4444; color: #fffaf5; padding: 6px 16px; border-radius: 20px; font-weight: 800; font-size: 13px;'>APPLICATION NOT SELECTED</span>"
                dec_icon = get_svg_icon("x-circle", 26, "#ef4444")
                dec_headline = "Official Recruiter Decision: Application Declined"
                dec_body = f"Thank you for your interest. {cand_reviewer} has reviewed your profile. While your application was not selected at this time, please review the tailored skill roadmap below to strengthen your profile for future openings."
            else:
                dec_card_bg = "linear-gradient(135deg, rgba(69, 26, 3, 0.40) 0%, rgba(120, 53, 15, 0.25) 100%)" if is_dark else "linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%)"
                dec_card_bd = "2px solid #f59e0b"
                dec_badge_html = "<span style='background: #f59e0b; color: #261308; padding: 6px 16px; border-radius: 20px; font-weight: 800; font-size: 13px;'>IN RECRUITER REVIEW (PENDING DECISION)</span>"
                dec_icon = get_svg_icon("target", 26, "#f59e0b")
                dec_headline = "Official Recruiter Decision: Under Active Review"
                dec_body = "Your submitted resume is currently in the active recruitment queue. The hiring team is evaluating your qualifications against vacancy requirements."

            st.markdown(
                f"""
                <div style="background: {dec_card_bg}; border: {dec_card_bd}; border-radius: 14px; padding: 18px 22px; margin: 18px 0 16px 0; box-shadow: 0 4px 20px rgba(0,0,0,0.15);">
                    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px; margin-bottom: 8px;">
                        <div style="display: flex; align-items: center; gap: 12px;">
                            <span>{dec_icon}</span>
                            <div>
                                <span style="font-size: 11px; text-transform: uppercase; font-weight: 800; color: #ea580c; letter-spacing: 0.5px;">Recruiter Decision Status Bar</span>
                                <div style="font-size: 17px; font-weight: 800; color: {'#fef3c7' if is_dark else '#261308'}; margin-top: 2px;">{dec_headline}</div>
                            </div>
                        </div>
                        {dec_badge_html}
                    </div>
                    <div style="font-size: 13px; color: {'#fed7aa' if is_dark else '#431407'}; line-height: 1.5; margin-top: 6px;">
                        {dec_body}
                    </div>
                    <div style="font-size: 11.5px; color: {'#fed7aa' if is_dark else '#7c2d12'}; margin-top: 8px; opacity: 0.9; font-weight: 600;">
                        Reviewer: <b>{cand_reviewer}</b> • Updated: <b>{cand_rev_time}</b>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # IMPROVEMENT & LEARNING SUGGESTIONS
            sugg_box_bg = "linear-gradient(135deg, rgba(38, 19, 8, 0.95) 0%, rgba(20, 14, 34, 0.95) 100%)" if is_dark else "linear-gradient(135deg, #f0fdf4 0%, #ecfdf5 100%)"
            sugg_box_bd = "1.5px solid rgba(245, 158, 11, 0.45)" if is_dark else "1.5px solid rgba(217, 119, 6, 0.35)"

            st.markdown(
                f"""
                <div style="background: {sugg_box_bg}; border: {sugg_box_bd}; border-radius: 12px; padding: 22px 24px; margin: 18px 0; box-shadow: 0 4px 18px rgba(0,0,0,0.12);">
                    <div style="font-size: 16px; font-weight: 800; color: #f59e0b; margin-bottom: 12px; display: flex; align-items: center; gap: 8px;">
                        <span>{get_svg_icon('book', 18, '#ea580c')}</span> Suggestions on What to Learn & Improve:
                    </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown("##### 💻 **1. Key Technologies to Master First:**")
            for s_item in elig_report["tech_roadmap"]:
                st.markdown(f"- **{s_item}**")

            st.markdown("##### 🛠️ **2. Recommended Portfolio Project to Build:**")
            st.info(f"**Hands-On Recommended Project:** {elig_report['recommended_project']}")

            st.markdown("##### 📝 **3. Resume Content Optimization Advice:**")
            st.markdown("""
            - **Incorporate Exact Keywords:** Ensure that the missing tools and frameworks you learn are explicitly listed in your resume's technical skills section.
            - **Quantify Project Impact:** Replace passive job descriptions with measurable outcomes (e.g., *'Built REST API using FastAPI and Docker, reducing response latency by 35%'*).
            - **One-Click Re-Evaluation:** Once you have updated your resume with these competencies, re-upload it in the section above to immediately re-test your eligibility!
            """)

            st.markdown("</div>", unsafe_allow_html=True)
