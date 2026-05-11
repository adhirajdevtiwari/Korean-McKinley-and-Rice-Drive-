import math

# =========================
# SKILL ALIASES
# =========================

SKILL_ALIASES = {

    # Languages
    "python": "python",
    "pyhton": "python",
    "java": "java",
    "javascript": "javascript",
    "javascrpit": "javascript",
    "js": "javascript",
    "typescript": "typescript",
    "typescrpit": "typescript",
    "c++": "cpp",
    "cpp": "cpp",
    "r": "r",
    "kotlin": "kotlin",

    # ML / Data
    "machinelearning": "machine_learning",
    "machine learning": "machine_learning",
    "ml": "machine_learning",
    "sklearn": "machine_learning",
    "deeplearning": "deep_learning",
    "deep learning": "deep_learning",
    "deep-learning": "deep_learning",
    "tensorflow": "tensorflow",
    "pytorch": "pytorch",
    "keras": "keras",
    "nlp": "nlp",
    "bert": "bert",
    "xgboost": "xgboost",
    "feature engineering": "feature_engineering",
    "statistics": "statistics",
    "stats": "statistics",
    "regression": "regression",
    "clustering": "clustering",
    "data-viz": "data_visualization",
    "data visualization": "data_visualization",
    "data viz": "data_visualization",
    "matplotlib": "data_visualization",
    "tableau": "data_visualization",
    "power-bi": "data_visualization",
    "power bi": "data_visualization",
    "powerbi": "data_visualization",
    "pandas": "pandas",
    "numpy": "numpy",

    # Frontend
    "react": "react",
    "reacts": "react",
    "reactjs": "react",
    "vue": "vue",
    "vue.js": "vue",
    "vuejs": "vue",
    "redux": "redux",
    "tailwind": "tailwind",
    "html/css": "html_css",
    "html css": "html_css",
    "html": "html_css",
    "css": "html_css",
    "jest": "jest",
    "graphql": "graphql",

    # Backend
    "node.js": "nodejs",
    "nodejs": "nodejs",
    "node js": "nodejs",
    "flask": "flask",
    "spring boot": "spring_boot",
    "springboot": "spring_boot",
    "rest api": "rest_api",
    "rest": "rest_api",
    "restapi": "rest_api",
    "microservices": "microservices",

    # Databases
    "sql": "sql",
    "mysql": "mysql",
    "mysq": "mysql",
    "postgresql": "postgresql",
    "postgres": "postgresql",
    "mongodb": "mongodb",
    "redis": "redis",

    # DevOps
    "docker": "docker",
    "kubernetes": "kubernetes",
    "kubernates": "kubernetes",
    "k8s": "kubernetes",
    "ci/cd": "ci_cd",
    "cicd": "ci_cd",
    "ci cd": "ci_cd",
    "aws": "aws",

    # Mobile
    "android": "android",
    "firebase": "firebase",

    # CS
    "algorithms": "algorithms",
    "algoritms": "algorithms",
    "data structure": "data_structures",
    "data structures": "data_structures",
    "competitive programming": "competitive_programming",

    # Design
    "ui/ux": "ui_ux",
    "ui ux": "ui_ux",
    "figma": "figma",
}



resumes = {
    "Arjun Sharma":
    "Pyhton, MachineLearning, SQL, pandas, numpy, Deep-learning",

    "Priya Nair":
    "JavaScrpit, Reacts, Node.JS, MongoDb, REST api, HTML/CSS",

    "Rahul Gupta":
    "Java, Spring Boot, MySql, Microservices, Docker, kubernates",

    "Sneha Patel":
    "Python, TensorFlow, Keras, NLP, BERT, data-viz, matplotlib",

    "Vikram Singh":
    "C++, Algoritms, Data Structure, competitive programming, python",

    "Ananya Krishnan":
    "javascript, vue.js, python, flask, PostgreSQL, AWS, CI/CD",

    "Karan Mehta":
    "Python, Sklearn, XGboost, feature engineering, SQL, tableau",

    "Deepika Rao":
    "Java, Android, Kotlin, Firebase, REST, UI/UX, figma",

    "Aditya Kumar":
    "Reactjs, TypeScrpit, GraphQL, redux, tailwind, nodejs, jest",

    "Meera Iyer":
    "python, R, statistics, ML, regression, clustering, Power-BI"
}



jds = {

    "JD-1 Kakao ML Engineer":
    "Python, Machine Learning, Deep Learning, TensorFlow, PyTorch, SQL, Data Visualization, NLP, BERT, Feature Engineering, Statistics",

    "JD-2 Naver Backend Engineer":
    "Java, Spring Boot, MySQL, PostgreSQL, Microservices, Docker, Kubernetes, REST API, CI/CD, Redis",

    "JD-3 Line Frontend Engineer":
    "JavaScript, React, Vue, TypeScript, REST API, HTML/CSS, Node.js, GraphQL, Redux, Jest, AWS"
}



def normalize_skills(skill_string):

    tokens = skill_string.lower().split(",")

    normalized = []

    for token in tokens:

        token = token.strip()

        if token in SKILL_ALIASES:
            normalized.append(SKILL_ALIASES[token])

    return list(set(normalized))




normalized_resumes = {}

for name, skills in resumes.items():
    normalized_resumes[name] = normalize_skills(skills)



vocab_set = set()

for skills in normalized_resumes.values():
    vocab_set.update(skills)

vocabulary = sorted(list(vocab_set))



df = {}

for word in vocabulary:

    count = 0

    for skills in normalized_resumes.values():
        if word in skills:
            count += 1

    df[word] = count

resume_vectors = {}

for name, skills in normalized_resumes.items():

    vector = []

    total_skills = len(skills)

    for word in vocabulary:

        if word in skills:

            tf = 1 / total_skills

            idf = math.log(10 / df[word])

            tfidf = tf * idf

        else:
            tfidf = 0

        vector.append(tfidf)

    resume_vectors[name] = vector



normalized_jds = {}

for jd_name, jd_skills in jds.items():
    normalized_jds[jd_name] = normalize_skills(jd_skills)

jd_vectors = {}

for jd_name, skills in normalized_jds.items():

    vector = []

    for word in vocabulary:

        if word in skills:
            vector.append(1)
        else:
            vector.append(0)

    jd_vectors[jd_name] = vector



def cosine_similarity(a, b):

    dot_product = 0

    for i in range(len(a)):
        dot_product += a[i] * b[i]

    magnitude_a = math.sqrt(sum(x * x for x in a))
    magnitude_b = math.sqrt(sum(x * x for x in b))

    if magnitude_a == 0 or magnitude_b == 0:
        return 0

    return dot_product / (magnitude_a * magnitude_b)


for jd_name, jd_vector in jd_vectors.items():

    scores = []

    for candidate, resume_vector in resume_vectors.items():

        score = cosine_similarity(resume_vector, jd_vector)

        scores.append((candidate, round(score, 2)))


    scores.sort(key=lambda x: (-x[1], x[0]))

    print("\n" + jd_name)

    for candidate, score in scores[:3]:
        print(f"{candidate}({score:.2f})")