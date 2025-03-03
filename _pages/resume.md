---
title: "Resume"
permalink: /resume
author_profile: true
classes: wide
header:
  image: ../assets/images/resume_cover.jpg
---
<style>
  .vertical-line {
    border-left: 2px solid #000;
    height: 100%;
    position: absolute;
    left: 50%;
    margin-left: -1px;
    top: 0;
  }
  .experience-container {
    position: relative;
    padding-left: 20px;
  }
</style>

## 1. Experience
---

### Bell <img src="https://upload.wikimedia.org/wikipedia/commons/9/91/Bell_logo.svg" alt="Bell logo" align="right" width="120" height="120"/>

<div class="experience-container">
  <div class="vertical-line"></div>
  
  **Senior Data Scientist - Full-time**  
  <span id="bell-senior-dates">Mar 2024 - Present</span>  
  *Remote*  

  - Leveraging machine learning algorithms to enhance customer-agent pairing and improve overall customer experience.
  - Focused on optimizing the matching process to ensure that each caller is efficiently connected with the most suitable agent, resulting in increased sales, reduced deactivations, and higher customer satisfaction.
  - The data-driven approach to customer service optimization significantly contributed to the company's revenue generating units (RGU's), while also elevating the level of customer engagement and loyalty.

  **Data Scientist - Full-time**  
  <span id="bell-data-dates">Nov 2022 - Aug 2024</span>  
  *Remote*  

  - Bell is Canada's largest telecommunications company, providing Mobile phone, TV, high-speed and wireless Internet, and residential Home phone services.
</div>


---

### FYCOMPUTIG / IREVOLUTION <img src="../assets/images/irevolution-logo-dark.svg" alt="IRevolution logo" align="right" width="200" height="200"/>

**Data Scientist - Full-time**  
<span id="fycomputig-dates">Aug 2020 - Present</span>  
*Rabat, Rabat-Salé-Kenitra, Morocco*  

Data Scientist contributing to the development of Data Science/AI solutions:

**Ziwig Health:**
- Collection, processing, and visualization of patient data.
- Development of ML models to diagnose patients suffering from chronic diseases (Endometriosis) based on features extracted from RNA analysis.
- Developed an AI assistant to help answer questions about the solution developed.
- Data creation, adaptation, and augmentation for the AI assistant using different approaches.
- Deployment of the AI assistant on multiple platforms such as WhatsApp, Messenger, React-Native app.
- Enriching platform with other production-ready solutions such as Search Engine, Article Recommendation using modern NLP solution: Haystack.
- Created a disease entity extraction model for both French and English languages (SpaCy/CamemBert).
- Dashboarding: Creation of a dashboard using ElasticSearch and Kibana, to help follow the platform development: users increases, survey responding, etc.

**Attrition Risk Platform:**
- Collection, processing, and visualization of employees data.
- Aggregation of data using different techniques: Sliding Window Aggregation, Last Window Aggregation.
- Feature selection, transformation, and normalization.
- Model building using different algorithms: Xgboost, Random Forest, etc.
- Handling imbalanced dataset using different approaches: Undersampling techniques, SMOTE.
- Hyperparameter tuning with GridSearch/RandomizedSearch/Bayesian optimization.
- Evaluation and models comparison with respect to the business use case.
- Using Model Explainability techniques: Shapley Values to explain model prediction.
- Model deployment on AWS EC2 instances, Docker, FastAPI.
- Built a monitoring dashboard to better measure key model performance metrics.

**Skills:** Machine Learning · Natural Language Processing (NLP) · Amazon EC2

---

### FYCOMPUTIG / IREVOLUTION <img src="../assets/images/irevolution-logo-dark.svg" alt="IRevolution logo" align="right" width="200" height="200"/>

**Data Scientist - Internship**  
<span id="fycomputig-intern-dates">Feb 2020 - Aug 2020</span>  
*Rabat, Rabat-Salé-Kenitra, Morocco*  

**Project: Star5 Intelligence**
- An AI SaaS solution based on Natural Language Processing to track and measure consumer perception of brands and products.

**Tasks realized:**
- Multi-Label Classification Engine: classify reviews.
- Keyword Extraction Engine: extract keyword from reviews.
- Text Similarity Engine: calculate similarity between two segments of text.
- Text Segmentation Engine: segmenting reviews into sentences.
- Sentiment Analysis Engine: detecting sentiment of sentences.
- Text Summarization Engine: summarizing documents and reviews.
- Named Entity Recognition Engines: building a specific use case for NER.

**Deep Learning Architecture:**
- RNN | LSTM | GRU | SEQ2SEQ | Attention Mechanisms | Transformers.

**Transformers Models used:**
- BERT | DistillBert | ULMFiT | T5 | CamemBERT.

**Cloud Providers:**
- AWS: Amazon SageMaker | AWS Lambda | AWS API Gateway.
- Azure: Azure VM's.

**Other:**
- Word Embeddings: Glove | FastText | Word2vec.
- Libraries: PyTorch | Tensorflow | Transformers library by Hugging Face | Fastai | NLTK | Spacy | Scikit-Learn.
- Model deployment: AWS SageMaker EndPoints & API's | Flask APP on a docker container.
- Version Controlling: Git (GitLab).
- Other techniques: Data Augmentation, Data Labeling.

**Skills:** Machine Learning · Natural Language Processing (NLP) · Amazon Web Services (AWS) · AWS SageMaker

## 2. Education
---

### Abdelmalek Essadi University <img src="../assets/images/logo_uae.png" alt="University logo" align="right" width="200" height="200"/>

**Master's degree, DATA SCIENCE AND BIG DATA**  
*2018 - 2020*  

Data Science, Big Data Architecture, Big Data Analytics, Data Mining, Machine Learning, NoSQL Databases, Statistics, Cloud Computing & Virtualization, IoT & Artificial Intelligence, Computer Vision, Databases Administration...

Faculty of Science and Technology, Tangier, Morocco (www.fstt.ac.ma)

---

### Sultan Moulay Slimane University <img src="../assets/images/logo_usms.png" alt="University logo" align="right" width="200" height="200"/>

**Bachelor's degree, Computer Science**  
*2017 - 2018*  
*Grade: Quite Good - B+*  

Java, C, JEE, Conception (UML & Merise), .Net, Oracle, SQL SERVER, Information System, Networking, Web Development, Project management...

Faculty of Science and Technology Béni Mellal, B.P: 523 Béni - Mellal Maroc

<script>
  function calculateExperience(startDate, endDate) {
    const start = new Date(startDate);
    const end = endDate.toLowerCase() === 'present' ? new Date() : new Date(endDate);
    const diff = new Date(end - start);
    const years = diff.getUTCFullYear() - 1970;
    const months = diff.getUTCMonth();
    return `${years} yr ${months} mo`;
  }

  document.getElementById('bell-senior-dates').innerHTML += ` · ${calculateExperience('2024-03-01', 'Present')}`;
  document.getElementById('bell-data-dates').innerHTML += ` · ${calculateExperience('2022-11-01', '2024-08-01')}`;
  document.getElementById('fycomputig-dates').innerHTML += ` · ${calculateExperience('2020-08-01', 'Present')}`;
  document.getElementById('fycomputig-intern-dates').innerHTML += ` · ${calculateExperience('2020-02-01', '2020-08-01')}`;
</script>