# AI Projects for Healthcare Fraud Detection
 
## Overview

This repository contains AI projects for healthcare fraud detection, fraud education, and healthcare question answering.

The projects use machine learning (ML), large language models (LLMs), and Retrieval-Augmented Generation (RAG) to analyze healthcare fraud information and build simple AI applications.

## Total Four Projects:

### 1. CMS Fraud Information Extraction

**See Folder:** Extraction of CMS Fraud Information

A Jupyter Notebook that explores CMS healthcare fraud types and common fraud detection methods.

### 2. Healthcare Fraud Detection App

**See Folder:** Deployment_of_Healthcare_Fraud_Detection

A Streamlit application that predicts provider fraud risk using a Logistic Regression model.

<br>

**Workflow:**

```text
Provider information
        ↓
Fraud prediction
        ↓
Fraud risk score
        ↓
AI explanation for flagged cases
```

The prediction model can run without an OpenAI API key. OpenAI is used only to explain flagged cases.

### 3. CMS Healthcare Fraud Assistant

**See Folder:** CMS_Healthcare_Fraud_Assistant

A RAG chatbot that answers questions about healthcare fraud, waste, abuse, and CMS program integrity.

It uses official CMS webpages and PDFs to provide source-based answers.

Main technologies:

- Python
- Flask
- OpenAI
- LangChain
- Pinecone
- Sentence Transformers
- BM25

Example questions:

- What is healthcare fraud?
- How can fraud be detected?
- How do I report Medicare fraud?
- What are common fraud warning signs?

### 4. Allergy AI Assistant

**See Folder:** Allergy_AI_Assistant_Final

An additional healthcare RAG chatbot for allergy education.

It retrieves information from trusted healthcare sources and provides answers with citations.


## Technologies

- Python
- Machine Learning
- Logistic Regression
- LLMs
- RAG
- LangChain
- Pinecone
- Flask
- Streamlit
- OpenAI
- Jupyter Notebook

## How to Run

Each project has its own setup instructions and requirements.

Open the folder for the project you want to run and follow its `README.md`.

## Purpose

These projects demonstrate how machine learning and generative AI can be used to:

- Detect possible healthcare fraud
- Explain fraud-risk predictions
- Retrieve healthcare fraud information
- Build source-grounded healthcare AI assistants

## Disclaimer

These projects are for educational and demonstration purposes only.


