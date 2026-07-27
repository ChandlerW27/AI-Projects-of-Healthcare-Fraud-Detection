
# CMS Provider Fraud Detection


## Purpose

This project provides an interactive platform for predicting fraud by using Streamlit and a logistic model. 

## Output 
![example](<docs/images/example.png>)

## 1. Activate the environment

```
conda activate venv
```

## 2. Install packages

```
pip install -r requirements.txt
```

## 3. Train the deployment model locally

```
python src/train_logistic.py
```

## 4. Configure the OpenAI explanation

Copy `.env.example` to `.env`:

```text
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-4o-mini
```

The prediction works without an OpenAI API key. The key is used only to explain
flagged cases.

## 5. Run the Streamlit app

From the project root:

```bash
python -m streamlit run app/app.py
```

## App workflow

```text
Enter or upload a provider profile
→ Logistic Regression predicts fraud probability
→ app shows fraud risk and strongest model contributions
→ LLM explains flagged cases using only those grounded model signals
```

## Important limitation

This is a provider-level screening model based on aggregated claim features.
A flagged result is not proof of fraud and requires human review.

The data is from: [text](https://www.kaggle.com/datasets/rohitrox/healthcare-provider-fraud-detection-analysis)
