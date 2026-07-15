# Research Paper Acceptance Classifier

Lightweight classifier demo that trains three models (Naive Bayes, Logistic Regression, Decision Tree) on a synthetic research-paper dataset and saves evaluation visualizations.

Quickstart

1. Create a virtual environment and install requirements:

```bash
python -m venv venv
venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

2. Run the script:

```bash
python "research_paper_acceptance_classifier.py"
```

3. Run the Streamlit app:

```bash
streamlit run app.py
```

4. Output images will be in the `outputs/` folder.

Notes
- This project uses regex-based tokenization to avoid network downloads (no NLTK required).
- The dataset is synthetic and intended for demo/education purposes; replace `generate_dataset()` with a real dataset for production experiments.
