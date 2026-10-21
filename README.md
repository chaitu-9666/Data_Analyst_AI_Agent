# Data Analyst AI Agent

Upload a CSV to explore its contents, review missing values, create charts, and
ask plain-English questions. The current question-answering foundation uses a
deterministic router and an allowlist of Pandas operations; it never executes
Python code from a question or sends uploaded data to an external LLM.

## Run the app

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Supported questions

- Dataset row and column counts
- Columns containing missing values
- Top records by a named column
- Average of a named numeric column
- Strongest correlation between numeric columns
- A bar chart of a numeric total grouped by a category

Questions that do not match a supported operation return guidance instead of
executing arbitrary code. The deterministic router is intended as a safe
foundation before adding optional LLM-based interpretation.

## Run tests

```bash
python -m pytest
```
