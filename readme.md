# Q&A Dataset Quality Metrics

This project implements a small reusable Python module that checks the quality of a question-answer dataset used for model training or evaluation.

The dataset is expected to be a JSON file containing a list of Q&A records.

## Dataset Format

Each record should contain:

```json
{
  "id": "q_001",
  "question": "What is the capital of France?",
  "answer": "Paris."
}
```

## Files

| File | Purpose |
| --- | --- |
| `quality_metrics.py` | Main implementation of all quality metrics |
| `run_quality_report.py` | Loads a JSON dataset and prints a quality report |
| `sample_dataset.json` | Example dataset used for demonstration |
| `test_qa_quality_metrics.py` | Unit tests for all metrics |
| `__init__.py` | Module export file |

## Metrics Implemented

| Metric | What it checks | Why it matters |
| --- | --- | --- |
| Field completeness | Checks that `id`, `question`, and `answer` exist and are non-empty strings | Missing or blank fields make records unusable for training/evaluation |
| Duplicate quality | Finds repeated questions and repeated Q&A pairs after normalization | Duplicates can bias training and inflate evaluation results |
| Question format quality | Checks question length and whether questions look question-like | Poorly formatted questions often indicate bad data extraction or labeling |
| Answer quality | Checks blank answers, placeholder answers like `N/A`, and answer length | Weak answers reduce the usefulness of supervised training data |
| Language consistency | Checks whether text appears to match the expected language, English by default | Mixed-language records can reduce dataset consistency |

## Assumptions

- The input dataset is a JSON list of records.
- Each record should have `id`, `question`, and `answer`.
- `question` and `answer` should be strings.
- Empty or whitespace-only values are treated as invalid.
- The default expected language is English.
- Empty datasets are handled safely and do not crash the program.

## Trade-offs

- The language check uses a lightweight heuristic instead of an external language detection library.
- The metrics check data quality, not factual correctness.
- Duplicate detection normalizes punctuation and casing, so similar repeated questions can be caught.
- The project uses only Python standard library modules, so no package installation is required.

## Dependencies

No external dependencies are required.

Recommended:

```text
Python 3.10+
```

## How to Run the Quality Report

Open a terminal in the project folder:

```powershell
cd E:\gowtham\CERAI\Assessment
```

Run:

```powershell
python run_quality_report.py sample_dataset.json
```

Expected summary from the sample dataset:

```text
total_records: 4
overall_score: 0.75
```

The sample dataset intentionally includes:

- a duplicate Q&A pair: `q_001` and `q_003`
- an invalid blank question: `q_004`
- a placeholder answer: `q_004`

So the report correctly flags these records.

## How to Run Tests

Run:

```powershell
python test_qa_quality_metrics.py
```

Expected result:

```text
Ran 29 tests in 0.011s

OK
```

## Report Output

The report is printed as JSON and includes:

- `total_records`
- `overall_score`
- one section per metric
- metric score
- passed record count
- failed record IDs
- detailed failure reasons

Example:

```json
{
  "total_records": 4,
  "overall_score": 0.75
}
```

## Conclusion

This module provides a simple and reusable way to detect common quality issues in Q&A datasets before using them for model training or evaluation.
