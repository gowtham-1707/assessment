# Q&A Dataset Quality Metrics

This project implements a small reusable Python module for evaluating the quality of a question-answering dataset.

I selected trust-focused metrics because Q&A data is often used to train or evaluate AI assistants. In that setting, the biggest risks are not only missing fields or duplicates. The more serious risks are unsupported answers, hallucinated claims, biased responses, privacy leaks, toxic language, and inconsistent answers for similar questions.

## Final Metrics Selected

Only the following seven metrics are implemented in the final version:

1. Faithfulness
2. Groundedness
3. Hallucination risk
4. Fairness / bias
5. Robustness consistency
6. Privacy leakage
7. Toxicity

No other quality metrics are included in the final report.

## Required Dataset Format

The assessment gives a basic Q&A format with `id`, `question`, and `answer`.

For these trust-focused metrics, I extend that format with one evidence field called `context`. This is required because faithfulness and groundedness cannot be measured honestly unless there is some source text to compare the answer against.

Each record should look like this:

```json
{
  "id": "q_001",
  "question": "What is the capital of France?",
  "answer": "Paris is the capital of France.",
  "context": "France has Paris as its capital city."
}
```

The code also accepts `source` or `reference` as evidence fields, but `context` is the preferred field in the sample dataset.

## Files

| File | Purpose |
| --- | --- |
| `quality_metrics.py` | Main implementation of all selected metrics |
| `run_quality_report.py` | Loads a JSON dataset and prints the quality report |
| `sample_dataset.json` | Example dataset with valid and intentionally risky records |
| `dataset_template.json` | Minimal valid dataset example |
| `test_qa_quality_metrics.py` | Unit tests for the metrics |

## Metric Choices and Reasoning

| Metric | What it measures | Why I chose it |
| --- | --- | --- |
| Faithfulness | Whether the answer is supported by the provided context | A Q&A model should not add details that are not present in the source evidence |
| Groundedness | Whether the answer can be traced back to the context at a basic term level | This is important for RAG and support-style assistants where answers should come from provided material |
| Hallucination risk | Unsupported numbers, percentages, and absolute claims such as `guaranteed` or `always` | Numbers and strong claims are common places where AI systems hallucinate |
| Fairness / bias | Biased statements involving protected groups | Q&A data should not train or evaluate a model using discriminatory answers |
| Robustness consistency | Whether similar or repeated questions receive conflicting answers | A reliable model should be stable when the same question is asked in a slightly different way |
| Privacy leakage | Emails, phone numbers, SSNs, credit-card-like strings, and API-key-like secrets | Datasets should not contain personal or sensitive information that a model may memorize or reveal |
| Toxicity | Insulting, abusive, or threatening language | Training or evaluation data should avoid unsafe or unprofessional responses |

## Assumptions

- The input is a JSON file containing a list of records.
- Each record should have `id`, `question`, `answer`, and `context`.
- `id`, `question`, `answer`, and `context` should be strings.
- `context` is treated as the evidence for the answer.
- Empty datasets are handled safely.
- The metrics are heuristic checks, not full human judgment.
- The implementation uses only the Python standard library.

## Trade-offs

- Faithfulness and groundedness use lexical overlap instead of a semantic entailment model. This keeps the solution simple, explainable, and dependency-free.
- Hallucination risk focuses on high-risk patterns such as unsupported numbers and absolute claims. It does not detect every possible hallucination.
- Fairness and toxicity use small keyword lists. In production, these should be replaced with stronger classifiers.
- Privacy leakage uses regular expressions for common sensitive data patterns.
- Robustness consistency checks normalized question terms. It catches simple paraphrase conflicts, but not all semantic paraphrases.

## Dependencies

No external packages are required.

Recommended:

```text
Python 3.10+
```

## Run the Quality Report

From the project folder:

```powershell
python run_quality_report.py sample_dataset.json
```

The output is a JSON report with:

- `total_records`
- `overall_score`
- metric-level scores
- failed record IDs
- detailed reasons for failures

## Run Tests

From the project folder:

```powershell
python test_qa_quality_metrics.py
```

Expected result:

```text
Ran 25 tests

OK
```

## Sample Dataset Notes

The sample dataset intentionally includes:

- faithful and grounded answers
- an unsupported refund claim
- a privacy leakage example
- a biased answer
- a toxic answer
- inconsistent answers for the same question

This makes it easier to verify that every selected metric is working.
