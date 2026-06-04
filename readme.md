# Q&A Dataset Quality Metrics

This project implements a reusable Python module to compute quality metrics for a question-answering dataset used for model training and evaluation.

## Dataset Format

The dataset is provided as a JSON file containing a list of records.

Each record should have:

```json
{
  "id": "q_001",
  "question": "What is the capital of France?",
  "answer": "Paris."
}

Metrics ImplementedField completeness
Duplicate quality
Question format quality
Answer quality
Language consistency
Each metric returns a score, total records, passed records, failed record IDs, and supporting details.
Filesquality_metrics.py - Main metrics implementation
run_quality_report.py - Loads a JSON dataset and prints the quality report
sample_dataset.json - Sample Q&A dataset
test_qa_quality_metrics.py - Unit tests for all metrics
__init__.py - Module exports
Run Quality Reportbash



python run_quality_report.py sample_dataset.json

Example result:
Add to chat
total_records: 4
overall_score: 0.75

The sample dataset intentionally includes one duplicate pair and one invalid record, so the report shows failed IDs such as q_001, q_003, and q_004.
Run Testsbash



python test_qa_quality_metrics.py

Latest test result:
Add to chat
Ran 29 tests in 0.011s

OK