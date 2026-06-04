import argparse
import json
from pathlib import Path
from typing import Any

from quality_metrics import QualityConfig, compute_quality_report


def load_qa_dataset(path: Path) -> list[dict[str, Any]]:
    
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError("Dataset JSON must contain a list of Q&A records.")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute quality metrics for a Q&A dataset JSON file."
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        default="sample_qa_dataset.json",
        help="Path to a JSON file containing a list of Q&A records.",
    )
    parser.add_argument(
        "--expected-language",
        default="en",
        choices=("en", "none"),
        help="Expected dataset language. Use 'none' to skip language checks.",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    expected_language = None if args.expected_language == "none" else args.expected_language
    report = compute_quality_report(
        load_qa_dataset(dataset_path),
        QualityConfig(expected_language=expected_language),
    )
    print(json.dumps(report.to_dict(), indent=2))


if __name__ == "__main__":
    main()
