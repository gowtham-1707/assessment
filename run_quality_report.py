import argparse
import json
from pathlib import Path
from typing import Any

from quality_metrics import QualityConfig, compute_quality_report


def load_dataset(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError("The dataset JSON file must contain a list of records.")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute quality metrics for a Q&A dataset JSON file."
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        default="sample_dataset.json",
        help="Path to a JSON file containing Q&A records.",
    )
    args = parser.parse_args()

    report = compute_quality_report(load_dataset(Path(args.dataset)), QualityConfig())
    print(json.dumps(report.to_dict(), indent=2))


if __name__ == "__main__":
    main()
