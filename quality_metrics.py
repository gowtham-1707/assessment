
import re
import string
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any


QARecord = Mapping[str, Any]

_WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?", flags=re.IGNORECASE)
_WHITESPACE_RE = re.compile(r"\s+")
_ENGLISH_HINT_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "can",
        "capital",
        "do",
        "does",
        "for",
        "from",
        "how",
        "in",
        "is",
        "of",
        "the",
        "to",
        "what",
        "when",
        "where",
        "who",
        "why",
    }
)
_NON_LATIN_RE = re.compile(
    r"[\u0400-\u04ff\u0590-\u05ff\u0600-\u06ff\u0900-\u097f\u4e00-\u9fff]"
)


@dataclass(frozen=True)
class QualityConfig:
    """Configuration values shared by the metric functions."""

    required_fields: tuple[str, ...] = ("id", "question", "answer")
    min_question_words: int = 3
    max_question_words: int = 40
    require_question_mark: bool = True
    min_answer_words: int = 1
    max_answer_words: int = 120
    expected_language: str | None = "en"
    placeholder_answers: frozenset[str] = field(
        default_factory=lambda: frozenset(
            {
                "",
                "n/a",
                "na",
                "none",
                "null",
                "unknown",
                "i don't know",
                "not available",
                "todo",
            }
        )
    )


@dataclass(frozen=True)
class MetricResult:
    """Result returned by one quality metric."""

    name: str
    score: float
    total_records: int
    passed_records: int
    failed_record_ids: tuple[str, ...] = ()
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class QualityReport:
    """Combined report returned by `compute_quality_report`."""

    total_records: int
    overall_score: float
    metrics: dict[str, MetricResult]

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_records": self.total_records,
            "overall_score": self.overall_score,
            "metrics": {
                name: result.to_dict()
                for name, result in self.metrics.items()
            },
        }


def compute_field_completeness(
    qa_pairs: Iterable[QARecord],
    config: QualityConfig | None = None,
) -> MetricResult:
    """Check that each record has the required non-empty string fields."""
    config = config or QualityConfig()
    records = tuple(qa_pairs)
    failed: list[str] = []
    missing_by_id: dict[str, list[str]] = {}

    for index, record in enumerate(records):
        record_id = _record_id(record, index)
        missing_fields = [
            field_name
            for field_name in config.required_fields
            if not _is_non_empty_string(_field_value(record, field_name))
        ]
        if missing_fields:
            failed.append(record_id)
            missing_by_id[record_id] = missing_fields

    return _metric_result(
        name="field_completeness",
        total_records=len(records),
        failed_record_ids=failed,
        details={"missing_or_blank_fields": missing_by_id},
    )


def compute_duplicate_quality(
    qa_pairs: Iterable[QARecord],
    config: QualityConfig | None = None,
) -> MetricResult:
    """Check for repeated questions and repeated question-answer pairs."""
    # The config parameter is kept for a consistent metric function signature.
    _ = config
    records = tuple(qa_pairs)
    question_groups: dict[str, list[str]] = defaultdict(list)
    pair_groups: dict[tuple[str, str], list[str]] = defaultdict(list)

    for index, record in enumerate(records):
        record_id = _record_id(record, index)
        question = _normalized_text(_field_value(record, "question"))
        answer = _normalized_text(_field_value(record, "answer"))
        if question:
            question_groups[question].append(record_id)
        if question and answer:
            pair_groups[(question, answer)].append(record_id)

    duplicate_question_ids = _ids_from_duplicate_groups(question_groups.values())
    duplicate_pair_ids = _ids_from_duplicate_groups(pair_groups.values())
    failed = sorted(duplicate_question_ids | duplicate_pair_ids)

    return _metric_result(
        name="duplicate_quality",
        total_records=len(records),
        failed_record_ids=failed,
        details={
            "duplicate_question_ids": tuple(sorted(duplicate_question_ids)),
            "duplicate_pair_ids": tuple(sorted(duplicate_pair_ids)),
        },
    )


def compute_question_format_quality(
    qa_pairs: Iterable[QARecord],
    config: QualityConfig | None = None,
) -> MetricResult:
    """Check that questions are present, reasonably sized, and question-like."""
    config = config or QualityConfig()
    records = tuple(qa_pairs)
    failed: list[str] = []
    reasons: dict[str, list[str]] = {}

    for index, record in enumerate(records):
        record_id = _record_id(record, index)
        question = _field_value(record, "question")
        record_reasons: list[str] = []

        if not _is_non_empty_string(question):
            record_reasons.append("question_missing_or_blank")
        else:
            question_text = str(question).strip()
            word_count = len(_word_tokens(question_text))
            if word_count < config.min_question_words:
                record_reasons.append("question_too_short")
            if word_count > config.max_question_words:
                record_reasons.append("question_too_long")
            if config.require_question_mark and not question_text.endswith("?"):
                record_reasons.append("missing_question_mark")

        if record_reasons:
            failed.append(record_id)
            reasons[record_id] = record_reasons

    return _metric_result(
        name="question_format_quality",
        total_records=len(records),
        failed_record_ids=failed,
        details={"reasons": reasons},
    )


def compute_answer_quality(
    qa_pairs: Iterable[QARecord],
    config: QualityConfig | None = None,
) -> MetricResult:
    """Check that answers are usable and not obvious placeholders."""
    config = config or QualityConfig()
    records = tuple(qa_pairs)
    failed: list[str] = []
    reasons: dict[str, list[str]] = {}

    for index, record in enumerate(records):
        record_id = _record_id(record, index)
        answer = _field_value(record, "answer")
        record_reasons: list[str] = []

        if not _is_non_empty_string(answer):
            record_reasons.append("answer_missing_or_blank")
        else:
            answer_text = str(answer).strip()
            normalized_answer = _normalized_text(answer_text)
            word_count = len(_word_tokens(answer_text))
            if normalized_answer in config.placeholder_answers:
                record_reasons.append("placeholder_answer")
            if word_count < config.min_answer_words:
                record_reasons.append("answer_too_short")
            if word_count > config.max_answer_words:
                record_reasons.append("answer_too_long")

        if record_reasons:
            failed.append(record_id)
            reasons[record_id] = record_reasons

    return _metric_result(
        name="answer_quality",
        total_records=len(records),
        failed_record_ids=failed,
        details={"reasons": reasons},
    )


def compute_language_consistency(
    qa_pairs: Iterable[QARecord],
    config: QualityConfig | None = None,
) -> MetricResult:
    
    config = config or QualityConfig()
    records = tuple(qa_pairs)

    if config.expected_language is None:
        return _metric_result(
            name="language_consistency",
            total_records=len(records),
            failed_record_ids=[],
            details={"expected_language": None, "skipped": True},
        )

    if config.expected_language.lower() != "en":
        raise ValueError("Only expected_language='en' or None is supported.")

    failed: list[str] = []
    reasons: dict[str, str] = {}
    for index, record in enumerate(records):
        record_id = _record_id(record, index)
        question = _field_value(record, "question")
        answer = _field_value(record, "answer")
        combined_text = f"{question or ''} {answer or ''}".strip()
        if not _looks_english(combined_text):
            failed.append(record_id)
            reasons[record_id] = "does_not_match_expected_english"

    return _metric_result(
        name="language_consistency",
        total_records=len(records),
        failed_record_ids=failed,
        details={"expected_language": "en", "reasons": reasons},
    )


def compute_quality_report(
    qa_pairs: Iterable[QARecord],
    config: QualityConfig | None = None,
) -> QualityReport:
    
    config = config or QualityConfig()
    records = tuple(qa_pairs)
    metric_functions = (
        compute_field_completeness,
        compute_duplicate_quality,
        compute_question_format_quality,
        compute_answer_quality,
        compute_language_consistency,
    )
    metrics: dict[str, MetricResult] = {}
    for metric_function in metric_functions:
        result = metric_function(records, config)
        metrics[result.name] = result

    overall_score = _average(result.score for result in metrics.values())
    return QualityReport(
        total_records=len(records),
        overall_score=overall_score,
        metrics=metrics,
    )


def _metric_result(
    name: str,
    total_records: int,
    failed_record_ids: Sequence[str],
    details: dict[str, Any],
) -> MetricResult:
    failed = tuple(failed_record_ids)
    passed = max(total_records - len(failed), 0)
    score = 1.0 if total_records == 0 else passed / total_records
    return MetricResult(
        name=name,
        score=score,
        total_records=total_records,
        passed_records=passed,
        failed_record_ids=failed,
        details=details,
    )


def _record_id(record: Any, index: int) -> str:
    if isinstance(record, Mapping):
        value = record.get("id")
        if _is_non_empty_string(value):
            return str(value).strip()
    return f"index:{index}"


def _field_value(record: Any, field_name: str) -> Any:
    if not isinstance(record, Mapping):
        return None
    return record.get(field_name)


def _is_non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _normalized_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    lowered = value.lower().strip()
    no_punctuation = "".join(
        char for char in lowered if char not in string.punctuation
    )
    return _WHITESPACE_RE.sub(" ", no_punctuation).strip()


def _word_tokens(value: str) -> list[str]:
    return _WORD_RE.findall(value.lower())


def _ids_from_duplicate_groups(groups: Iterable[list[str]]) -> set[str]:
    duplicate_ids: set[str] = set()
    for ids in groups:
        if len(ids) > 1:
            duplicate_ids.update(ids)
    return duplicate_ids


def _looks_english(value: str) -> bool:
    if not value.strip():
        return False
    if _NON_LATIN_RE.search(value):
        return False

    tokens = _word_tokens(value)
    if not tokens:
        return False
    if len(tokens) < 4:
        return True
    return bool(set(tokens) & _ENGLISH_HINT_WORDS)


def _average(values: Iterable[float]) -> float:
    collected = tuple(values)
    if not collected:
        return 0.0
    return sum(collected) / len(collected)
