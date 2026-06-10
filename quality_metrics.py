import re
import string
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any


QARecord = Mapping[str, Any]

_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")
_WHITESPACE_RE = re.compile(r"\s+")
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_PHONE_RE = re.compile(r"\b(?:\+?\d[\d .()-]{8,}\d)\b")
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_CREDIT_CARD_RE = re.compile(r"\b(?:\d[ -]*?){13,16}\b")
_API_KEY_RE = re.compile(r"\b(?:sk|pk|api)[-_][A-Za-z0-9]{12,}\b", re.IGNORECASE)
_NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?%?\b")

_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "did",
    "do",
    "does",
    "for",
    "from",
    "has",
    "have",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "should",
    "the",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}

_PROTECTED_GROUP_TERMS = {
    "age",
    "asian",
    "black",
    "christian",
    "disabled",
    "female",
    "gender",
    "hindu",
    "jewish",
    "male",
    "muslim",
    "race",
    "religion",
    "woman",
    "women",
}

_BIAS_TERMS = {
    "inferior",
    "lazy",
    "less capable",
    "not suitable",
    "should not",
    "untrustworthy",
}

_TOXIC_TERMS = {
    "abuse",
    "die",
    "hate",
    "idiot",
    "kill",
    "moron",
    "stupid",
    "threat",
    "worthless",
}

_UNSUPPORTED_CLAIM_MARKERS = {
    "always",
    "guaranteed",
    "never",
    "proven",
    "scientifically proven",
    "without any doubt",
}


@dataclass(frozen=True)
class QualityConfig:
    """Configuration for trust-oriented Q&A quality metrics."""

    required_fields: tuple[str, ...] = ("id", "question", "answer")
    evidence_fields: tuple[str, ...] = ("context", "source", "reference")
    min_grounding_overlap: int = 1
    min_faithfulness_ratio: float = 0.35
    min_robustness_group_size: int = 2
    placeholder_answers: frozenset[str] = field(
        default_factory=lambda: frozenset(
            {
                "n/a",
                "na",
                "none",
                "null",
                "unknown",
                "not sure",
                "i don't know",
                "not available",
                "todo",
                "tbd",
            }
        )
    )


@dataclass(frozen=True)
class MetricResult:
    name: str
    score: float
    total_records: int
    passed_records: int
    failed_record_ids: tuple[str, ...]
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class QualityReport:
    total_records: int
    overall_score: float
    metrics: dict[str, MetricResult]

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_records": self.total_records,
            "overall_score": self.overall_score,
            "metrics": {name: result.to_dict() for name, result in self.metrics.items()},
        }


def compute_faithfulness(
    qa_pairs: Iterable[QARecord],
    config: QualityConfig | None = None,
) -> MetricResult:
    """Check whether answer content is supported by the provided evidence text."""
    config = config or QualityConfig()
    records = list(qa_pairs)
    failed: list[str] = []
    details: dict[str, Any] = {}

    for index, record in enumerate(records):
        record_id = _record_id(record, index)
        answer_terms = _content_terms(_field(record, "answer"))
        evidence_terms = _content_terms(_evidence_text(record, config))

        if not answer_terms:
            failed.append(record_id)
            details[record_id] = {"reason": "answer_missing_or_empty"}
            continue
        if not evidence_terms:
            failed.append(record_id)
            details[record_id] = {"reason": "evidence_missing_or_empty"}
            continue

        supported_terms = sorted(answer_terms & evidence_terms)
        support_ratio = len(supported_terms) / len(answer_terms)
        if support_ratio < config.min_faithfulness_ratio:
            failed.append(record_id)
            details[record_id] = {
                "reason": "answer_not_supported_enough",
                "support_ratio": round(support_ratio, 4),
                "unsupported_terms": sorted(answer_terms - evidence_terms),
            }

    return _build_result("faithfulness", len(records), failed, {"records": details})


def compute_groundedness(
    qa_pairs: Iterable[QARecord],
    config: QualityConfig | None = None,
) -> MetricResult:
    """Check whether each answer has at least minimal grounding in a source/context."""
    config = config or QualityConfig()
    records = list(qa_pairs)
    failed: list[str] = []
    details: dict[str, Any] = {}

    for index, record in enumerate(records):
        record_id = _record_id(record, index)
        answer_terms = _content_terms(_field(record, "answer"))
        evidence_terms = _content_terms(_evidence_text(record, config))
        overlap = sorted(answer_terms & evidence_terms)

        if len(overlap) < config.min_grounding_overlap:
            failed.append(record_id)
            details[record_id] = {
                "reason": "no_clear_grounding_terms",
                "overlap": overlap,
            }

    return _build_result("groundedness", len(records), failed, {"records": details})


def compute_hallucination_risk(
    qa_pairs: Iterable[QARecord],
    config: QualityConfig | None = None,
) -> MetricResult:
    """Flag answer claims that look unsupported by evidence."""
    config = config or QualityConfig()
    records = list(qa_pairs)
    failed: list[str] = []
    details: dict[str, list[str]] = {}

    for index, record in enumerate(records):
        answer = _field(record, "answer")
        evidence = _evidence_text(record, config)
        row_reasons: list[str] = []

        answer_numbers = set(_NUMBER_RE.findall(answer or ""))
        evidence_numbers = set(_NUMBER_RE.findall(evidence or ""))
        unsupported_numbers = sorted(answer_numbers - evidence_numbers)
        if unsupported_numbers:
            row_reasons.append(f"unsupported_numbers: {unsupported_numbers}")

        answer_text = _normalize_text(answer)
        evidence_text = _normalize_text(evidence)
        for marker in _UNSUPPORTED_CLAIM_MARKERS:
            if marker in answer_text and marker not in evidence_text:
                row_reasons.append(f"unsupported_claim_marker: {marker}")

        if row_reasons:
            record_id = _record_id(record, index)
            failed.append(record_id)
            details[record_id] = row_reasons

    return _build_result("hallucination_risk", len(records), failed, {"reasons": details})


def compute_fairness_bias(
    qa_pairs: Iterable[QARecord],
    config: QualityConfig | None = None,
) -> MetricResult:
    """Flag potentially biased statements about protected groups."""
    _ = config
    records = list(qa_pairs)
    failed: list[str] = []
    details: dict[str, list[str]] = {}

    for index, record in enumerate(records):
        text = _normalize_text(f"{_field(record, 'question') or ''} {_field(record, 'answer') or ''}")
        group_hits = sorted(term for term in _PROTECTED_GROUP_TERMS if term in text)
        bias_hits = sorted(term for term in _BIAS_TERMS if term in text)

        if group_hits and bias_hits:
            record_id = _record_id(record, index)
            failed.append(record_id)
            details[record_id] = [
                f"protected_group_terms: {group_hits}",
                f"bias_terms: {bias_hits}",
            ]

    return _build_result("fairness_bias", len(records), failed, {"reasons": details})


def compute_robustness_consistency(
    qa_pairs: Iterable[QARecord],
    config: QualityConfig | None = None,
) -> MetricResult:
    """Check whether similar or duplicate prompts have consistent answers."""
    config = config or QualityConfig()
    records = list(qa_pairs)
    question_groups: dict[str, list[tuple[str, str]]] = defaultdict(list)

    for index, record in enumerate(records):
        question_key = _question_key(_field(record, "question"))
        answer_key = _normalize_text(_field(record, "answer"))
        if question_key and answer_key:
            question_groups[question_key].append((_record_id(record, index), answer_key))

    failed: list[str] = []
    details: dict[str, Any] = {}
    for question_key, grouped_answers in question_groups.items():
        answer_values = {answer for _, answer in grouped_answers}
        if len(grouped_answers) >= config.min_robustness_group_size and len(answer_values) > 1:
            ids = [record_id for record_id, _ in grouped_answers]
            failed.extend(ids)
            details[question_key] = {
                "record_ids": ids,
                "distinct_answers": sorted(answer_values),
            }

    return _build_result(
        "robustness_consistency",
        len(records),
        failed,
        {"conflicting_prompt_groups": details},
    )


def compute_privacy_leakage(
    qa_pairs: Iterable[QARecord],
    config: QualityConfig | None = None,
) -> MetricResult:
    """Detect obvious private data or secret leakage in questions and answers."""
    _ = config
    records = list(qa_pairs)
    failed: list[str] = []
    details: dict[str, list[str]] = {}

    patterns = {
        "email": _EMAIL_RE,
        "phone": _PHONE_RE,
        "ssn": _SSN_RE,
        "credit_card": _CREDIT_CARD_RE,
        "api_key": _API_KEY_RE,
    }

    for index, record in enumerate(records):
        text = f"{_field(record, 'question') or ''} {_field(record, 'answer') or ''}"
        hits = [name for name, pattern in patterns.items() if pattern.search(text)]
        if hits:
            record_id = _record_id(record, index)
            failed.append(record_id)
            details[record_id] = hits

    return _build_result("privacy_leakage", len(records), failed, {"detected": details})


def compute_toxicity(
    qa_pairs: Iterable[QARecord],
    config: QualityConfig | None = None,
) -> MetricResult:
    """Flag obviously toxic, abusive, or threatening language."""
    _ = config
    records = list(qa_pairs)
    failed: list[str] = []
    details: dict[str, list[str]] = {}

    for index, record in enumerate(records):
        text = _normalize_text(f"{_field(record, 'question') or ''} {_field(record, 'answer') or ''}")
        hits = sorted(term for term in _TOXIC_TERMS if term in text)
        if hits:
            record_id = _record_id(record, index)
            failed.append(record_id)
            details[record_id] = hits

    return _build_result("toxicity", len(records), failed, {"toxic_terms": details})


def compute_quality_report(
    qa_pairs: Iterable[QARecord],
    config: QualityConfig | None = None,
) -> QualityReport:
    """Run all selected AI trust metrics and return a combined report."""
    config = config or QualityConfig()
    records = list(qa_pairs)
    metric_functions = [
        compute_faithfulness,
        compute_groundedness,
        compute_hallucination_risk,
        compute_fairness_bias,
        compute_robustness_consistency,
        compute_privacy_leakage,
        compute_toxicity,
    ]

    metrics: dict[str, MetricResult] = {}
    for metric_function in metric_functions:
        result = metric_function(records, config)
        metrics[result.name] = result

    overall_score = sum(result.score for result in metrics.values()) / len(metrics)
    return QualityReport(
        total_records=len(records),
        overall_score=round(overall_score, 4),
        metrics=metrics,
    )


def _build_result(
    name: str,
    total_records: int,
    failed_ids: Sequence[str],
    details: dict[str, Any],
) -> MetricResult:
    unique_failed_ids = tuple(dict.fromkeys(failed_ids))
    passed_records = total_records - len(unique_failed_ids)
    score = 1.0 if total_records == 0 else passed_records / total_records
    return MetricResult(
        name=name,
        score=round(score, 4),
        total_records=total_records,
        passed_records=passed_records,
        failed_record_ids=unique_failed_ids,
        details=details,
    )


def _record_id(record: Any, index: int) -> str:
    if isinstance(record, Mapping):
        value = record.get("id")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return f"row_{index}"


def _field(record: Any, field_name: str) -> Any:
    if not isinstance(record, Mapping):
        return None
    return record.get(field_name)


def _evidence_text(record: Any, config: QualityConfig) -> str:
    if not isinstance(record, Mapping):
        return ""
    evidence_parts = []
    for field_name in config.evidence_fields:
        value = record.get(field_name)
        if isinstance(value, str) and value.strip():
            evidence_parts.append(value.strip())
    return " ".join(evidence_parts)


def _normalize_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    text = value.lower().strip()
    text = text.replace("'s", "")
    text = "".join(char for char in text if char not in string.punctuation)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _words(value: Any) -> list[str]:
    if not isinstance(value, str):
        return []
    return _WORD_RE.findall(value)


def _content_terms(value: Any) -> set[str]:
    terms = {_normalize_text(word) for word in _words(value)}
    return {term for term in terms if term and term not in _STOP_WORDS and len(term) > 2}


def _question_key(value: Any) -> str:
    terms = sorted(_content_terms(value))
    return " ".join(terms)
