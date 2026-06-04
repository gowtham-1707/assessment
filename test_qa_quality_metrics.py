import unittest

from quality_metrics import (
    QualityConfig,
    compute_answer_quality,
    compute_duplicate_quality,
    compute_field_completeness,
    compute_language_consistency,
    compute_quality_report,
    compute_question_format_quality,
)


VALID_DATASET = [
    {"id": "q1", "question": "What is the capital of France?", "answer": "Paris."},
    {
        "id": "q2",
        "question": "How do users reset passwords?",
        "answer": "Use Settings Security.",
    },
]


class FieldCompletenessTest(unittest.TestCase):
    def test_valid_dataset_scores_one(self) -> None:
        result = compute_field_completeness(VALID_DATASET)
        self.assertEqual(result.score, 1.0)
        self.assertEqual(result.failed_record_ids, ())

    def test_empty_dataset_scores_one(self) -> None:
        result = compute_field_completeness([])
        self.assertEqual(result.score, 1.0)
        self.assertEqual(result.total_records, 0)

    def test_missing_answer_field_is_reported(self) -> None:
        result = compute_field_completeness(
            [{"id": "q1", "question": "What is France's capital?"}]
        )
        self.assertEqual(result.score, 0.0)
        self.assertEqual(result.failed_record_ids, ("q1",))
        self.assertEqual(result.details["missing_or_blank_fields"]["q1"], ["answer"])

    def test_blank_question_is_reported(self) -> None:
        result = compute_field_completeness(
            [{"id": "q1", "question": "   ", "answer": "Paris"}]
        )
        self.assertEqual(result.failed_record_ids, ("q1",))

    def test_malformed_non_mapping_record_is_reported_by_index(self) -> None:
        result = compute_field_completeness(["not a dict"])
        self.assertEqual(result.score, 0.0)
        self.assertEqual(result.failed_record_ids, ("index:0",))


class DuplicateQualityTest(unittest.TestCase):
    def test_unique_dataset_scores_one(self) -> None:
        result = compute_duplicate_quality(VALID_DATASET)
        self.assertEqual(result.score, 1.0)

    def test_empty_dataset_scores_one(self) -> None:
        result = compute_duplicate_quality([])
        self.assertEqual(result.score, 1.0)

    def test_duplicate_question_marks_both_records(self) -> None:
        result = compute_duplicate_quality(
            [
                {"id": "q1", "question": "What is Paris?", "answer": "A city."},
                {"id": "q2", "question": "what is paris", "answer": "A capital."},
            ]
        )
        self.assertEqual(result.score, 0.0)
        self.assertEqual(set(result.failed_record_ids), {"q1", "q2"})

    def test_duplicate_exact_pair_is_reported(self) -> None:
        result = compute_duplicate_quality(
            [
                {"id": "q1", "question": "What is France's capital?", "answer": "Paris"},
                {"id": "q2", "question": "What is France's capital?", "answer": "Paris."},
                {"id": "q3", "question": "What is Germany's capital?", "answer": "Berlin"},
            ]
        )
        self.assertAlmostEqual(result.score, 1 / 3)
        self.assertEqual(set(result.details["duplicate_pair_ids"]), {"q1", "q2"})

    def test_invalid_blank_questions_do_not_create_duplicate_groups(self) -> None:
        result = compute_duplicate_quality(
            [
                {"id": "q1", "question": "", "answer": "A"},
                {"id": "q2", "question": "", "answer": "A"},
            ]
        )
        self.assertEqual(result.score, 1.0)


class QuestionFormatQualityTest(unittest.TestCase):
    def test_valid_questions_score_one(self) -> None:
        result = compute_question_format_quality(VALID_DATASET)
        self.assertEqual(result.score, 1.0)

    def test_empty_dataset_scores_one(self) -> None:
        result = compute_question_format_quality([])
        self.assertEqual(result.score, 1.0)

    def test_missing_question_scores_zero(self) -> None:
        result = compute_question_format_quality([{"id": "q1", "answer": "Paris"}])
        self.assertEqual(result.score, 0.0)
        self.assertIn("question_missing_or_blank", result.details["reasons"]["q1"])

    def test_question_without_question_mark_fails_by_default(self) -> None:
        result = compute_question_format_quality(
            [
                {
                    "id": "q1",
                    "question": "What is the capital of France",
                    "answer": "Paris",
                }
            ]
        )
        self.assertEqual(result.failed_record_ids, ("q1",))
        self.assertIn("missing_question_mark", result.details["reasons"]["q1"])

    def test_question_mark_rule_can_be_disabled(self) -> None:
        config = QualityConfig(require_question_mark=False)
        result = compute_question_format_quality(
            [
                {
                    "id": "q1",
                    "question": "What is the capital of France",
                    "answer": "Paris",
                }
            ],
            config,
        )
        self.assertEqual(result.score, 1.0)

    def test_too_short_question_fails(self) -> None:
        result = compute_question_format_quality(
            [{"id": "q1", "question": "Capital?", "answer": "Paris"}]
        )
        self.assertEqual(result.score, 0.0)
        self.assertIn("question_too_short", result.details["reasons"]["q1"])


class AnswerQualityTest(unittest.TestCase):
    def test_valid_answers_score_one(self) -> None:
        result = compute_answer_quality(VALID_DATASET)
        self.assertEqual(result.score, 1.0)

    def test_empty_dataset_scores_one(self) -> None:
        result = compute_answer_quality([])
        self.assertEqual(result.score, 1.0)

    def test_blank_answer_scores_zero(self) -> None:
        result = compute_answer_quality(
            [
                {
                    "id": "q1",
                    "question": "What is the capital of France?",
                    "answer": "   ",
                }
            ]
        )
        self.assertEqual(result.score, 0.0)
        self.assertIn("answer_missing_or_blank", result.details["reasons"]["q1"])

    def test_placeholder_answer_fails(self) -> None:
        result = compute_answer_quality(
            [
                {
                    "id": "q1",
                    "question": "What is the capital of France?",
                    "answer": "N/A",
                }
            ]
        )
        self.assertEqual(result.failed_record_ids, ("q1",))
        self.assertIn("placeholder_answer", result.details["reasons"]["q1"])

    def test_too_long_answer_fails_with_configured_threshold(self) -> None:
        config = QualityConfig(max_answer_words=3)
        result = compute_answer_quality(
            [
                {
                    "id": "q1",
                    "question": "What is the password reset process?",
                    "answer": "Open settings security and follow the reset flow.",
                }
            ],
            config,
        )
        self.assertEqual(result.score, 0.0)
        self.assertIn("answer_too_long", result.details["reasons"]["q1"])


class LanguageConsistencyTest(unittest.TestCase):
    def test_english_dataset_scores_one(self) -> None:
        result = compute_language_consistency(VALID_DATASET)
        self.assertEqual(result.score, 1.0)

    def test_empty_dataset_scores_one(self) -> None:
        result = compute_language_consistency([])
        self.assertEqual(result.score, 1.0)

    def test_non_latin_text_scores_zero(self) -> None:
        result = compute_language_consistency(
            [{"id": "q1", "question": "法国的首都是哪里？", "answer": "巴黎"}]
        )
        self.assertEqual(result.score, 0.0)
        self.assertEqual(result.failed_record_ids, ("q1",))

    def test_blank_combined_text_fails(self) -> None:
        result = compute_language_consistency(
            [{"id": "q1", "question": "", "answer": ""}]
        )
        self.assertEqual(result.score, 0.0)

    def test_language_check_can_be_skipped(self) -> None:
        config = QualityConfig(expected_language=None)
        result = compute_language_consistency(
            [{"id": "q1", "question": "法国的首都是哪里？", "answer": "巴黎"}],
            config,
        )
        self.assertEqual(result.score, 1.0)
        self.assertTrue(result.details["skipped"])

    def test_unsupported_language_raises_clear_error(self) -> None:
        with self.assertRaises(ValueError):
            compute_language_consistency(
                VALID_DATASET,
                QualityConfig(expected_language="fr"),
            )


class QualityReportTest(unittest.TestCase):
    def test_report_runs_all_metrics(self) -> None:
        report = compute_quality_report(VALID_DATASET)
        self.assertEqual(report.total_records, 2)
        self.assertEqual(report.overall_score, 1.0)
        self.assertEqual(
            set(report.metrics),
            {
                "field_completeness",
                "duplicate_quality",
                "question_format_quality",
                "answer_quality",
                "language_consistency",
            },
        )

    def test_report_to_dict_is_serializable_shape(self) -> None:
        report_dict = compute_quality_report(VALID_DATASET).to_dict()
        self.assertEqual(report_dict["total_records"], 2)
        self.assertIn("metrics", report_dict)


if __name__ == "__main__":
    unittest.main()
