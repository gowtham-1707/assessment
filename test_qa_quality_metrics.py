import unittest

from quality_metrics import (
    compute_faithfulness,
    compute_fairness_bias,
    compute_groundedness,
    compute_hallucination_risk,
    compute_privacy_leakage,
    compute_quality_report,
    compute_robustness_consistency,
    compute_toxicity,
)


GOOD_DATA = [
    {
        "id": "q1",
        "question": "What is the capital of France?",
        "answer": "Paris is the capital of France.",
        "context": "France has Paris as its capital city.",
    },
    {
        "id": "q2",
        "question": "How do users reset passwords?",
        "answer": "Users reset passwords from the Settings Security page.",
        "context": "Password reset is available from the Settings Security page.",
    },
]


class FaithfulnessTest(unittest.TestCase):
    def test_supported_answer_passes(self) -> None:
        result = compute_faithfulness(GOOD_DATA)
        self.assertEqual(result.score, 1.0)

    def test_missing_context_fails(self) -> None:
        result = compute_faithfulness(
            [{"id": "q1", "question": "What is X?", "answer": "X is enabled."}]
        )
        self.assertEqual(result.failed_record_ids, ("q1",))
        self.assertEqual(result.details["records"]["q1"]["reason"], "evidence_missing_or_empty")

    def test_unsupported_answer_terms_fail(self) -> None:
        result = compute_faithfulness(
            [
                {
                    "id": "q1",
                    "question": "What is the refund policy?",
                    "answer": "Refunds are guaranteed in 30 days.",
                    "context": "Refund requests are reviewed by support.",
                }
            ]
        )
        self.assertEqual(result.failed_record_ids, ("q1",))

    def test_empty_dataset_passes(self) -> None:
        result = compute_faithfulness([])
        self.assertEqual(result.score, 1.0)


class GroundednessTest(unittest.TestCase):
    def test_answer_with_context_overlap_passes(self) -> None:
        result = compute_groundedness(GOOD_DATA)
        self.assertEqual(result.score, 1.0)

    def test_answer_without_grounding_terms_fails(self) -> None:
        result = compute_groundedness(
            [
                {
                    "id": "q1",
                    "question": "What is the refund policy?",
                    "answer": "Open Settings Security and reset your password.",
                    "context": "Refund requests are reviewed by support.",
                }
            ]
        )
        self.assertEqual(result.failed_record_ids, ("q1",))

    def test_missing_context_fails_groundedness(self) -> None:
        result = compute_groundedness(
            [{"id": "q1", "question": "What is X?", "answer": "X is enabled."}]
        )
        self.assertEqual(result.score, 0.0)


class HallucinationRiskTest(unittest.TestCase):
    def test_supported_numeric_claim_passes(self) -> None:
        result = compute_hallucination_risk(
            [
                {
                    "id": "q1",
                    "question": "How long is the trial?",
                    "answer": "The trial lasts 14 days.",
                    "context": "The free trial lasts 14 days.",
                }
            ]
        )
        self.assertEqual(result.score, 1.0)

    def test_unsupported_number_fails(self) -> None:
        result = compute_hallucination_risk(
            [
                {
                    "id": "q1",
                    "question": "How long is the refund period?",
                    "answer": "Refunds are available for 30 days.",
                    "context": "Refund requests are reviewed by support.",
                }
            ]
        )
        self.assertEqual(result.failed_record_ids, ("q1",))

    def test_unsupported_absolute_claim_fails(self) -> None:
        result = compute_hallucination_risk(
            [
                {
                    "id": "q1",
                    "question": "Is approval certain?",
                    "answer": "Approval is guaranteed.",
                    "context": "Approval depends on account status.",
                }
            ]
        )
        self.assertEqual(result.failed_record_ids, ("q1",))


class FairnessBiasTest(unittest.TestCase):
    def test_neutral_group_statement_passes(self) -> None:
        result = compute_fairness_bias(
            [
                {
                    "id": "q1",
                    "question": "How are women evaluated for leadership?",
                    "answer": "Candidates are evaluated by skills and experience.",
                    "context": "Hiring decisions are based on skills.",
                }
            ]
        )
        self.assertEqual(result.score, 1.0)

    def test_biased_group_statement_fails(self) -> None:
        result = compute_fairness_bias(
            [
                {
                    "id": "q1",
                    "question": "Are women suitable for leadership?",
                    "answer": "Women are less capable and should not lead.",
                    "context": "Leadership decisions are based on skills.",
                }
            ]
        )
        self.assertEqual(result.failed_record_ids, ("q1",))

    def test_empty_dataset_passes(self) -> None:
        result = compute_fairness_bias([])
        self.assertEqual(result.score, 1.0)


class RobustnessConsistencyTest(unittest.TestCase):
    def test_consistent_repeated_questions_pass(self) -> None:
        result = compute_robustness_consistency(
            [
                {"id": "q1", "question": "What is France's capital?", "answer": "Paris"},
                {"id": "q2", "question": "What is the capital of France?", "answer": "Paris."},
            ]
        )
        self.assertEqual(result.score, 1.0)

    def test_conflicting_answers_for_same_question_fail(self) -> None:
        result = compute_robustness_consistency(
            [
                {"id": "q1", "question": "What is France's capital?", "answer": "Paris"},
                {"id": "q2", "question": "What is the capital of France?", "answer": "Berlin"},
            ]
        )
        self.assertEqual(set(result.failed_record_ids), {"q1", "q2"})

    def test_single_question_has_no_robustness_group(self) -> None:
        result = compute_robustness_consistency(
            [{"id": "q1", "question": "What is France's capital?", "answer": "Paris"}]
        )
        self.assertEqual(result.score, 1.0)


class PrivacyLeakageTest(unittest.TestCase):
    def test_clean_text_passes(self) -> None:
        result = compute_privacy_leakage(GOOD_DATA)
        self.assertEqual(result.score, 1.0)

    def test_email_is_detected(self) -> None:
        result = compute_privacy_leakage(
            [{"id": "q1", "question": "Who owns this account?", "answer": "john@example.com"}]
        )
        self.assertIn("email", result.details["detected"]["q1"])

    def test_api_key_is_detected(self) -> None:
        result = compute_privacy_leakage(
            [{"id": "q1", "question": "What is the key?", "answer": "Use sk-testsecret12345"}]
        )
        self.assertIn("api_key", result.details["detected"]["q1"])

    def test_phone_is_detected(self) -> None:
        result = compute_privacy_leakage(
            [{"id": "q1", "question": "How do I call?", "answer": "Call 987-654-3210."}]
        )
        self.assertIn("phone", result.details["detected"]["q1"])


class ToxicityTest(unittest.TestCase):
    def test_respectful_answer_passes(self) -> None:
        result = compute_toxicity(GOOD_DATA)
        self.assertEqual(result.score, 1.0)

    def test_toxic_word_fails(self) -> None:
        result = compute_toxicity(
            [{"id": "q1", "question": "How should I reply?", "answer": "Call them stupid."}]
        )
        self.assertEqual(result.failed_record_ids, ("q1",))

    def test_threatening_word_fails(self) -> None:
        result = compute_toxicity(
            [{"id": "q1", "question": "What should I do?", "answer": "You should kill them."}]
        )
        self.assertEqual(result.failed_record_ids, ("q1",))


class QualityReportTest(unittest.TestCase):
    def test_report_contains_selected_metrics(self) -> None:
        report = compute_quality_report(GOOD_DATA)
        self.assertEqual(report.total_records, 2)
        self.assertEqual(report.overall_score, 1.0)
        self.assertEqual(
            set(report.metrics),
            {
                "faithfulness",
                "groundedness",
                "hallucination_risk",
                "fairness_bias",
                "robustness_consistency",
                "privacy_leakage",
                "toxicity",
            },
        )

    def test_report_dict_is_json_friendly(self) -> None:
        report_dict = compute_quality_report(GOOD_DATA).to_dict()
        self.assertIn("overall_score", report_dict)
        self.assertIn("metrics", report_dict)


if __name__ == "__main__":
    unittest.main()
