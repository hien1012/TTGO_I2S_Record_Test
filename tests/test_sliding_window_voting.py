"""Focused tests for hierarchical sliding-window voting."""

from __future__ import annotations

import unittest

import numpy as np

from sliding_window_logmel_dataset import decision_spans, sliding_starts
from train_cnn_from_wav_logmel_cmvn_sliding import (
    resolve_decision_vote,
    resolve_majority_vote,
)


class SlidingWindowVotingTest(unittest.TestCase):
    def test_twenty_second_wav_produces_four_five_second_votes(self) -> None:
        decision_ranges = decision_spans(20 * 16_000, 5 * 16_000, "drop")
        self.assertEqual(len(decision_ranges), 4)
        for start, end in decision_ranges:
            child_starts = sliding_starts(end - start, 16_000, 16_000)
            self.assertEqual(child_starts, [0, 16_000, 32_000, 48_000, 64_000])

    def test_one_second_windows_inside_three_second_decision(self) -> None:
        starts = sliding_starts(48_000, 16_000, 8_000)
        self.assertEqual(starts, [0, 8_000, 16_000, 24_000, 32_000])

    def test_decision_tail_policy(self) -> None:
        self.assertEqual(decision_spans(100, 40, "drop"), [(0, 40), (40, 80)])
        self.assertEqual(
            decision_spans(100, 40, "pad"),
            [(0, 40), (40, 80), (80, 100)],
        )

    def test_strict_majority_requires_more_than_half(self) -> None:
        probabilities = np.asarray(
            [
                [0.8, 0.1, 0.1],
                [0.7, 0.2, 0.1],
                [0.2, 0.7, 0.1],
                [0.6, 0.3, 0.1],
                [0.1, 0.8, 0.1],
            ],
            dtype=np.float32,
        )
        result = resolve_majority_vote(probabilities, 3, "mean-probability")
        self.assertEqual(result["votes"], [3, 2, 0])
        self.assertTrue(result["majority_reached"])
        self.assertEqual(result["final_label_id"], 0)

    def test_no_majority_can_be_rejected(self) -> None:
        probabilities = np.asarray(
            [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1]], dtype=np.float32
        )
        result = resolve_majority_vote(probabilities, 3, "uncertain")
        self.assertEqual(result["votes"], [1, 1, 0])
        self.assertFalse(result["majority_reached"])
        self.assertEqual(result["final_label_id"], -1)

    def test_whole_wav_votes_decision_labels(self) -> None:
        decision_probabilities = np.asarray(
            [
                [0.7, 0.2, 0.1],
                [0.6, 0.3, 0.1],
                [0.2, 0.7, 0.1],
            ],
            dtype=np.float32,
        )
        result = resolve_decision_vote(
            [0, 0, 1], decision_probabilities, 3, "mean-probability"
        )
        self.assertEqual(result["votes"], [2, 1, 0])
        self.assertTrue(result["majority_reached"])
        self.assertEqual(result["final_label_id"], 0)


if __name__ == "__main__":
    unittest.main()
