import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

# Ensure src is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from adaptivegraph import LearnableEdge  # noqa: E402


class TestPolicyPersistence(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.policy_path = os.path.join(self.test_dir, "test_policy")

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_save_and_load_policy(self):
        """Test that policy state can be saved and loaded correctly."""
        # 1. Create edge and train it
        edge1 = LearnableEdge(options=["A", "B"], feature_dim=4, exploration_alpha=0.5)

        # Train with some data
        for i in range(10):
            state = "test_state"
            action = edge1(state)
            edge1.record_feedback(result={}, reward=1.0 if action == "A" else 0.0)

        # Save policy
        edge1.save_policy(self.policy_path)

        # Verify file exists
        self.assertTrue(os.path.exists(f"{self.policy_path}.pkl"))

        # 2. Create new edge and load policy
        edge2 = LearnableEdge(options=["A", "B"], feature_dim=4, exploration_alpha=0.5)

        edge2.load_policy(self.policy_path)

        # 3. Verify that A and b matrices match
        np.testing.assert_array_almost_equal(edge1.policy.A, edge2.policy.A)
        np.testing.assert_array_almost_equal(edge1.policy.b, edge2.policy.b)
        self.assertEqual(edge1.policy.alpha, edge2.policy.alpha)

    def test_load_nonexistent_policy(self):
        """Test that loading a nonexistent policy raises FileNotFoundError."""
        edge = LearnableEdge(options=["A", "B"], feature_dim=4)

        with self.assertRaises(FileNotFoundError):
            edge.load_policy("/nonexistent/path")

    def test_load_incompatible_policy(self):
        """Test that loading an incompatible policy raises ValueError."""
        # 1. Create and save policy with 2 actions
        edge1 = LearnableEdge(options=["A", "B"], feature_dim=4)
        edge1.save_policy(self.policy_path)

        # 2. Try to load into edge with 3 actions
        edge2 = LearnableEdge(options=["A", "B", "C"], feature_dim=4)

        with self.assertRaises(ValueError) as context:
            edge2.load_policy(self.policy_path)

        self.assertIn("n_actions mismatch", str(context.exception))

    def test_load_incompatible_feature_dim(self):
        """Test that loading policy with wrong feature_dim raises ValueError."""
        # 1. Create and save policy with feature_dim=4
        edge1 = LearnableEdge(options=["A", "B"], feature_dim=4)
        edge1.save_policy(self.policy_path)

        # 2. Try to load into edge with feature_dim=8
        edge2 = LearnableEdge(options=["A", "B"], feature_dim=8)

        with self.assertRaises(ValueError) as context:
            edge2.load_policy(self.policy_path)

        self.assertIn("feature_dim mismatch", str(context.exception))


class TestInputValidation(unittest.TestCase):
    def test_empty_options(self):
        """Test that empty options list raises ValueError."""
        with self.assertRaises(ValueError) as context:
            LearnableEdge(options=[])

        self.assertIn("non-empty", str(context.exception))

    def test_duplicate_options(self):
        """Test that duplicate options raise ValueError."""
        with self.assertRaises(ValueError) as context:
            LearnableEdge(options=["A", "B", "A"])

        self.assertIn("unique", str(context.exception))

    def test_negative_feature_dim(self):
        """Test that negative feature_dim raises ValueError."""
        with self.assertRaises(ValueError) as context:
            LearnableEdge(options=["A", "B"], feature_dim=-1)

        self.assertIn("positive", str(context.exception))

    def test_zero_feature_dim(self):
        """Test that zero feature_dim raises ValueError."""
        with self.assertRaises(ValueError) as context:
            LearnableEdge(options=["A", "B"], feature_dim=0)

        self.assertIn("positive", str(context.exception))

    def test_negative_alpha(self):
        """Test that negative exploration_alpha raises ValueError."""
        with self.assertRaises(ValueError) as context:
            LearnableEdge(options=["A", "B"], exploration_alpha=-0.5)

        self.assertIn("non-negative", str(context.exception))


class TestRewardValidation(unittest.TestCase):
    def test_nan_reward(self):
        """Test that NaN reward raises ValueError."""
        edge = LearnableEdge(options=["A", "B"], feature_dim=4)
        edge("test")

        with self.assertRaises(ValueError) as context:
            edge.record_feedback(result={}, reward=float("nan"))

        self.assertIn("finite", str(context.exception))

    def test_inf_reward(self):
        """Test that infinite reward raises ValueError."""
        edge = LearnableEdge(options=["A", "B"], feature_dim=4)
        edge("test")

        with self.assertRaises(ValueError) as context:
            edge.record_feedback(result={}, reward=float("inf"))

        self.assertIn("finite", str(context.exception))

    def test_negative_inf_reward(self):
        """Test that negative infinite reward raises ValueError."""
        edge = LearnableEdge(options=["A", "B"], feature_dim=4)
        edge("test")

        with self.assertRaises(ValueError) as context:
            edge.record_feedback(result={}, reward=float("-inf"))

        self.assertIn("finite", str(context.exception))

    def test_nan_in_complete_trace(self):
        """Test that NaN final_reward in complete_trace raises ValueError."""
        edge = LearnableEdge(options=["A", "B"], feature_dim=4)
        edge({"value": "test", "trace_id": "t1"})

        with self.assertRaises(ValueError) as context:
            edge.complete_trace("t1", final_reward=float("nan"))

        self.assertIn("finite", str(context.exception))


class TestEdgeCases(unittest.TestCase):
    def test_single_option(self):
        """Test edge with single option (no real choice)."""
        edge = LearnableEdge(options=["only_option"], feature_dim=4)

        result = edge("test")
        self.assertEqual(result, "only_option")

        edge.record_feedback(result={}, reward=1.0)
        # Should not crash

    def test_feedback_without_call(self):
        """Test that feedback without prior call is handled gracefully."""
        edge = LearnableEdge(options=["A", "B"], feature_dim=4)

        # This should not crash, just do nothing
        edge.record_feedback(result={}, reward=1.0)

        # Memory should be empty
        mem = edge.memory.get_all()
        self.assertEqual(len(mem["actions"]), 0)

    def test_double_feedback(self):
        """Test that double feedback clears state after first one."""
        edge = LearnableEdge(options=["A", "B"], feature_dim=4)
        edge("test")

        edge.record_feedback(result={}, reward=1.0)
        # Second feedback should do nothing (state cleared)
        edge.record_feedback(result={}, reward=0.5)

        # Should have only one entry
        mem = edge.memory.get_all()
        self.assertEqual(len(mem["actions"]), 1)

    def test_complete_trace_nonexistent(self):
        """Test completing a trace that doesn't exist."""
        edge = LearnableEdge(options=["A", "B"], feature_dim=4)

        # Should not crash
        edge.complete_trace("nonexistent_trace", final_reward=1.0)

        # Memory should be empty
        mem = edge.memory.get_all()
        self.assertEqual(len(mem["actions"]), 0)


if __name__ == "__main__":
    unittest.main()
