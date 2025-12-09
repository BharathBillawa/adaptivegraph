import unittest
import numpy as np
import random
import sys
import os

# Ensure src is in path for testing without install
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from adaptivegraph import LearnableEdge

class TestConvergence(unittest.TestCase):
    def test_simple_context_bandit(self):
        """
        Verify that the edge learns to distinguish between two contexts.
        Context A -> Option 0
        Context B -> Option 1
        """
        edge = LearnableEdge(
            options=["option_A", "option_B"],
            feature_dim=4,
            exploration_alpha=1.0
        )
        
        # Training loop
        window = 20
        recent_rewards = []
        
        for i in range(200):
            is_type_A = random.random() < 0.5
            state = "context_A" if is_type_A else "context_B"
            optimal_action = "option_A" if is_type_A else "option_B"
            
            choice = edge(state)
            
            reward = 1.0 if choice == optimal_action else 0.0
            edge.record_feedback(result={}, reward=reward)
            
            recent_rewards.append(reward)
            if len(recent_rewards) > window:
                recent_rewards.pop(0)

            # Check convergence early
            if i > 50 and sum(recent_rewards)/len(recent_rewards) > 0.95:
                break
        
        avg_acc = sum(recent_rewards)/len(recent_rewards)
        print(f"Final Accuracy: {avg_acc}")
        self.assertGreater(avg_acc, 0.8, "Model failed to converge on simple problem")

if __name__ == "__main__":
    unittest.main()

