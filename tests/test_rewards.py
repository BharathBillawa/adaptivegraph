import unittest
from adaptivegraph.core import LearnableEdge
from adaptivegraph.rewards import ErrorScorer
import numpy as np

class TestRewardsAndAsync(unittest.TestCase):
    def test_id_based_feedback(self):
        # 1. Setup Edge
        edge = LearnableEdge(options=["A", "B"], feature_dim=4)
        
        # 2. Call with event_id
        state = {"value": "test", "event_id": "cust_123"}
        action = edge(state)
        
        # 3. Verify it is pending
        self.assertIn("cust_123", edge.pending_decisions)
        
        # 4. Record feedback with ID
        edge.record_feedback(result={}, reward=1.0, event_id="cust_123")
        
        # 5. Verify it is cleared
        self.assertNotIn("cust_123", edge.pending_decisions)
        
        # 6. Verify memory updated
        mem = edge.memory.get_all()
        self.assertEqual(len(mem["actions"]), 1)
        self.assertEqual(mem["rewards"][0], 1.0)

    def test_error_scorer(self):
        scorer = ErrorScorer(penalty=-5.0)
        
        bad_state = {"error": "Something went wrong"}
        self.assertEqual(scorer.score(bad_state), -5.0)
        
        good_state = {"result": "ok"}
        self.assertEqual(scorer.score(good_state), 1.0)
        
    def test_trajectory_reward(self):
        edge = LearnableEdge(options=["A", "B"], feature_dim=4)
        
        # Step 1
        edge({"value": "s1", "trace_id": "t1"})
        # Step 2
        edge({"value": "s2", "trace_id": "t1"})
        
        self.assertIn("t1", edge.active_traces)
        self.assertEqual(len(edge.active_traces["t1"]), 2)
        
        # Complete Trace with Decay
        edge.complete_trace("t1", final_reward=1.0, decay=0.5)
        
        # Verify active trace cleared
        self.assertNotIn("t1", edge.active_traces)
        
        # Verify memory: Should have 2 entries
        mem = edge.memory.get_all()
        self.assertEqual(len(mem["actions"]), 2)
        
        # Last step (s2) gets 1.0, First step (s1) gets 0.5
        # Note: memory stores in order of addition.
        # Logic was: reversed(decisions). So s2 added first, s1 added second?
        # Let's check impl: 
        # for (ctx, act) in reversed(decisions): ...
        # decisions = [s1, s2]
        # reversed = [s2, s1]
        # First iteration: s2 -> reward 1.0 -> add to memory (idx 0)
        # Second iteration: s1 -> reward 0.5 -> add to memory (idx 1)
        
        self.assertAlmostEqual(mem["rewards"][0], 1.0)
        self.assertAlmostEqual(mem["rewards"][1], 0.5)

if __name__ == "__main__":
    unittest.main()
