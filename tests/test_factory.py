import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Ensure src is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from adaptivegraph import LearnableEdge
from adaptivegraph.memory import InMemoryExperienceStore


class TestLearnableEdgeFactory(unittest.TestCase):
    def test_create_default(self):
        """Test default creation falls back to standard Init if no special params."""
        # This might fail if defaults in create() require libs.
        # But wait, create() defaults are "sentence-transformers" and "faiss".
        # So it will try to import them.
        pass

    @patch.dict(
        sys.modules, {"sentence_transformers": MagicMock(), "faiss": MagicMock()}
    )
    def test_create_with_mocks(self):
        """Test that create() correctly initializes components when libs are present."""

        # We need to ensure the modules are mocked BEFORE import in the factory
        # But LearnableEdge imports them inside the method.

        edge = LearnableEdge.create(
            options=["A", "B"],
            embedding="sentence-transformers",
            memory="faiss",
            feature_dim=16,
        )

        self.assertEqual(edge.encoder.output_dim, 16)
        # Check encoder has an embedding_fn
        self.assertIsNotNone(edge.encoder.embedding_fn)
        # Check memory is Faiss (mocked)
        self.assertTrue(edge.memory.__class__.__name__, "FaissExperienceStore")

    def test_create_explicit_memory(self):
        """Test using simple memory."""
        # We might need to mock sentence-transformers if it's the default embedding
        with patch.dict(sys.modules, {"sentence_transformers": MagicMock()}):
            edge = LearnableEdge.create(
                options=["A", "B"], embedding="sentence-transformers", memory="memory"
            )
            self.assertIsInstance(edge.memory, InMemoryExperienceStore)


if __name__ == "__main__":
    unittest.main()
