import unittest
import sys
import os
import shutil
import numpy as np

# Ensure src is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

# Mocking packages if needed, but since we installed [all], they should be present.
try:
    from adaptivegraph.memory import FaissExperienceStore
except ImportError:
    print("Skipping Persistence Test: faiss-cpu not installed")
    sys.exit(0)

class TestFaissPersistence(unittest.TestCase):
    def setUp(self):
        self.test_dir = "./test_persist_data"
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        os.makedirs(self.test_dir)
        self.persist_path = os.path.join(self.test_dir, "test_store")

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_save_and_load(self):
        # 1. Create and add data
        store1 = FaissExperienceStore(dim=4, persist_path=self.persist_path)
        vec1 = np.array([1.0, 0.0, 0.0, 0.0])
        store1.add(vec1, action=1, reward=0.5, metadata={"k":"v"})
        
        # files should exist immediately (auto-save)
        self.assertTrue(os.path.exists(self.persist_path + ".index"))
        self.assertTrue(os.path.exists(self.persist_path + ".pkl"))

        # 2. Load into new store
        store2 = FaissExperienceStore(dim=4, persist_path=self.persist_path)
        data = store2.get_all()
        
        self.assertEqual(len(data["actions"]), 1)
        self.assertEqual(data["actions"][0], 1)
        self.assertEqual(data["metadata"][0]["k"], "v")
        
    def test_clear_deletes_Files(self):
        store = FaissExperienceStore(dim=4, persist_path=self.persist_path)
        store.add(np.array([1,1,1,1]), 0, 0)
        self.assertTrue(os.path.exists(self.persist_path + ".index"))
        
        store.clear()
        self.assertFalse(os.path.exists(self.persist_path + ".index"))
        self.assertFalse(os.path.exists(self.persist_path + ".pkl"))

if __name__ == "__main__":
    unittest.main()
