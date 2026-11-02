import unittest
import os
import shutil
import tempfile
from dust_impact.common.io import ensure_dir


class TestCommonIO(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.filepath = os.path.join(self.test_dir, "test_output.npz")

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_ensure_dir(self):
        nested_file = os.path.join(self.test_dir, "nested", "dir", "file.txt")
        ensure_dir(nested_file)
        self.assertTrue(os.path.exists(os.path.dirname(nested_file)))


if __name__ == "__main__":
    unittest.main()
