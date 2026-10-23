import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec=importlib.util.spec_from_file_location('guard_dispatch',Path(__file__).with_name('guard_dispatch.py'))
guard=importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)


class GuardTest(unittest.TestCase):
    def test_only_terminal_interface_error_with_api_error_triggers(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder); (p/'requests').mkdir()
            self.assertFalse(guard.api_exhausted(p))
            (p/'result.json').write_text(json.dumps({'termination':'interface_error'}))
            (p/'requests/000.json').write_text(json.dumps({'status':'ok'}))
            self.assertFalse(guard.api_exhausted(p))
            (p/'requests/001.json').write_text(json.dumps({'status':'api_error'}))
            self.assertTrue(guard.api_exhausted(p))
            (p/'result.json').write_text(json.dumps({'termination':'step_limit'}))
            self.assertFalse(guard.api_exhausted(p))


if __name__=='__main__':
    unittest.main()
