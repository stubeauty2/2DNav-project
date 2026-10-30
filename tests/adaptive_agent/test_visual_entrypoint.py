"""Check evidence-model defaults without changing initialization or decision roles."""
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path[:0] = [str(Path(__file__).resolve().parents[2]), str(Path(__file__).resolve().parent)]
from fixtures import episode


class VisualEntrypointTests(unittest.TestCase):
    def test_python_default_disable_and_legacy_modes_keep_role_separation(self):
        for ds in ('AVDN', 'CityNav'):
            module = importlib.import_module(ds + '.run')
            for options, expected in (({}, 'qwen3.8-flash'),
                                      ({'visual_model': None}, None),
                                      ({'grounding_mode': 'direct'}, None),
                                      ({'grounding_mode': 'skill'}, None)):
                with self.subTest(dataset=ds, options=options), tempfile.TemporaryDirectory() as td:
                    root = Path(td)
                    ep = episode(root, ds)
                    labels = ep.pop('initialization')
                    with patch.object(module.analysis, 'analyze', return_value=labels) as analyze, \
                         patch.object(module, 'Model') as model, patch.object(module, 'Agent') as agent:
                        agent.return_value.run.return_value = {'completed': True}
                        module.run_episode(ep, root/'run', key='test', **options)
                    self.assertEqual(analyze.call_args.kwargs['model'], 'qwen3.5-flash')
                    self.assertNotIn('prompt_mode', analyze.call_args.kwargs)
                    self.assertEqual(model.call_args.args[1], 'qwen3.5-flash')
                    self.assertEqual(model.call_args.kwargs['visual_model'], expected)
                    self.assertEqual(agent.call_args.args[0]['initialization'], labels)

    def test_cli_default_override_and_disable_reach_runtime(self):
        for ds in ('AVDN', 'CityNav'):
            module = importlib.import_module(ds + '.run')
            for flags, expected in (([], 'qwen3.8-flash'),
                                    (['--visual-model', 'test-vlm'], 'test-vlm'),
                                    (['--no-visual-model'], None)):
                with self.subTest(dataset=ds, flags=flags), tempfile.TemporaryDirectory() as td:
                    root = Path(td)
                    inputs = root/'inputs.jsonl'
                    inputs.write_text(json.dumps(episode(root, ds))+'\n', encoding='utf-8')
                    with patch.dict(os.environ, {'DASHSCOPE_API_KEY': 'test'}), \
                         patch.object(module, 'run_episode', return_value={'completed': True}) as run, \
                         patch('builtins.print'):
                        module.main(['--inputs', str(inputs), '--root', str(root/'output')] + flags)
                    self.assertEqual(run.call_args.kwargs['model'], 'qwen3.5-flash')
                    self.assertEqual(run.call_args.kwargs['visual_model'], expected)


if __name__ == '__main__':
    unittest.main()
