import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from . import resume_repairs
from .cli import save_json, write_jsonl
from .prompts import system_prompt
from .test_initialization import agent, answer


class DeferredRepairTests(unittest.TestCase):
    def fixture(self, folder, limit=10):
        root = Path(folder); source = root/'source'; source.mkdir()
        inputs = root/'inputs.jsonl'
        write_jsonl(inputs, [{'episode_id':'test', 'agent_input':agent()}])
        original = {'episode_id':'test', 'dataset':'CityNav', 'setting':'visual',
                    'model':'qwen3.5-flash', 'status':'error', 'error':'BudgetExceeded',
                    'usage':{'input_tokens':300, 'output_tokens':30},
                    'repair_attempts':1, 'elapsed_seconds':2.0}
        write_jsonl(source/'predictions.jsonl', [original])
        raw = '<answer>{"direction":'
        save_json(source/'calls/test.answer.json', {'content':raw})
        save_json(source/'calls/test.repair.json', {'feedback':'Return valid JSON.'})
        save_json(source/'run_manifest.json', {
            'status':'completed', 'model':'qwen3.5-flash', 'endpoint':'https://example.invalid/api',
            'protocol':'native', 'agent_inputs_sha256':resume_repairs.digest(inputs),
            'system_prompt_sha256':hashlib.sha256(system_prompt().encode()).hexdigest(),
            'runtime_source_sha256':{name:resume_repairs.digest(Path(resume_repairs.__file__).with_name(name))
                                     for name in ['cli.py','contracts.py','client.py','budget.py']},
            'inference_parameters':{'enable_thinking':False,'thinking_budget':50,'max_tokens':2048}})
        args = SimpleNamespace(source_run=source, inputs=inputs, out=root/'repairs',
                               budget_ledger=root/'budget.json', max_cost_cny=limit,
                               timeout=60, execute=False)
        return args, original, raw

    def test_dry_run_validates_without_key_or_budget_and_rejects_changed_inputs(self):
        with tempfile.TemporaryDirectory() as folder:
            args, _, _ = self.fixture(folder)
            with patch.object(resume_repairs, 'call_model') as api:
                result = resume_repairs.run(args)
                self.assertEqual(result['episodes'],1)
                self.assertFalse(args.budget_ledger.exists())
                args.inputs.write_text('changed',encoding='utf-8')
                with self.assertRaisesRegex(ValueError,'unchanged original'):
                    resume_repairs.run(args)
                api.assert_not_called()

    def test_only_saved_repair_is_sent_and_rerun_does_not_repeat(self):
        with tempfile.TemporaryDirectory() as folder:
            args, original, raw = self.fixture(folder); args.execute=True
            response = {'status':200,'text':'<answer>'+json.dumps(answer())+'</answer>',
                        'usage':{'input_tokens':400,'output_tokens':40}}
            with patch.dict('os.environ',{'DASHSCOPE_API_KEY':'unit-test-placeholder'}), \
                    patch.object(resume_repairs,'call_model',return_value=response) as api:
                resume_repairs.run(args); resume_repairs.run(args)
                api.assert_called_once()
                messages = api.call_args.args[3]
                self.assertEqual(messages[-2],{'role':'assistant','content':raw})
                self.assertEqual(messages[-1],{'role':'user','content':'Return valid JSON.'})
            result = resume_repairs.read(args.out/'records/test.json')
            self.assertEqual(result['usage'], original['usage'])
            self.assertEqual(result['repair_usage'],response['usage'])
            self.assertEqual(result['status'],'ok')
            self.assertEqual(resume_repairs.read(args.budget_ledger)['settled_requests'],1)
            self.assertEqual(json.loads((args.source_run/'predictions.jsonl').read_text()),original)

    def test_insufficient_budget_prevents_network(self):
        with tempfile.TemporaryDirectory() as folder:
            args, _, _ = self.fixture(folder,limit=0.000001); args.execute=True
            with patch.dict('os.environ',{'DASHSCOPE_API_KEY':'unit-test-placeholder'}), \
                    patch.object(resume_repairs,'call_model') as api:
                resume_repairs.run(args)
                api.assert_not_called()
            self.assertEqual(resume_repairs.read(args.out/'records/test.json')['error'],'BudgetExceeded')
            self.assertEqual(resume_repairs.read(args.budget_ledger)['charged_cny'],0)


if __name__ == '__main__':
    unittest.main()
