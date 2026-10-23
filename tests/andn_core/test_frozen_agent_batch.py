"""Offline process-level check: public-only workers and cached resume without a cost ledger."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from PIL import Image


class FrozenBatchTest(unittest.TestCase):
    def test_parallel_workers_and_resume_use_public_inputs_without_ledger(self):
        for dataset in ('AVDN','CityNav'):
            with self.subTest(dataset=dataset):
                self.check_dataset(dataset)

    def check_dataset(self, dataset):
        project = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            source = base / 'source'
            for path in (project / dataset).rglob('*.py'):
                target = source / dataset / path.relative_to(project / dataset)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(path.read_bytes())
            # Only the external model transport is stubbed; real Agent/geometry/journals run.
            transport = source / dataset / 'tools/model.py'
            with transport.open('a', encoding='utf-8') as handle:
                handle.write('\ndef call_model(*args, **kwargs):\n'
                    '    return {"status":200,"text":\'{"action":"give_up","reason":"offline test"}\','
                    '"usage":{"input_tokens":10,"output_tokens":5}}\n')
            image = base / 'map.png'
            Image.new('RGB', (32, 32), (100, 100, 100)).save(image)
            folder = base / 'run' / dataset.lower()
            folder.mkdir(parents=True)
            entries = []
            for index in range(4):
                entries.append(dict(episode_id=f'offline-{index}',
                    agent_input=dict(dataset=dataset, setting='visual' if dataset=='AVDN' else 'landmark_assisted', instruction_raw='Find a plaza',
                        turns=[dict(turn_id='t0', role='INS', text='Find a plaza')],
                        initial_pose=dict(xy_m=[16., 16.], heading_deg=0.), coordinate_frame='east/north meters',
                        named_landmarks=[], views=[]),
                    initialization=dict(direction=0., final_target='a plaza', auxiliary=[]),
                    environment=dict(image_path=str(image), image_size=[32, 32],
                        world_to_raster=[[1,0,0],[0,-1,32],[0,0,1]], main_width_m=10., main_height_m=10.)))
            (folder / 'episodes.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in entries), encoding='utf-8')
            (folder / 'evaluator_only.jsonl').write_text('INVALID: runtime must never read this file', encoding='utf-8')
            command = [sys.executable, str(project / 'tests/andn_core/run_frozen_agent_batch.py'),
                '--dataset', dataset, '--source', str(source), '--root', str(base/'run'),
                '--model', 'qwen3.8-flash', '--workers', '2']
            environment = dict(os.environ, DASHSCOPE_API_KEY='offline-test')
            first = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=60)
            self.assertEqual(first.returncode, 0, first.stderr + first.stdout)
            self.assertFalse((base/'ledger.json').exists())
            before = {p: p.read_bytes() for p in folder.glob('episodes/*/requests/*.json')}
            self.assertEqual(len(before), 4)
            self.assertTrue(all(json.loads(data)['status'] == 'ok' for data in before.values()))
            for entry in entries:
                result = json.loads((folder/'episodes'/entry['episode_id']/'result.json').read_text(encoding='utf-8'))
                self.assertEqual(result['termination'], 'give_up')
                self.assertIsNone(result['final'])
            second = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=60)
            self.assertEqual(second.returncode, 0, second.stderr + second.stdout)
            after = {p: p.read_bytes() for p in folder.glob('episodes/*/requests/*.json')}
            self.assertEqual(before, after)
            self.assertFalse((base/'ledger.json').exists())
            self.assertEqual(second.stdout.count('"cached": true'), 4)

            # The ordinary CLI must also run without --budget or a price entry.
            serial = base / 'serial' / dataset.lower()
            serial.mkdir(parents=True)
            (serial/'episodes.jsonl').write_bytes((folder/'episodes.jsonl').read_bytes())
            command = [sys.executable, '-m', dataset+'.run',
                       '--root', str(base/'serial'), '--model', 'unlisted-model', '--limit', '1']
            result = subprocess.run(command, cwd=source, env=environment,
                                    capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            request = json.loads((serial/'episodes/offline-0/requests/000.json').read_text(encoding='utf-8'))
            self.assertEqual(request['status'], 'ok')
            self.assertEqual(request['response']['usage']['input_tokens'], 10)
            self.assertNotIn('reservation', request)
            self.assertNotIn('budget_path', json.loads((serial/'run_manifest.json').read_text(encoding='utf-8')))


if __name__ == '__main__':
    unittest.main()
