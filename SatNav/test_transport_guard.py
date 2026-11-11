import tempfile
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
from AVDN.agent.io import save_json, load_json
from SatNav.run import check_transport_result
from AVDN.initialization.client import call_model


class TransportGuardTest(unittest.TestCase):
    def test_extended_upload_timeout_is_opt_in(self):
        response = MagicMock()
        response.status_code = 200
        response.headers = {'Content-Type': 'application/json'}
        response.json.return_value = {'choices': [{'message': {'content': 'ok'}}]}
        response.__enter__.return_value = response
        with patch('AVDN.initialization.client.requests.post', return_value=response) as post:
            args = ('https://example.invalid/chat', 'model', 'test-key',
                    [{'role': 'user', 'content': 'test'}])
            call_model(*args, timeout=90)
            self.assertEqual(post.call_args.kwargs['timeout'], (20, 90))
            call_model(*args, timeout=180, connect_timeout=90)
            self.assertEqual(post.call_args.kwargs['timeout'], (90, 180))

    def test_api_failure_stops_without_overwriting_episode(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            output = root / 'episodes' / 'example'
            (output / 'requests').mkdir(parents=True)
            result = dict(termination='interface_error', completed=True)
            save_json(output / 'result.json', result)
            save_json(output / 'requests' / '000.json', dict(status='api_error', error='ConnectionError'))
            with self.assertRaises(RuntimeError):
                check_transport_result(result, output)
            self.assertEqual(load_json(output / 'result.json'), result)
            self.assertEqual(load_json(root / 'transport_failure.json')['episode_id'], 'example')

    def test_policy_failures_do_not_abort_remaining_samples(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'episodes' / 'example'
            (output / 'requests').mkdir(parents=True)
            save_json(output / 'requests' / '000.json', dict(status='ok'))
            for termination in ('interface_error', 'step_limit', 'confirmed_final'):
                check_transport_result(dict(termination=termination), output)
            self.assertFalse((Path(folder) / 'transport_failure.json').exists())
