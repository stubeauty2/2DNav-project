"""Recording must preserve requests, outputs and baseline error behavior."""
import base64
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'PSC-AVDN/script'), str(ROOT/'src')]
import numpy as np
from andn_core.tools import telemetry as audit
from andn_core.tools.model import qwen_locate_bbox_in_view
from andn_core.runner import decide_turn_from_land


class Response:
    def __init__(self, body, status=200, content_type='application/json', json_error=False):
        self.text = json.dumps(body) if not isinstance(body, str) else body
        self.value = body
        self.status_code = status
        self.headers = {'Content-Type': content_type}
        self.json_error = json_error

    def json(self):
        if self.json_error:
            raise ValueError('invalid transport JSON')
        return self.value


def reply(value):
    return Response({'choices':[{'message':{'content':json.dumps(value)}}]})


class Recording(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix='andn-step0-')
        self.recorder = audit.Recorder(self.directory.name, {'api_key':'test-secret-not-for-logs'})
        self.patch_image = np.zeros((16,16,3), dtype=np.uint8)
        self.good = {'dest_present':True,'bbox_2d':[1,2,25,30],'confidence':.8,'reason':'roof','grid_5x5':[]}

    def tearDown(self):
        self.directory.cleanup()

    def invoke(self, callback, response, enabled):
        calls = []
        def post(url, **kwargs):
            calls.append((url, deepcopy(kwargs)))
            if isinstance(response, Exception):
                raise response
            return response
        token = audit._run.set(self.recorder if enabled else None)
        try:
            with patch('requests.post', post):
                result = callback()
        finally:
            audit._run.reset(token)
        return result, calls

    def localization(self):
        return qwen_locate_bbox_in_view('white roof', self.patch_image,
            patch_s8_bgr=self.patch_image, segmentation_image=self.patch_image,
            api_key='test-secret-not-for-logs')

    def final_calls(self):
        return [json.loads(p.read_text(encoding='utf-8'))
                for p in sorted((self.recorder.root/'calls').glob('*.json'))
                if not p.name.endswith('.request.json')]

    def test_exact_payload_images_and_results_with_trace_on(self):
        before, requests_before = self.invoke(self.localization, reply(self.good), False)
        after, requests_after = self.invoke(self.localization, reply(self.good), True)
        self.assertEqual(before, after)
        self.assertEqual(requests_before, requests_after)
        call = self.final_calls()[0]
        reconstructed = deepcopy(call['request'])
        assets = {item['path']:item for item in call['images']}
        for msg in reconstructed['messages']:
            if isinstance(msg['content'],list):
                for item in msg['content']:
                    if item['type']=='image_url':
                        relative=item['image_url']['url'].removeprefix('asset:')
                        data=(self.recorder.root/relative).read_bytes()
                        self.assertEqual(hashlib.sha256(data).hexdigest(),assets[relative]['sha256'])
                        item['image_url']['url']=assets[relative]['data_url_prefix']+','+base64.b64encode(data).decode()
        self.assertEqual(reconstructed,requests_before[0][1]['json'])
        self.assertEqual([a['role'] for a in call['images']],['main','context','sam3'])
        self.assertEqual(call['structured_reply'],self.good)
        self.assertEqual(call['http']['status'],200)
        self.assertIsNone(call['issue'])
        for path in self.recorder.root.rglob('*.json'):
            self.assertNotIn('test-secret-not-for-logs',path.read_text(encoding='utf-8'))
            self.assertNotIn('Authorization',path.read_text(encoding='utf-8'))

    def test_api_transport_and_non_json_fallbacks_preserved(self):
        responses = [Response({'error':{'message':'server failed'}},status=500),
                     Response('<html>unavailable</html>',status=502,content_type='text/html'),
                     Response('{',json_error=True), RuntimeError('transport timeout')]
        for response in responses:
            with self.subTest(response=type(response).__name__):
                before, a = self.invoke(self.localization,response,False)
                after, b = self.invoke(self.localization,response,True)
                self.assertEqual(before,after)
                self.assertEqual(a,b)
                self.assertTrue(self.final_calls()[-1]['fallback'])

    def test_no_match_and_malformed_model_reply_are_distinguished(self):
        self.invoke(self.localization,reply({**self.good,'dest_present':False}),True)
        self.assertIsNone(self.final_calls()[-1]['issue'])
        malformed=Response({'choices':[{'message':{'content':'no json here'}}]})
        before,_=self.invoke(self.localization,malformed,False)
        after,_=self.invoke(self.localization,malformed,True)
        self.assertEqual(before,after)
        self.assertEqual(self.final_calls()[-1]['issue'],'empty_or_unparseable_reply')

    def test_land_request_and_result_unchanged(self):
        def land():
            return decide_turn_from_land('land by the building',self.patch_image,
                api_key='test-secret-not-for-logs',heading_deg=90)
        for response in (reply({'abs_deg':180,'reason':'south','confidence':.9}),
                         Response('not json',content_type='text/html')):
            before,a=self.invoke(land,response,False)
            after,b=self.invoke(land,response,True)
            self.assertEqual(before,after)
            self.assertEqual(a,b)
            self.assertEqual(self.final_calls()[-1]['kind'],'land')
            self.assertEqual(len(self.final_calls()[-1]['images']),1)

    def test_recording_failure_does_not_change_localization(self):
        expected,_=self.invoke(self.localization,reply(self.good),False)
        with patch.object(self.recorder,'write',side_effect=OSError('disk full')):
            actual,_=self.invoke(self.localization,reply(self.good),True)
        self.assertEqual(expected,actual)
        self.assertGreater(self.recorder.errors,0)

    def test_sam3_failure_is_reraised_and_recorded(self):
        @audit.sam3_call
        def service(patch_main_bgr,target_terms,*,image_path,artifact_dir,vision_tool_url):
            raise RuntimeError('SAM3 unavailable')
        token=audit._run.set(self.recorder)
        try:
            with self.assertRaisesRegex(RuntimeError,'SAM3 unavailable'):
                service(self.patch_image,['building'],image_path='main.jpg',artifact_dir='evidence',vision_tool_url='http://localhost')
        finally:
            audit._run.reset(token)
        events=[json.loads(line) for line in (self.recorder.root/'events.jsonl').read_text(encoding='utf-8').splitlines()]
        self.assertEqual([e['event'] for e in events],['sam3_request','sam3_error'])

    def test_sam3_client_preserves_wire_arguments_and_records_payload(self):
        from andn_core.tools.sam3 import VisionToolClient
        from andn_core.tools.grounding.client import VisionToolClient as Original
        payload={'backend':'sam3','views':[{'view_id':'main','image_path':'main.png'}],
                 'concepts':[{'concept_id':'target-1','text':'building'}], 'max_candidates_per_view':1000000}
        response=Response({'backend':'sam3','candidates':[]})
        wire=[]
        def request(*args,**kwargs):
            wire.append((args,deepcopy(kwargs)));return response
        with patch('requests.request',request):
            expected=Original('http://localhost:8765',timeout=180).candidates(payload)
            token=audit._run.set(self.recorder)
            try:
                actual=VisionToolClient('http://localhost:8765',timeout=180).candidates(payload)
            finally:
                audit._run.reset(token)
        self.assertEqual(actual,expected)
        self.assertEqual(wire[0],wire[1])
        events=[json.loads(line) for line in (self.recorder.root/'events.jsonl').read_text(encoding='utf-8').splitlines()]
        self.assertEqual(events[0]['payload'],payload)

    def test_disabled_run_does_not_create_trace(self):
        @audit.navigation_run
        def run(out_dir):
            return ['original result']
        self.assertEqual(run(out_dir=self.directory.name,trace=False),['original result'])
        self.assertEqual(len(list((Path(self.directory.name)/'trace').iterdir())),1)

    def test_source_snapshot_preserves_bytes_and_redacts_real_credentials(self):
        data=b'headers = {"Authorization": f"Bearer {api_key}"}\r\n'
        self.assertEqual(self.recorder.source_bytes(data),data)
        with_secret=b'key = "test-secret-not-for-logs"\r\n'
        self.assertNotIn(b'test-secret-not-for-logs',self.recorder.source_bytes(with_secret))

    def test_manifest_snapshot_hashes_match_sources(self):
        token=audit._run.set(self.recorder)
        try:
            audit._start_run({'results_csv':str(ROOT/'out/01_parse_only/parse/preds_out_Parse_test_unseen_full/parsing_results_full.csv'),
                              'anno_dir':str(ROOT/'datasets/sample2'),'split':'test_unseen_full'})
        finally:
            audit._run.reset(token)
        manifest=json.loads((self.recorder.root/'manifest.json').read_text(encoding='utf-8'))
        for relative,digest in manifest['source_sha256'].items():
            self.assertEqual(hashlib.sha256((self.recorder.root/'snapshot'/relative).read_bytes()).hexdigest(),digest,relative)
        self.assertEqual(self.recorder.errors,0)


if __name__=='__main__':
    unittest.main()
