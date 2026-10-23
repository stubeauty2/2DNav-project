"""Replay historical public observations/responses through the default MLLM path.

This keeps the old mechanical AST gate intact. RGB rendering is replayed from
recorded views, with requested crop geometry checked; it is not a rasterizer test.
No model/service calls or evaluator labels are read.
"""
from collections import Counter
import copy
import hashlib
import importlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

WORK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORK))
HISTORY = WORK/'out/refactor_online_compare_20261021/after'
OUT = WORK/'out/jev_study_20261022/default_mllm_replay'
ADDITIONAL = {'review_backend','jev_requests','jev_calls','jev_error_count'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    if isinstance(value,dict):
        return {k:({'file_sha256':sha(v)} if v and k in
                   ('image_path','valid_mask_path','unmarked_image_path','mask_path') else canonical(v))
                for k,v in value.items()}
    if isinstance(value,list):return [canonical(x) for x in value]
    return value


def replay(test, dataset, folder, output):
    module=importlib.import_module(dataset+'.loop')
    models=importlib.import_module(dataset+'.tools.model')
    views=importlib.import_module(dataset+'.tools.views')
    source=read(folder/'public_episode.json')
    expected_result=read(folder/'result.json')
    expected_trace=read(folder/'trace.json')
    recorded_views={v['view_id']:v for v in expected_trace['views']}
    records=[read(p) for p in sorted((folder/'requests').glob('*.json'))]
    capabilities=read(folder/'tool_capabilities.json')
    delivered=[]

    class RecordedEnvironment(views.Environment):
        def __init__(self):
            self.spec=source['environment'];self.output=output/'views'
            self.output.mkdir(parents=True,exist_ok=True)
            self.image=SimpleNamespace(size=tuple(self.spec['image_size']))
            self.views={}

        def close(self):pass

        def render(self,corners,view_id,kind,parent_id=None):
            test.assertIn(view_id,recorded_views,'Requested a previously unseen observation')
            old=recorded_views[view_id]
            test.assertTrue(np.allclose(corners,old['world_corners_xy_m'],rtol=0,atol=1e-6),
                            'Default path requested different crop geometry: '+view_id)
            test.assertEqual(kind,old['source']);test.assertEqual(parent_id,old.get('parent_id'))
            view=copy.deepcopy(old)
            if view.get('unmarked_image_path'):
                view['image_path']=view.pop('unmarked_image_path')
                view.pop('proposal_bbox',None);view.pop('annotation',None)
            self.views[view_id]=view
            return view

    class RecordedGrounder:
        def __init__(self,name):self.name=name
        def available(self):return capabilities[self.name]=='available'
        def candidates(self,view,query,target):
            result=read(folder/'tools'/Path(target).name)
            test.assertEqual(result['backend'],self.name)
            test.assertEqual(result['input']['concepts'][0]['text'],query)
            prior_view=result['input']['views'][0]
            test.assertEqual(prior_view['view_id'],view['view_id'])
            test.assertEqual(sha(prior_view['image_path']),sha(view['image_path']))
            return result

    class CheckedModel(models.Model):
        def request(self,context,images):
            test.assertLess(self.cursor,len(records),'Default path requested an extra model decision')
            old=records[self.cursor]
            def response(*args,**kwargs):
                # The exact historical provider response is replayed, not inferred.
                if 'response' in old:return copy.deepcopy(old['response'])
                raise type(old.get('error','RecordedError'),(Exception,),{})('recorded failure')
            with patch.object(models,'call_model',side_effect=response):
                actual=super().request(context,images)
            for key in ('status','system_prompt','context','enable_thinking','thinking_budget',
                        'temperature','response_format','request_format','max_tokens'):
                test.assertEqual(canonical(actual.get(key)),canonical(old.get(key)),
                                 f'{dataset}/{folder.name}/{self.cursor-1}: {key}')
            test.assertEqual([(a['view_id'],a['sha256']) for a in actual['assets']],
                             [(a['view_id'],a['sha256']) for a in old['assets']])
            delivered.extend(a['view_id'] for a in actual['assets'])
            return actual

    env=RecordedEnvironment()
    first=records[0]
    model=CheckedModel(output/'requests',first['model'],'unused',url='https://offline.invalid',
                       enable_thinking=first['enable_thinking'],thinking_budget=first['thinking_budget'])
    grounders={name:RecordedGrounder(name) for name in capabilities}
    agent=module.Agent(source,env,model,grounders['sam3'],output,
        grounders=grounders,observation_grounder=expected_result['observation_grounder'],
        observation_query_mode=expected_result['observation_query_mode'])
    # No review_backend argument: exercise the public default.
    result=agent.run()
    test.assertEqual(model.cursor,len(records))
    test.assertEqual({k:v for k,v in result.items() if k not in ADDITIONAL},expected_result)
    actual_trace=read(output/'trace.json')
    test.assertEqual(canonical({k:v for k,v in actual_trace.items() if k not in ADDITIONAL}),
                     canonical(expected_trace))
    test.assertEqual(result['review_backend'],'mllm')
    test.assertEqual(result['jev_requests'],0);test.assertEqual(result['jev_calls'],[])
    return dict(dataset=dataset,episode_id=folder.name,requests=len(records),
                review_requests=sum(r['context']['mode']=='review' for r in records),
                image_deliveries=len(delivered),termination=result['termination'],
                event_counts=dict(Counter(e['kind'] for e in actual_trace['events'])),
                source_result_sha256=sha(folder/'result.json'),source_trace_sha256=sha(folder/'trace.json'))


class DefaultHistoricalReplayTests(unittest.TestCase):
    def test_all_twenty_historical_trajectories(self):
        OUT.mkdir(parents=True,exist_ok=True)
        rows=[]
        with tempfile.TemporaryDirectory(prefix='replay-',dir=OUT) as tmp:
            for dataset in ('AVDN','CityNav'):
                episodes=sorted((HISTORY/dataset.lower()/'episodes').iterdir())
                self.assertEqual(len(episodes),10)
                for folder in episodes:
                    with self.subTest(dataset=dataset,episode=folder.name):
                        rows.append(replay(self,dataset,folder,Path(tmp)/dataset/folder.name))
        report=dict(status='passed' if len(rows)==20 else 'failed',episodes=len(rows),
                    model_requests=sum(r['requests'] for r in rows),
                    review_requests=sum(r['review_requests'] for r in rows),rows=rows,
                    scope='Default MLLM path: exact historical prompt, context, image hashes, tool query, crop geometry, complete trace and original result fields',
                    historical_ast_gate='tests/agent_refactor/test_entrypoint.py::test_loop_and_view_algorithms_unchanged remains intact. Its whole-Agent AST equality requirement is superseded for this feature extension; the new Agent necessarily contains optional Jev branches.',
                    limitations=['Historical responses and rendered observations are replayed; no new model inference or raster decoding.',
                                 'Additional review/Jev telemetry fields are excluded from old result/trace equality.',
                                 'This verifies implementation compatibility on 20 historical trajectories, not success-rate improvement or new-backend visual quality.'])
        (OUT/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        self.assertEqual(len(rows),20)


if __name__=='__main__':unittest.main()
