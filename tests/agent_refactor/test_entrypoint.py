"""Public orchestration, initialization replay, prompt and state-machine preservation."""
import ast
import copy
import importlib
import json
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import zipfile
from PIL import Image

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
OUT=ROOT/'out/agent_refactor_20261021/test_artifacts'


def original(dataset, relative):
    with zipfile.ZipFile(ROOT/'out/agent_refactor_20261021/before_refactor.zip') as z:
        return z.read(dataset+'/'+relative).decode('utf-8-sig')


class EntryTests(unittest.TestCase):
    def setUp(self):
        OUT.mkdir(parents=True,exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(prefix='case-',dir=OUT)
        self.root=Path(self.temp.name)
        Image.new('RGB',(100,100),'gray').save(self.root/'map.png')

    def tearDown(self):self.temp.cleanup()

    def episode(self,d):
        return dict(episode_id='fixture',agent_input=dict(dataset=d,
            setting='visual' if d=='AVDN' else 'landmark_assisted',
            instruction_raw='Find the blue building',turns=[dict(turn_id='t0',role='INS',text='Find the blue building')],
            initial_pose=dict(xy_m=[50.,50.],heading_deg=0.),coordinate_frame='east/north meters',
            named_landmarks=[] if d=='AVDN' else [dict(id='lm-1',name='museum',type='Building',xy_m=[70.,50.],bbox_xy_m=None)],views=[]),
            environment=dict(image_path=str(self.root/'map.png'),image_size=[100,100],
                world_to_raster=[[1,0,0],[0,-1,100],[0,0,1]],main_width_m=40.,main_height_m=40.))

    def response(self,d):
        plan=dict(stages=[dict(move_dir='EAST',weight=1,target='building')],assumptions=[]) if d=='AVDN' else dict(landmark_id='lm-1')
        return '<plan>'+json.dumps(plan)+'</plan><answer>'+json.dumps(dict(direction=None,final_target='blue building',auxiliary=[]))+'</answer>'

    def test_analysis_then_loop_and_replay_never_calls_transport_twice(self):
        for d in ('AVDN','CityNav'):
            with self.subTest(dataset=d):
                run=importlib.import_module(d+'.run');model=importlib.import_module(d+'.tools.model')
                entry=self.episode(d);before=copy.deepcopy(entry)
                responses=[dict(status=200,text=self.response(d),usage={}),
                           dict(status=200,text=json.dumps(dict(action='give_up',reason='offline end')),usage={})]
                with patch.object(model,'call_model',side_effect=responses) as transport:
                    result=run.run_episode(entry,self.root/d,key='offline-key',url='https://offline.invalid/api')
                    self.assertEqual(transport.call_count,2)
                    self.assertEqual(result['termination'],'give_up')
                public=json.loads((self.root/d/'public_episode.json').read_text(encoding='utf-8'))
                self.assertEqual(public['initialization']['direction'],90.)
                self.assertEqual(entry,before)
                with patch.object(model,'call_model',side_effect=AssertionError('replayed request sent')):
                    replay=run.run_episode(entry,self.root/d,key='offline-key',url='https://offline.invalid/api')
                self.assertEqual(result,replay)
                with patch.object(model,'call_model',side_effect=AssertionError('mismatch sent')):
                    with self.assertRaisesRegex(model.Halt,'analysis_resume_input_mismatch'):
                        run.run_episode(entry,self.root/d,key='offline-key',url='https://changed.invalid/api')

    def test_cached_initialization_skips_analysis(self):
        for d in ('AVDN','CityNav'):
            run=importlib.import_module(d+'.run');model=importlib.import_module(d+'.tools.model')
            ep=self.episode(d);ep['initialization']=dict(direction=10.,final_target='blue building',auxiliary=[])
            with patch.object(run.analysis,'analyze',side_effect=AssertionError('cached analysis called')), patch.object(model,'call_model',return_value=dict(status=200,text='{"action":"give_up","reason":"end"}',usage={})) as transport:
                result=run.run_episode(ep,self.root/d,key='offline-key')
            self.assertEqual(transport.call_count,1);self.assertEqual(result['termination'],'give_up')

    def test_analysis_only_format_repair_and_pending_stop(self):
        for d in ('AVDN','CityNav'):
            run=importlib.import_module(d+'.run');model=importlib.import_module(d+'.tools.model')
            ep=self.episode(d)
            with patch.object(model,'call_model',side_effect=[dict(status=200,text='bad json',usage={}),dict(status=200,text=self.response(d),usage={})]) as transport:
                result=run.run_episode(ep,self.root/d,key='offline-key',analysis_only=True)
            self.assertEqual(result['status'],'analyzed');self.assertEqual(transport.call_count,2)
            self.assertFalse((self.root/d/'requests').exists())
            request=self.root/d/'analysis/000.json'
            pending=json.loads(request.read_text(encoding='utf-8'));pending['status']='pending'
            request.write_text(json.dumps(pending),encoding='utf-8')
            with patch.object(model,'call_model',side_effect=AssertionError('pending resent')):
                with self.assertRaisesRegex(model.Halt,'analysis_uncertain_request_pending'):
                    run.run_episode(ep,self.root/d,key='offline-key')

    def test_private_fields_and_path_escape_rejected_before_request(self):
        for d in ('AVDN','CityNav'):
            run=importlib.import_module(d+'.run')
            ep=self.episode(d);ep['goal_xy_m']=[1,2]
            with self.assertRaises(ValueError):run.validate_input(ep)
            ep=self.episode(d);ep['episode_id']='../escape'
            with self.assertRaises(ValueError):run.validate_input(ep)

    def test_merged_prompts_are_verbatim(self):
        for d in ('AVDN','CityNav'):
            p=importlib.import_module(d+'.prompts');a=importlib.import_module(d+'.analysis')
            namespace=dict(json=json,SCHEMA=a.SCHEMA,PLAN_SCHEMA=a.PLAN_SCHEMA)
            for relative in ('prompt.py','initialization/prompts.py','agent/prompts.py'):
                tree=ast.parse(original(d,relative))
                tree.body=[n for n in tree.body if not isinstance(n,(ast.Import,ast.ImportFrom))]
                exec(compile(tree,relative,'exec'),namespace)
            self.assertEqual(namespace['SYSTEM'],p.SYSTEM)
            self.assertEqual(namespace['REVIEW_SYSTEM'],p.REVIEW_SYSTEM)
            for mode in (('direct','staged') if d=='AVDN' else (None,)):
                self.assertEqual(namespace['system_prompt'](mode) if mode else namespace['system_prompt'](),
                                 p.initialization_prompt(mode) if mode else p.initialization_prompt())

    def test_loop_and_view_algorithms_unchanged(self):
        for d in ('AVDN','CityNav'):
            for old,new,kind,name in [('navigation.py','loop.py',ast.ClassDef,'Agent'),
                                      ('tools/views.py','tools/views.py',ast.ClassDef,'Environment'),
                                      ('agent/contracts.py','loop.py',ast.FunctionDef,'validate_action')]:
                before=next(n for n in ast.parse(original(d,old)).body if isinstance(n,kind) and n.name==name)
                after=next(n for n in ast.parse((ROOT/d/new).read_text(encoding='utf-8')).body if isinstance(n,kind) and n.name==name)
                self.assertEqual(ast.dump(before),ast.dump(after),d+' '+name)

    def test_online_imports_have_no_offline_or_removed_dependency(self):
        for d in ('AVDN','CityNav'):
            importlib.import_module(d+'.run')
            for path in (ROOT/d).rglob('*.py'):
                ast.parse(path.read_text(encoding='utf-8'))
            self.assertFalse((ROOT/d/'agent').exists())
            self.assertFalse((ROOT/d/'initialization').exists())
        probe = subprocess.run([sys.executable,'-B','-c',
            "import AVDN.run, CityNav.run, sys; assert not any(n.startswith('nav_offline.') for n in sys.modules)"],
            cwd=ROOT,capture_output=True,text=True)
        self.assertEqual(probe.returncode,0,probe.stderr)


if __name__=='__main__':unittest.main()
