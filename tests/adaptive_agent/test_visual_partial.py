"""Partial visual answers retain evidence without fabricating ROI assessments."""
import copy
import importlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from fixtures import episode


class VisualPartialTests(unittest.TestCase):
    def test_partial_raster_region_reaches_visual_tool_but_not_final_locate(self):
        for ds in ('AVDN','CityNav'):
            with self.subTest(dataset=ds),tempfile.TemporaryDirectory() as td:
                root=Path(td);ep=episode(root,ds)
                ep['agent_input']['initial_pose']['xy_m']=[5.,50.]
                loop=importlib.import_module(ds+'.loop');model_module=importlib.import_module(ds+'.tools.model')
                views=importlib.import_module(ds+'.tools.views');grounding=importlib.import_module(ds+'.tools.grounding_skill')
                visual=importlib.import_module(ds+'.tools.visual_evidence')
                env=views.Environment(ep['environment'],root/'views')
                model=model_module.Model(root/'requests','qwen3.5-flash','test',enable_thinking=False,
                    tool_protocol='structured_v2',visual_model='qwen3.8-flash')
                questions=[];decisions=[]
                box=[200,300,700,700]
                def transport(url,name,key,messages,**kw):
                    ctx=json.loads(messages[1]['content'][0]['text'])
                    if ctx['mode']=='visual_evidence':
                        questions.append(ctx)
                        self.assertEqual(name,'qwen3.8-flash')
                        self.assertEqual(ctx['regions'],[dict(id='edge',bbox=box)])
                        coverage=ctx['region_visibility'][0]
                        self.assertTrue(coverage['touches_padding'])
                        self.assertGreater(coverage['valid_area_fraction'],0)
                        self.assertLess(coverage['valid_area_fraction'],1)
                        answer=dict(observations=[dict(region_id='edge',support=['Visible portion only'],
                            contradictions=[],unobservable=['Padded portion'])],comparison='Partial region',uncertainty=[],suggested_check='none')
                    else:
                        decisions.append(ctx)
                        if len(decisions)==1:
                            answer=dict(action='tool',tool='visual_question',view_id=ctx['views'][0]['view_id'],
                                question='What is visible in this region?',regions=[dict(id='edge',bbox=box)],reason='Inspect edge')
                        else:answer=dict(action='give_up',reason='Insufficient identity evidence')
                    return dict(status=200,text=json.dumps(answer),usage={})
                try:
                    agent=loop.Agent(ep,env,model,grounding.SAM3(),root,grounders={},grounding_mode='adaptive',max_rounds=1)
                    with patch.object(model_module,'call_model',side_effect=transport):result=agent.run()
                    self.assertEqual(result['termination'],'give_up');self.assertEqual(len(questions),1)
                    outputs=[e['output'] for e in agent.events if e['kind']=='tool']
                    self.assertEqual(len(outputs),1);self.assertEqual(outputs[0]['status'],'ok')
                    self.assertEqual(outputs[0]['region_visibility'],questions[0]['region_visibility'])
                    main=env.views['r00-main']
                    with self.assertRaises(ValueError):env.locate(main,box)
                    with patch.object(model,'request',side_effect=AssertionError('No paid request for invalid ROI')):
                        for bad in ([0,300,100,700],[-1,300,700,700]):
                            with self.assertRaises(ValueError):
                                visual.VisualEvidence(model).inspect(main,'Inspect region',[dict(id='bad',bbox=bad)],root/'bad.json')
                finally:env.close()

    def test_missing_and_unrequested_regions_are_explicit_in_actual_tool_result(self):
        for ds in ('AVDN','CityNav'):
            module=importlib.import_module(ds+'.tools.visual_evidence')
            regions=[dict(id='A',bbox=[10,10,20,20]),dict(id='B',bbox=[30,30,40,40])]
            answer=dict(observations=[dict(region_id='A',support=['Visible red roof']),
                dict(region_id='unbound',support=['Another object'],contradictions=[],unobservable=[])],
                comparison='Different visible roofs',uncertainty=['Lighting ambiguous'],suggested_check='Inspect B',proposals=[])
            original=copy.deepcopy(answer)
            class Model:
                cursor=1
                def request(self,context,views):
                    return dict(status='ok',model='qwen3.8-flash',response=dict(text=json.dumps(answer)))
            with tempfile.TemporaryDirectory() as td:
                result=module.VisualEvidence(Model()).inspect(dict(view_id='main'),'Compare visible roofs',regions,Path(td)/'tool.json')
            self.assertEqual(answer,original)
            self.assertEqual(result['status'],'partial')
            self.assertEqual(result['missing_regions'],['B'])
            self.assertEqual(result['unrequested_region_ids'],['unbound'])
            a,b=result['observations']
            self.assertEqual(a['support'],['Visible red roof'])
            self.assertTrue(a['unobservable'])
            self.assertEqual(b['support'],[])
            self.assertEqual(b['contradictions'],[])
            self.assertIn('did not assess',b['unobservable'][0])
            self.assertNotIn('Another object',str(result['observations']))

    def test_region_evidence_survives_missing_summary_without_inventing_comparison(self):
        for ds in ('AVDN','CityNav'):
            module=importlib.import_module(ds+'.tools.visual_evidence')
            regions=[dict(id='A',bbox=[10,10,20,20])]
            observation=dict(region_id='A',support=['Dark roof'],contradictions=['No reflective panel visible'],unobservable=[])
            answer=dict(observations=[observation]);original=copy.deepcopy(answer)
            class Model:
                cursor=1
                def request(self,context,views):
                    return dict(status='ok',model='qwen3.8-flash',response=dict(text=json.dumps(answer)))
            with tempfile.TemporaryDirectory() as td:
                result=module.VisualEvidence(Model()).inspect(dict(view_id='main'),'Compare roof attributes',regions,Path(td)/'tool.json')
            self.assertEqual(answer,original)
            self.assertEqual(result['status'],'partial')
            self.assertEqual(result['observations'],[observation])
            self.assertEqual(result['comparison'],'')
            self.assertEqual(result['suggested_check'],'')
            self.assertIn('Model omitted summary fields',result['uncertainty'][0])
            self.assertTrue(result['normalization_notes'])
            partial=dict(answer,comparison='Existing comparison',uncertainty=['Existing uncertainty'])
            normalized,notes,_,_=module.recover_answer(partial,regions,'main')
            self.assertEqual(normalized['comparison'],'Existing comparison')
            self.assertEqual(normalized['uncertainty'][0],'Existing uncertainty')
            self.assertIn('suggested_check',notes[0])

    def test_missing_summary_does_not_hide_invalid_or_unbound_evidence(self):
        for ds in ('AVDN','CityNav'):
            module=importlib.import_module(ds+'.tools.visual_evidence')
            regions=[dict(id='A',bbox=[10,10,20,20])]
            obs=dict(region_id='A',support=['Visible roof'],contradictions=[],unobservable=[])
            for answer in (dict(observations=[]),dict(observations=[dict(region_id='A')]),
                    dict(observations=[dict(obs,region_id='B')]),
                    dict(observations=[obs],uncertainty='wrong type'),
                    dict(observations=[obs],comparison=None),
                    dict(observations=[obs,obs]),dict(observations=[obs],unexpected='value'),
                    dict(observations=[obs],source_view_id='other'),
                    dict(observations=[obs],proposals=[dict(id='p',bbox=[0,0,1001,100],category='building',evidence='roof',uncertainty=[])])):
                with self.assertRaises(ValueError):module.recover_answer(answer,regions,'main')

    def test_structural_recovery_preserves_summary_and_checks_view_and_boxes(self):
        for ds in ('AVDN','CityNav'):
            m=importlib.import_module(ds+'.tools.visual_evidence')
            summary=dict(comparison='Visible differences',uncertainty=[],suggested_check='none',proposals=[])
            valid=dict(observations=[],**summary)
            normalized,notes,missing,unrequested=m.recover_answer(valid,[],'main')
            self.assertEqual(normalized,valid);self.assertEqual(notes,[])
            nested=dict(observations=[summary]);original=copy.deepcopy(nested)
            normalized,notes,_,_=m.recover_answer(nested,[],'main')
            self.assertEqual(normalized,valid);self.assertTrue(notes);self.assertEqual(nested,original)
            normalized,notes,_,_=m.recover_answer(dict(valid,source_view_id='main'),[],'main')
            self.assertEqual(normalized,valid);self.assertTrue(notes)
            bad_proposal=dict(id='p',bbox=[0,0,1001,100],category='building',evidence='roof',uncertainty=[])
            for bad in (dict(valid,source_view_id='other'),dict(valid,proposals=[bad_proposal]),
                        dict(observations=[summary,summary]),dict(valid,unexpected='x'),
                        dict(valid,observations=[dict(region_id='A'),dict(region_id='A')])):
                with self.assertRaises(ValueError):m.recover_answer(bad,[],'main')


if __name__=='__main__':unittest.main()
