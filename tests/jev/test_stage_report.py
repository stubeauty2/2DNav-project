import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('stage_report', Path(__file__).with_name('stage_report.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
geometry = module.module_at('test_geometry', Path(__file__).resolve().parents[2]/'AVDN/tools/geometry.py')


class StageReportTest(unittest.TestCase):
    def test_action_validation_messages_do_not_include_api_or_format_failures(self):
        events=[dict(kind='invalid_action',error=message) for message in [
            'Requested position [1, 2] is outside map. Valid east/north map corners: []',
            'Image point is in the invalid raster area',
            'Optional tool limit reached; explore, locate or give_up',
            'Unknown public landmark_id', 'Unknown public landmark_id']]
        events.extend([dict(kind='invalid_format',error='outside map'),
                       dict(kind='model_error',error='API error')])
        counts=module.invalid_action_reasons(events)
        self.assertEqual(counts,dict(outside_map=1,invalid_raster=1,tool_budget_exhausted=1,other=2))
        self.assertEqual(sum(counts.values()),5)

    def setUp(self):
        identity = [[1,0,0],[0,1,0],[0,0,1]]
        self.public = dict(environment=dict(image_size=[100,100],world_to_raster=identity))
        self.truth = dict(goal_xy_m=[50,50])
        self.view = dict(view_id='main',world_corners_xy_m=[[0,0],[100,0],[100,100],[0,100]],
                         pixel_to_world=identity,image_size=[101,101])
        self.trace = dict(views=[self.view],delivered_view_ids=['main'],events=[])
        self.good = dict(view_id='main',center_xy_m=[50,50],bbox_pixels=[40,40,60,60],bbox=[400,400,600,600])
        self.bad = dict(view_id='main',center_xy_m=[90,90],bbox_pixels=[80,80,100,100])

    def run_case(self, arm='mllm'):
        return module.diagnose('CityNav',self.public,self.truth,self.trace,dict(arm=arm),geometry)

    def test_auxiliary_candidates_not_scored_as_final_targets(self):
        self.trace['events'] = [dict(kind='candidate',round=0,target='auxiliary',candidate=self.good),
            dict(kind='review_result',round=0,candidate=self.good,answer={'verdict':'uncertain'}),
            dict(kind='candidate',round=1,target='final',candidate=self.bad)]
        result = self.run_case()
        self.assertEqual(result['final_proposals'],1)
        self.assertEqual(result['qualified_final_proposals'],0)
        self.assertEqual(result['final_reviews'],0)

    def test_undelivered_view_cannot_supply_coverage(self):
        small = copy.deepcopy(self.view)
        small.update(view_id='small',world_corners_xy_m=[[0,0],[10,0],[10,10],[0,10]])
        self.trace.update(views=[self.view,small],delivered_view_ids=['small'])
        self.assertFalse(self.run_case()['geometric_center_covered'])

    def test_service_error_not_counted_as_model_abstention(self):
        for i,status in enumerate(('ok','error')):
            self.trace['events'].extend([dict(kind='candidate',round=i,target='final',candidate=self.good),
                dict(kind='jev_response',round=i,status=status),
                dict(kind='review_result',round=i,candidate=self.good,answer={'verdict':'uncertain'})])
        result = self.run_case('jev')
        self.assertEqual(result['qualified_reviews_not_confirmed'],1)
        self.assertEqual(result['qualified_reviews_service_error'],1)
        self.assertEqual(result['qualified_review_verdicts'],{'uncertain':1})

    def test_refinement_damage_and_tool_proposal_are_separate(self):
        self.trace['events'] = [dict(kind='candidate',round=0,target='final',candidate=self.good),
            dict(kind='review_result',round=0,candidate=self.good,refined=self.bad,answer={'verdict':'confirm'}),
            dict(kind='tool',round=0,output=dict(status='ok',candidates=[dict(view_id='main',bbox=[400,400,600,600])]))]
        result = self.run_case()
        self.assertEqual(result['refinement_damaged'],1)
        self.assertEqual(result['qualified_tool_proposals'],1)
        self.assertEqual(result['qualified_reviews_not_confirmed'],0)

    def start_review(self, target='final'):
        return [dict(kind='candidate',round=0,target=target,candidate=self.good),
                dict(kind='review_created',round=0,candidate=self.good,view={'view_id':'r00-review'})]

    def test_jev_halt_without_result_is_error_not_abstention(self):
        self.trace['events'] = self.start_review()+[dict(kind='jev_halt',round=0,
            reason='jev_service_identity_changed',view_ids=['main','r00-review'])]
        self.trace['termination'] = 'jev_service_identity_changed'
        result = self.run_case('jev')
        self.assertEqual(result['qualified_reviews_service_error'],1)
        self.assertEqual(result['qualified_reviews_not_confirmed'],0)
        self.assertEqual(result['qualified_review_verdicts'],{})
        self.assertEqual((result['final_reviews'],result['final_review_results'],result['interrupted_final_reviews']),(1,0,1))
        self.assertIsNone(result['reviews'][0]['verdict'])
        self.assertEqual(result['reviews'][0]['reviewer_error'],'jev_service_identity_changed')

    def test_mllm_review_interface_exhaustion_is_one_failed_review(self):
        for arm in ('mllm','mllm_fixed'):
            with self.subTest(arm=arm):
                self.trace['events'] = self.start_review()+[dict(kind='model_response',round=0,
                    status='api_error',view_ids=['main','r00-review']) for _ in range(3)]
                self.trace['termination'] = 'interface_error'
                result = self.run_case(arm)
                self.assertEqual(result['qualified_reviews_service_error'],1)
                self.assertEqual(result['qualified_reviews_not_confirmed'],0)
                self.assertEqual(result['qualified_review_verdicts'],{})
                self.assertEqual(result['final_reviews'],1)

    def test_planning_errors_and_budget_exhaustion_are_not_review_service_errors(self):
        for status,view_ids in (('interface_error',['main','zoom']),
                                ('model_request_limit',['main','r00-review'])):
            with self.subTest(termination=status):
                self.trace['events'] = self.start_review()+[dict(kind='model_response',round=0,
                    status='api_error',view_ids=view_ids) for _ in range(3)]
                self.trace['termination'] = status
                self.assertEqual(self.run_case()['qualified_reviews_service_error'],0)
        # A completed review from an earlier round must not inherit later planning errors.
        self.trace['events'] = self.start_review()+[
            dict(kind='review_result',round=0,candidate=self.good,answer={'verdict':'uncertain'}),
            dict(kind='model_response',round=1,status='api_error',view_ids=['main','zoom'])]
        self.trace['termination'] = 'interface_error'
        result = self.run_case()
        self.assertEqual(result['qualified_reviews_service_error'],0)
        self.assertEqual(result['qualified_reviews_not_confirmed'],1)

    def test_recovered_review_retry_and_auxiliary_halt_are_not_failed_final_reviews(self):
        self.trace['events'] = self.start_review()+[
            dict(kind='model_response',round=0,status='api_error',view_ids=['main','r00-review']),
            dict(kind='model_response',round=0,status='ok',view_ids=['main','r00-review']),
            dict(kind='review_result',round=0,candidate=self.good,answer={'verdict':'uncertain'})]
        result = self.run_case()
        self.assertEqual(result['qualified_reviews_service_error'],0)
        self.assertEqual(result['qualified_reviews_not_confirmed'],1)
        self.trace['events'] = self.start_review(target='auxiliary')+[
            dict(kind='jev_halt',round=0,reason='jev_uncertain_request_pending',view_ids=['main','r00-review'])]
        result = self.run_case('jev')
        self.assertEqual(result['final_reviews'],0)
        self.assertEqual(result['qualified_reviews_service_error'],0)

    def write_jev_review(self, folder, round_number=0):
        path = folder/'tools'/f'r{round_number:02d}-jev-review.json'
        path.parent.mkdir(parents=True,exist_ok=True)
        probabilities = dict(confirm=.18,reject=.58,uncertain=.24)
        journal = dict(status='ok',verdict='uncertain',choice='reject',probabilities=probabilities,
            abstained=True,min_probability=.7,
            input={'state':{'context':dict(mode='review',round=round_number+1,
                candidate=dict(self.good,target='final'))}},
            raw={'answers':{'review':dict(choice='reject',probabilities=probabilities)}})
        path.write_text(json.dumps(journal),encoding='utf-8')
        events = [dict(kind='candidate',round=round_number,target='final',candidate=self.good),
            dict(kind='review_created',round=round_number,candidate=self.good,view={'view_id':f'r{round_number:02d}-review'}),
            dict(kind='jev_response',round=round_number,status='ok',verdict='uncertain',journal=str(path)),
            dict(kind='review_result',round=round_number,candidate=self.good,answer={'verdict':'uncertain'})]
        return path,journal,events

    def diagnose_with_folder(self, folder):
        return module.diagnose('CityNav',self.public,self.truth,self.trace,dict(arm='jev'),geometry,folder)

    def test_raw_choice_is_distinct_from_threshold_verdict_and_counts_events(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            first,journal,events = self.write_jev_review(folder)
            _,_,second = self.write_jev_review(folder,1)
            self.trace['events'] = events+second
            result = self.diagnose_with_folder(folder)
            raw = result['reviews'][0]['jev_raw']
            self.assertEqual(raw['status'],'ok')
            self.assertEqual(raw['choice'],'reject')
            self.assertEqual(raw['probabilities'],journal['probabilities'])
            self.assertTrue(raw['abstained'])
            self.assertEqual(raw['journal_sha256'],hashlib.sha256(first.read_bytes()).hexdigest())
            total = result['qualified_jev_raw_choices']
            self.assertEqual(total['choices'],dict(confirm=0,reject=2,uncertain=0))
            self.assertEqual((total['total'],total['threshold_abstentions']),(2,2))
            self.assertEqual(total['unit'],'review_events_not_independent_episodes')
            self.assertEqual(result['qualified_review_verdicts'],{'uncertain':2})

    def test_wrong_round_candidate_status_and_verdict_are_not_raw_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            for fault in ('round','candidate','target','status','verdict'):
                with self.subTest(fault=fault):
                    path,journal,events = self.write_jev_review(folder)
                    context = journal['input']['state']['context']
                    if fault=='round': context['round']=2
                    elif fault=='candidate': context['candidate']['bbox']=[0,0,100,100]
                    elif fault=='target': context['candidate']['target']='auxiliary'
                    elif fault=='status': journal['status']='error'
                    else: journal['verdict']='reject'
                    path.write_text(json.dumps(journal),encoding='utf-8')
                    self.trace['events']=events
                    result=self.diagnose_with_folder(folder)
                    self.assertEqual(result['qualified_jev_raw_choices']['journal_statuses'],{'invalid':1})
                    self.assertEqual(sum(result['qualified_jev_raw_choices']['choices'].values()),0)
                    self.assertIsNone(result['reviews'][0]['jev_raw']['choice'])

    def test_missing_and_outside_job_journals_are_separate_from_choices(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)/'job'
            path,journal,events = self.write_jev_review(folder)
            self.trace['events']=events
            path.unlink()
            result=self.diagnose_with_folder(folder)
            self.assertEqual(result['qualified_jev_raw_choices']['journal_statuses'],{'missing':1})
            self.assertIsNone(result['reviews'][0]['jev_raw']['choice'])
            outside=Path(directory)/'another_job.json'
            outside.write_text(json.dumps(journal),encoding='utf-8')
            events[2]['journal']=str(outside)
            raw=self.diagnose_with_folder(folder)['reviews'][0]['jev_raw']
            self.assertEqual(raw['reason'],'journal_outside_job')
            self.assertEqual(raw['status'],'invalid')
            self.assertIsNone(raw['journal_sha256'])

    def test_duplicate_trace_response_and_missing_choice_never_pick_a_default(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory)
            path,journal,events=self.write_jev_review(folder)
            self.trace['events']=events[:3]+[dict(events[2])]+events[3:]
            raw=self.diagnose_with_folder(folder)['reviews'][0]['jev_raw']
            self.assertEqual(raw['reason'],'ambiguous_jev_response')
            del journal['choice']
            path.write_text(json.dumps(journal),encoding='utf-8')
            self.trace['events']=events
            result=self.diagnose_with_folder(folder)
            self.assertEqual(result['qualified_jev_raw_choices']['journal_statuses'],{'missing':1})
            self.assertEqual(sum(result['qualified_jev_raw_choices']['choices'].values()),0)

    def test_failed_service_journal_does_not_contribute_a_choice(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory)
            path,journal,events=self.write_jev_review(folder)
            journal['status']='error'
            events[2]['status']='error'
            path.write_text(json.dumps(journal),encoding='utf-8')
            self.trace['events']=events
            result=self.diagnose_with_folder(folder)
            self.assertEqual(result['qualified_jev_raw_choices']['journal_statuses'],{'service_error':1})
            self.assertEqual(sum(result['qualified_jev_raw_choices']['choices'].values()),0)
            self.assertEqual(result['qualified_reviews_service_error'],1)


if __name__ == '__main__':
    unittest.main()
