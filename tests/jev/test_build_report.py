"""Evidence gating and confusion-count tests for the static report builder."""
import importlib.util
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('build_report', Path(__file__).with_name('build_report.py'))
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def save(self, path, value):
        file = self.root/path
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(json.dumps(value), encoding='utf-8')

    def test_missing_results_never_become_completed_claim(self):
        state = report.build(self.root)
        self.assertFalse(state['online']['complete'])
        page = (self.root/'index.html').read_text(encoding='utf-8')
        self.assertIn('当前不足以判定替换成功', page)
        self.assertIn('未准备', page)

    def test_uncertain_is_separate_and_only_original_labels_used(self):
        folder = self.root/'fixed_review_08b'
        folder.mkdir()
        cases = [dict(case_id=str(i), split='avdn_capacity20x3', input_sha256=str(i)) for i in range(6)]
        (folder/'public_requests.jsonl').write_text('\n'.join(json.dumps(c) for c in cases), encoding='utf-8')
        self.save('fixed_review_08b/evaluator_only.json', {str(i):{'proposal_score':{'correct':i < 3}} for i in range(6)})
        for i, verdict in enumerate(['confirm', 'reject', 'uncertain', 'confirm', 'reject', 'uncertain']):
            self.save(f'fixed_review_08b/responses/{i}.json', dict(input_sha256=str(i), status='ok', verdict=verdict))
        stage = report.fixed_stage(report.Evidence(self.root), 'fixed_review_08b', 'sample')
        self.assertEqual(stage['terminal'], 6)
        self.assertEqual(stage['errors'], 0)
        g = stage['groups']['avdn_capacity20x3']
        self.assertEqual([g[k] for k in ('TP','FN','FP','TN','uncertain_positive','uncertain_negative')], [1]*6)

    def test_input_mismatch_is_error_not_positive(self):
        folder = self.root/'fixed_review_08b'
        folder.mkdir()
        (folder/'public_requests.jsonl').write_text(json.dumps(dict(case_id='one',split='test',input_sha256='a')), encoding='utf-8')
        self.save('fixed_review_08b/evaluator_only.json', {'one':{'proposal_score':{'correct':True}}})
        self.save('fixed_review_08b/responses/one.json', dict(status='ok', input_sha256='b', verdict='confirm'))
        stage = report.fixed_stage(report.Evidence(self.root), 'fixed_review_08b', 'test')
        self.assertEqual(stage['errors'], 1)
        self.assertEqual(stage['groups']['test']['TP'], 0)
        self.assertEqual(stage['groups']['test']['invalid'], 1)

    def test_threshold_below_count_is_not_changed_decision_count(self):
        folder = self.root/'fixed_review_4b'
        folder.mkdir()
        cases = [dict(case_id=str(i), split='test', input_sha256=str(i)) for i in range(4)]
        (folder/'public_requests.jsonl').write_text('\n'.join(json.dumps(c) for c in cases), encoding='utf-8')
        self.save('fixed_review_4b/evaluator_only.json', {str(i):{'proposal_score':{'correct':True}} for i in range(4)})
        for i, (choice, verdict, abstained) in enumerate([
                ('confirm','uncertain',True), ('reject','uncertain',True),
                ('uncertain','uncertain',True), ('reject','reject',False)]):
            self.save(f'fixed_review_4b/responses/{i}.json', dict(status='ok', input_sha256=str(i),
                verdict=verdict, response=dict(choice=choice, abstained=abstained)))
        stage = report.fixed_stage(report.Evidence(self.root), 'fixed_review_4b', 'test')
        self.assertEqual(stage['below_threshold'], 3)
        self.assertEqual(stage['threshold_changed'], {'confirm → uncertain':1,'reject → uncertain':1})
        self.assertIn('原本就选择 uncertain', stage['body'])

    def test_stale_or_different_plan_analysis_cannot_complete(self):
        job = dict(dataset='AVDN',model='qwen3.8-flash',arm='jev',episode_id='one')
        self.save('online/plan.json', dict(jobs=[job]))
        self.save('online/episodes/AVDN/qwen3.8-flash/jev/one/attempt.json', dict(status='complete'))
        self.save('online/analysis.json', dict(plan_sha256='different',all_terminal=True))
        online = report.online_section(report.Evidence(self.root))
        self.assertEqual(online['terminal'], 1)
        self.assertTrue(online['stale'])
        self.assertFalse(online['complete'])

    def test_infrastructure_subset_is_paired_and_never_imputes_pending(self):
        rows=[]
        for ident, left, right in [('a','complete','complete'),('b','complete','complete'),
                                    ('c','complete','pending'),('d','failed','complete')]:
            for arm,status in [('mllm',left),('jev',right)]:
                rows.append(dict(dataset='AVDN',model='flash',episode_id=ident,arm=arm,status=status,
                    success=arm=='mllm',api_errors=int(ident=='b' and arm=='jev'),jev_errors=0,tool_errors=0,
                    model_requests=4 if arm=='mllm' else 12,all_image_mllm_requests=4 if arm=='mllm' else 12,
                    termination='interface_error' if ident=='b' else 'step_limit'))
        data=report.infrastructure_summary({'rows':rows})
        pair=next(r for r in data['pairs'] if r['baseline']=='mllm')
        self.assertEqual(pair['planned'],4)
        self.assertEqual(pair['complete_pairs'],3)
        self.assertEqual(pair['unaffected_pairs'],2)
        self.assertEqual(pair['baseline_successes'],1)
        self.assertEqual(pair['jev_successes'],0)
        self.assertEqual(pair['all_lost'],2)
        self.assertEqual(pair['affected_lost'],1)
        self.assertEqual(pair['mean_mllm_request_change'],8)
        self.assertEqual(pair['mean_image_request_change'],8)
        self.assertEqual(data['api_interface_terminations'],1)
        self.assertEqual(data['affected_episodes'],1)

    def test_recovered_api_error_and_unknown_tool_are_excluded_without_changing_outcomes(self):
        rows=[dict(dataset='CityNav',model='max',episode_id=str(i),arm=arm,status='complete',success=True,
                   api_errors=int(i==0 and arm=='mllm'),jev_errors=0,tool_errors=0,
                   unknown_planned_grounders=['dino'] if i==1 and arm=='jev' else [],termination='confirmed_final')
              for i in range(2) for arm in ('mllm','jev')]
        data=report.infrastructure_summary({'rows':rows})
        pair=next(r for r in data['pairs'] if r['baseline']=='mllm')
        self.assertEqual(pair['complete_pairs'],2)
        self.assertEqual(pair['unaffected_pairs'],0)
        self.assertIsNone(pair['sr_change'])
        self.assertIsNone(pair['mean_image_request_change'])
        self.assertEqual(data['api_interface_terminations'],0)
        self.assertTrue(all(r['success'] for r in rows))

    def test_discordant_cases_include_gains_losses_and_failures_but_not_pending(self):
        rows=[]
        for ident, base_status, base_success, jev_status, jev_success in [
                ('lost','complete',True,'complete',False),
                ('gained','failed',True,'complete',True),
                ('pending','complete',True,'pending',False),
                ('both_failed','complete',False,'failed',True)]:
            for arm,status,success in [('mllm',base_status,base_success),('jev',jev_status,jev_success)]:
                rows.append(dict(dataset='AVDN',model='max',episode_id=ident,arm=arm,
                    job_id=ident+'-'+arm,status=status,success=success,
                    api_errors=int(ident=='lost' and arm=='jev')))
        data=report.discordant_pair_details({'rows':rows},{'rows':[]})
        self.assertEqual([(r['episode_id'],r['direction']) for r in data],[('gained','gained'),('lost','lost')])
        self.assertFalse(data[0]['involves_infrastructure'])
        self.assertTrue(data[1]['involves_infrastructure'])
        self.assertIsNone(data[0]['jev_stage'])

    def test_case_diagnostics_join_exact_job_not_another_model_or_arm(self):
        rows=[dict(dataset='AVDN',model='max',episode_id='one',arm=arm,job_id='one-'+arm,
                   status='complete',success=arm=='mllm') for arm in ('mllm','jev')]
        stages={'rows':[dict(job_id='one-jev',qualified_proposal_reviews=5),
                        dict(job_id='other-jev',qualified_proposal_reviews=100)]}
        pair=report.discordant_pair_details({'rows':rows},stages)[0]
        self.assertEqual(pair['jev_stage']['qualified_proposal_reviews'],5)
        self.assertIsNone(pair['base_stage'])

    def test_stale_stage_cannot_generate_paired_case_table(self):
        self.save('online/analysis.json',{'plan_sha256':'plan','rows':[]})
        self.save('online/stage_analysis.json',dict(source_analysis_sha256='stale',plan_sha256='plan'))
        page=report.stage_section(report.Evidence(self.root))
        self.assertIn('配对案例表等待',page)
        self.assertNotIn('当前已齐配对尚无',page)

    def test_ablation_plan_without_results_does_not_claim_execution(self):
        self.save('usage_ablation_4b/plan.json',dict(jobs=[dict(case_id='one',arm='full')]))
        page=report.usage_ablation_section(report.Evidence(self.root))
        self.assertIn('尚无对照分析结果',page)
        self.assertNotIn('全部对照已完成',page)

    def test_ablation_never_scores_substituted_images_and_binds_response_bytes(self):
        folder='usage_ablation_4b'
        self.save(folder+'/plan.json',dict(jobs=[dict(case_id='one',arm='substituted_images')]))
        self.save(folder+'/responses/substituted_images/one.json',dict(status='ok'))
        def sha(path):
            return hashlib.sha256((self.root/path).read_bytes()).hexdigest()
        self.save(folder+'/analysis.json',dict(complete=True,plan_sha256=sha(folder+'/plan.json'),
            response_sha256={'responses/substituted_images/one.json':sha(folder+'/responses/substituted_images/one.json')},
            rows=[dict(split='citynav_primary14',arm='substituted_images',ok=1,planned=1,
                       verdict_geometry_proxy=dict(TP=999),paired_with_full=dict(n=1))]))
        page=report.usage_ablation_section(report.Evidence(self.root))
        self.assertIn('全部对照已完成',page)
        self.assertIn('不适用（图像已替换）',page)
        self.assertNotIn('999',page)
        self.save(folder+'/responses/substituted_images/one.json',dict(status='changed'))
        page=report.usage_ablation_section(report.Evidence(self.root))
        self.assertIn('证据已变更或缺失',page)
        self.assertNotIn('全部对照已完成',page)

    def test_ablation_different_plan_cannot_show_results(self):
        self.save('usage_ablation_4b/plan.json',dict(jobs=[]))
        self.save('usage_ablation_4b/analysis.json',dict(complete=True,plan_sha256='different'))
        page=report.usage_ablation_section(report.Evidence(self.root))
        self.assertIn('与当前设计不一致',page)
        self.assertNotIn('全部对照已完成',page)

    def test_ablation_windows_manifest_paths_keep_hash_validation(self):
        folder='usage_ablation_4b'
        self.save(folder+'/plan.json',dict(jobs=[dict(case_id='one',arm='full')]))
        self.save(folder+'/responses/full/one.json',dict(status='ok'))
        def sha(path):
            return hashlib.sha256((self.root/path).read_bytes()).hexdigest()
        self.save(folder+'/analysis.json',dict(complete=True,plan_sha256=sha(folder+'/plan.json'),
            response_sha256={'responses\\full\\one.json':sha(folder+'/responses/full/one.json')},rows=[]))
        page=report.usage_ablation_section(report.Evidence(self.root))
        self.assertIn('全部对照已完成',page)
        self.save(folder+'/responses/full/one.json',dict(status='changed'))
        page=report.usage_ablation_section(report.Evidence(self.root))
        self.assertIn('证据已变更或缺失',page)
        self.assertNotIn('全部对照已完成',page)


if __name__ == '__main__':
    unittest.main()
