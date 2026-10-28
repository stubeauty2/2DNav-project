"""Check paired outcomes, uncertainty, and comparability without model requests."""
import copy
import unittest
from paired_stats import condition,summarize_pairs,wilson


def rows(outcomes):
    return [dict(dataset='AVDN',episode_id=str(i),arm=arm,success=bool(pair[j]),requests=1)
        for i,pair in enumerate(outcomes) for j,arm in enumerate(('before','after'))]


class StatisticsTests(unittest.TestCase):
    def test_pairing_is_not_marginal_success_subtraction(self):
        x=summarize_pairs(rows([(0,1)]*4+[(1,0)]+[(1,1)]*2+[(0,0)]*3))
        self.assertEqual((x['n'],x['before_success'],x['after_success']),(10,3,6))
        self.assertEqual((x['wins'],x['losses'],x['both_success'],x['both_fail']),(4,1,2,3))
        self.assertAlmostEqual(x['mcnemar_exact_two_sided_p'],.375)
        self.assertAlmostEqual(x['paired_rate_difference'],.3)
        self.assertLessEqual(x['paired_bootstrap95'][0],.3)
        self.assertGreaterEqual(x['paired_bootstrap95'][1],.3)
        self.assertEqual(x,summarize_pairs(list(reversed(rows([(0,1)]*4+[(1,0)]+[(1,1)]*2+[(0,0)]*3)))))

    def test_no_discordance_is_not_significant_and_zero_success_is_uncertain(self):
        x=summarize_pairs(rows([(0,0)]*10))
        self.assertEqual(x['mcnemar_exact_two_sided_p'],1)
        self.assertEqual(x['paired_rate_difference'],0)
        self.assertTrue(x['paired_bootstrap_degenerate'])
        self.assertAlmostEqual(wilson(0,10)[1],.2775328,places=6)
        self.assertAlmostEqual(wilson(10,10)[0],1-wilson(0,10)[1])
        self.assertEqual(summarize_pairs(rows([(0,1)]*6))['mcnemar_exact_two_sided_p'],.03125)

    def test_incomplete_and_repeated_pairs_are_rejected(self):
        rr=rows([(0,1)])
        for bad in [[],rr[:1],rr+[rr[0]]]:
            with self.assertRaises(ValueError):summarize_pairs(bad)

    def test_model_confirmation_is_separate_from_metric_success(self):
        rr=rows([(0,0),(0,1)])
        for r in rr:r['termination']='confirmed_final' if r['arm']=='after' else 'step_limit'
        x=summarize_pairs(rr)
        self.assertEqual((x['after_confirmed_finals'],x['after_success'],x['after_confirmed_metric_failures']),(2,1,1))
        self.assertEqual(x['before_confirmed_finals'],0)

    def test_reasoning_model_budget_and_runtime_changes_cannot_be_pooled(self):
        plan=dict(scope='visible',sources={'runtime/after/a.py':'abc'},budgets={'max_requests':32},
            decision_model='qwen3.5-flash',visual_model='qwen3.8-flash',url='test')
        self.assertEqual(condition(plan),condition(dict(plan,thinking_budget=0)))
        for key,value in [('thinking_budget',1024),('decision_model','different'),('budgets',{'max_requests':48}),('sources',{'runtime/after/a.py':'def'})]:
            changed=copy.deepcopy(plan);changed[key]=value
            self.assertNotEqual(condition(plan),condition(changed))


if __name__=='__main__':unittest.main()
