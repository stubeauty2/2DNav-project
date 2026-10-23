import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('ablation', Path(__file__).with_name('usage_ablation.py'))
ab = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ab)


def case(ident, episode, pixel):
    return dict(case_id=ident, dataset='AVDN', episode_id=episode,
        context=dict(mode='review', instruction_raw='full instruction', turns=['extra condition'],
                     candidate={'bbox':[1,2,3,4]}, navigation={'last_executed_moves':['move']},
                     rounds_remaining=5, views=[{'view_id':'main'},{'view_id':'detail'}]),
        assets=[dict(view_id=v, image_path=pixel+v, sha256=pixel+v) for v in ('main','detail')])


class AblationTest(unittest.TestCase):
    def test_substitution_preserves_text_and_does_not_mutate_source(self):
        a,b=case('a','episode1','rgb1'),case('b','episode2','rgb2')
        original=copy.deepcopy(a)
        changed=ab.transformed(a,'substituted_images',b)
        self.assertEqual(a,original)
        self.assertEqual(changed['context'],a['context'])
        self.assertEqual(ab.image_key(changed),ab.image_key(b))
        self.assertNotEqual(ab.image_key(changed),ab.image_key(a))
        self.assertEqual([v['view_id'] for v in changed['assets']], ['main','detail'])

    def test_compact_preserves_full_instruction_and_additional_turns(self):
        a=case('a','episode1','rgb1')
        changed=ab.transformed(a,'compact')
        for field in ('instruction_raw','turns','candidate','views'):
            self.assertEqual(changed['context'][field],a['context'][field])
        self.assertEqual(changed['assets'],a['assets'])
        self.assertNotIn('rounds_remaining',changed['context'])
        self.assertNotIn('navigation',changed['context'])

    def test_donors_cannot_use_same_episode_or_same_pixels(self):
        cases=[case('a','e1','rgb1'),case('b','e2','rgb2'),case('c','e1','rgb3')]
        mapping=ab.donor_map(cases)
        self.assertEqual(mapping['a'],'b')
        self.assertEqual(mapping['c'],'b')
        with self.assertRaises(ValueError):
            ab.donor_map([case('a','e1','same'),case('b','e2','same')])

    def test_online_gate_checks_each_frozen_attempt_not_report_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            jobs=[dict(dataset='AVDN',model='m',arm='a',episode_id=str(i)) for i in range(240)]
            ab.fixed.save(root/'plan.json',{'jobs':jobs})
            digest=ab.fixed.sha(root/'plan.json')
            with self.assertRaises(RuntimeError):
                ab.require_terminal(root,digest)
            for i in range(240):
                ab.fixed.save(root/'episodes'/'AVDN'/'m'/'a'/str(i)/'attempt.json',
                              {'status':'failed' if i==1 else 'complete'})
            ab.require_terminal(root,digest)
            ab.fixed.save(root/'episodes/AVDN/m/a/239/attempt.json',{'status':'pending'})
            with self.assertRaises(RuntimeError):
                ab.require_terminal(root,digest)
            with self.assertRaises(ValueError):
                ab.require_terminal(root,'changed-hash')


if __name__ == '__main__':
    unittest.main()
