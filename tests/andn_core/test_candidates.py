"""C1 coordinate binding, independent prompts, previews and movement contracts."""
from contextlib import ExitStack
import json
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'PSC-AVDN/script'), str(ROOT/'src')]
import numpy as np
from andn_core import navigation as nav
from andn_core.tools import candidates as protocol, telemetry as audit, views
from andn_core.tools.model import qwen_locate_bbox_in_view


def proposal(box=(50, 50, 150, 150), **kwargs):
    return dict(source_view_id='search_main', bbox_2d=list(box), confidence=.8,
                supporting_evidence='SECRET_SEARCH_REASON', unresolved_attributes=['SECRET_ATTRIBUTE'], **kwargs)


def selection(cid='r1_c2', view=None, box=(300, 310, 420, 430)):
    return dict(dest_present=True, selected_candidate_id=cid, view_id=view or cid+'_confirm_main',
                bbox_2d=list(box), confidence=.9, reason='independent current view evidence', grid_5x5=[])


class Reply:
    headers = {'Content-Type': 'application/json'}
    def __init__(self, content): self.content = content
    def json(self): return {'choices': [{'message': {'content': json.dumps(self.content)}}]}


class CandidateProtocol(unittest.TestCase):
    def test_search_budget_source_duplicates_and_nonfinite(self):
        a, b = proposal(), proposal((200, 50, 300, 150))
        self.assertEqual(len(protocol.normalize_reply({'candidates':[a,b]}, '')['candidates']), 2)
        self.assertEqual(len(protocol.normalize_reply({'candidates':[a,a,b]}, '')['candidates']), 1)
        for value in (proposal((1,2,float('nan'),5)), dict(a, source_view_id='sam3'), proposal((800,800,900,900))):
            self.assertEqual(protocol.normalize_reply({'candidates':[value]}, '')['candidates'], [])
        self.assertEqual(protocol.normalize_reply({'candidates':[]}, '')['validation_errors'], [])
        self.assertTrue(protocol.normalize_reply([], '')['validation_errors'])

    def test_confirmation_requires_legal_pair_and_main_view(self):
        images = [dict(candidate_id='r1_c2', view_id='r1_c2_confirm_'+r, role=r) for r in ('main','detail')]
        self.assertTrue(protocol.normalize_reply(selection(), '', images)['dest_present'])
        for result in (selection('r1_c1'), selection(view='r1_c2_confirm_detail'), selection(view='r1_c1_confirm_main'),
                       selection(box=(0,0,.5,.5)), selection(box=(1,2,float('inf'),5)),
                       dict(selection(), dest_present='true')):
            self.assertFalse(protocol.normalize_reply(result, '', images)['dest_present'])
        absent = protocol.normalize_reply(dict(dest_present=False, selected_candidate_id=None, view_id=None), '', images)
        self.assertFalse(absent['dest_present'])
        self.assertEqual(absent['validation_errors'], [])

    def test_square_matches_frozen_baseline_helper(self):
        import ast
        source = subprocess.check_output(['git', 'show',
            '57b9ecb31c2b64967d6dea9aaebdbce92c8f02a8:PSC-AVDN/script/ANDH-Full/Search_Confirmation.py'], cwd=ROOT).decode('utf-8')
        fn = next(n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.FunctionDef) and n.name == '_minimal_square_uv_from_bbox')
        namespace = {'FIXED_CROP_SIDE':768}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), '<baseline-square>', 'exec'), namespace)
        for box in ([1,2,750,240], [5,600,40,766], [300,310,420,430]):
            self.assertEqual(protocol.square_uv(box), namespace['_minimal_square_uv_from_bbox'](box))


class CandidateFlow(unittest.TestCase):
    def run_flow(self, replies, crop_failure=False, trace=False, confirmation_review='off'):
        self.temp = tempfile.TemporaryDirectory(prefix='andn-c1-')
        self.addCleanup(self.temp.cleanup)
        self.payloads, self.positions, self.draw_positions = [], [], []
        self.i = 0
        def corners(pos, ob, scale_factor, angle_deg):
            x, y = pos
            return [[x,y],[x+1,y],[x+1,y+1],[x,y+1]]
        def render(c, ob, **kw):
            if crop_failure and c[0][0] >= 1 and 'confirm_main' in (kw.get('save_path') or ''):
                return None
            return np.zeros((768,768,3), dtype=np.uint8)
        def project(x, y, c, ob): return [float(c[0][0])+x/100, float(c[0][1])+y/100]
        def move(pos, heading, meters):
            self.positions.append(pos[:])
            return [pos[0]+10, pos[1]]
        def post(url, **kwargs):
            self.payloads.append(kwargs['json'])
            value = replies[self.i]; self.i += 1
            return Reply(value)
        recorder = audit.Recorder(self.temp.name, {}) if trace else None
        token = audit._run.set(recorder)
        try:
            with ExitStack() as stack:
                for module, name, replacement in (
                    (nav,'generate_view_corners_with_scale',corners), (views,'generate_view_corners_with_scale',corners),
                    (nav,'create_view_image',render), (views,'create_view_image',render),
                    (nav,'_patch_xy_to_latlng',project), (nav,'_move_forward_geo',move),
                    (nav,'draw_pos_on_patch',lambda image,c,pos: self.draw_positions.append(pos[:])),
                    (views,'draw_pos_on_patch',lambda image,c,pos: self.draw_positions.append(pos[:])),
                    (nav,'make_minimap_crop_basemap',lambda *a,**kw: np.zeros((10,10,3),dtype=np.uint8)),
                    (nav,'_sam3_image5',lambda *a,**kw: (np.zeros((768,768,3),dtype=np.uint8),{}))):
                    stack.enter_context(patch.object(module,name,replacement))
                stack.enter_context(patch('andn_core.tools.geometry._patch_xy_to_latlng',project))
                stack.enter_context(patch('requests.post',post))
                stack.enter_context(patch('cv2.imwrite',return_value=True))
                result = nav.search_and_reach_destination('example_step01', {}, 'blue roof', '', [0,0], 0, 5,
                    self.temp.name, candidate_count=2, api_key='test-only', confirmation_review=confirmation_review)
        finally:
            audit._run.reset(token)
        self.record = json.loads((Path(self.temp.name)/'example_step01_search.json').read_text(encoding='utf-8'))
        self.recorder = recorder
        return result

    def test_select_second_moves_only_after_confirmation_and_binds_geometry(self):
        result = self.run_flow([{'candidates':[proposal(),proposal((200,50,300,150))]},selection()], trace=True)
        self.assertTrue(result[0])
        self.assertEqual(result[1],[2.5,1.0])
        self.assertEqual(len(result[2]),3)  # search, selected center, final square; no first preview.
        self.assertEqual(result[2][1][0].tolist(),[2.5,1.0])
        self.assertEqual(result[2][2][0].tolist(),[5.5,4.1])
        self.assertEqual(self.positions,[])
        self.assertEqual(self.draw_positions,[[0,0],[0,0],[0,0]])
        self.assertEqual(self.record['steps'][0]['candidates'][0]['status'],'pending')
        confirm = self.payloads[1]
        text = json.dumps(confirm, ensure_ascii=False)
        self.assertNotIn('SECRET_SEARCH_REASON', text)
        self.assertNotIn('SECRET_ATTRIBUTE', text)
        self.assertNotIn('Previous step grid', text)
        self.assertNotIn('Image 4', text)
        self.assertIn('Simple Direct Localization', text)
        self.assertIn('Reasoning Chain', text)
        self.assertIn('Compare the candidate-centered views', text)
        content = confirm['messages'][1]['content']
        self.assertEqual(sum(b['type']=='image_url' for b in content),6)
        for i, block in enumerate(content):
            if block['type']=='image_url': self.assertIn('r1_c',content[i-1]['text'])
        search_text = self.payloads[0]['messages'][0]['content']
        for marker in ('Destination parsing','Scene understanding','5 rows','TOP HALF','shadows','up to 2'):
            self.assertIn(marker,search_text)
        self.assertNotIn('EXACTLY ONE',search_text)
        self.assertNotIn('dest_present=false',search_text)
        self.assertNotIn('Return exactly one final bbox',search_text)
        self.assertIn('[[c11,c12,c13,c14,c15]',search_text)
        calls = [json.loads(p.read_text(encoding='utf-8')) for p in (self.recorder.root/'calls').glob('*.json') if not p.name.endswith('.request.json')]
        confirmation = next(c for c in calls if c['kind']=='confirmation')
        for call in calls:
            self.assertEqual(call['context']['trajectory_id'],'example')
            self.assertEqual(call['context']['stage_id'],'example_step01')
            self.assertEqual(call['context']['round'],1)
        self.assertEqual(len(confirmation['images']),6)
        self.assertEqual([i['view_id'] for i in confirmation['images']], [v['view_id'] for v in self.record['steps'][0]['previews']])
        self.assertEqual(sum(i['localization_allowed'] for i in confirmation['images']),2)
        self.assertEqual(self.recorder.errors,0)

    def test_invalid_selection_continues_from_anchor_no_first_candidate_fallback(self):
        replies = []
        for n in range(3): replies.extend([{'candidates':[proposal(),proposal((200,50,300,150))]},selection('unknown')])
        result = self.run_flow(replies)
        self.assertFalse(result[0]); self.assertIsNone(result[1])
        self.assertEqual(len(self.payloads),6)
        self.assertEqual(self.positions,[[0,0],[10,0],[20,0]])
        self.assertEqual(len(result[2]),4)  # Only search anchors and baseline failure fallback.
        self.assertEqual(self.record['final_pos'],[10,0])

    def test_empty_candidates_skip_confirmation(self):
        result = self.run_flow([{'candidates':[]}]*3)
        self.assertFalse(result[0]); self.assertEqual(len(self.payloads),3)
        self.assertEqual(self.positions,[[0,0],[10,0],[20,0]])

    def test_failed_main_crop_never_submitted_or_selected(self):
        result = self.run_flow([{'candidates':[proposal()]}]*3, crop_failure=True)
        self.assertFalse(result[0]); self.assertEqual(len(self.payloads),3)
        self.assertTrue(all(s['candidates'][0]['status']=='unavailable' for s in self.record['steps'] if s.get('candidates')))

    def test_api_failure_remains_failure_without_selection(self):
        image = np.zeros((20,20,3),dtype=np.uint8)
        with patch('requests.post',side_effect=RuntimeError('offline failure')):
            result = qwen_locate_bbox_in_view('roof', image, candidate_count=2)
        self.assertFalse(result['dest_present']); self.assertNotIn('candidates',result)

    def test_review_reject_advances_anchor_and_consumes_one_stage_budget(self):
        budgets=[]
        def review(*args):
            initial=args[2];available=args[-1];budgets.append(available)
            result=dict(initial)
            if available:result.update(dest_present=False,selected_candidate_id=None,view_id=None,bbox_2d=None)
            return result,dict(attempted=available,applied='reject' if available else 'initial')
        replies=[{'candidates':[proposal()]},selection('r1_c1'),
                 {'candidates':[proposal()]},selection('r2_c1')]
        with patch.object(protocol,'verify_confirmation',side_effect=review):
            result=self.run_flow(replies,confirmation_review='once',trace=True)
        self.assertTrue(result[0]);self.assertEqual(budgets,[True,False])
        self.assertEqual(self.positions,[[0,0]])
        self.assertTrue(self.record['steps'][0]['confirm']['dest_present'])
        self.assertFalse(self.record['steps'][0]['applied_confirmation']['dest_present'])
        self.assertEqual(self.recorder.errors,0)

    def test_review_revision_changes_final_box_without_changing_search_center(self):
        def review(*args):
            return dict(args[2],bbox_2d=[200,210,240,250]),dict(attempted=True,applied='revise')
        replies=[{'candidates':[proposal()]},selection('r1_c1')]
        with patch.object(protocol,'verify_confirmation',side_effect=review):
            result=self.run_flow(replies,confirmation_review='once')
        self.assertTrue(result[0]);self.assertEqual(result[1],[1.0,1.0])
        self.assertEqual(self.record['steps'][0]['confirm']['bbox_2d'],[300,310,420,430])
        self.assertEqual(self.record['steps'][0]['applied_confirmation']['bbox_2d'],[200,210,240,250])
        self.assertEqual(result[2][-1][0].tolist(),[3.0,3.1])


if __name__=='__main__': unittest.main()
