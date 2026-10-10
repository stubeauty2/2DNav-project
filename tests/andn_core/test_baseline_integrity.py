"""Protect the first SAM3 experiment from accidental algorithm changes."""
import ast
import hashlib
import subprocess
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
PACKAGE=ROOT/'PSC-AVDN/script/andn_core'
REF='57b9ecb31c2b64967d6dea9aaebdbce92c8f02a8'

def historical(path):
    return subprocess.check_output(['git','show',REF+':'+path],cwd=ROOT)

def functions(source):
    return {n.name:ast.get_source_segment(source,n) for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)}

class RemovePassiveHooks(ast.NodeTransformer):
    """Ignore passive hooks and explicit, default-off STEP 2 extension points."""
    HOOKS={'context','event','move','bbox','parsed'}

    def visit_FunctionDef(self,node):
        allowed = {'qwen_locate_bbox_in_view': {'candidate_count': 0, 'confirmation_views': None, 'search_context': None},
                   'search_and_reach_destination': {'candidate_count': 0, 'confirmation_review': 'off', 'spatial_state': None},
                   'run_turn_and_crop_full_trajectory': {'candidate_count': 0, 'confirmation_review': 'off', 'spatial_mode': 'off'}}.get(node.name, {})
        offset = len(node.args.args) - len(node.args.defaults)
        for i in range(len(node.args.args)-1, offset-1, -1):
            name = node.args.args[i].arg
            if name in allowed:
                self_default = node.args.defaults[i-offset]
                assert isinstance(self_default, ast.Constant) and self_default.value == allowed[name]
                del node.args.args[i]
                del node.args.defaults[i-offset]
        return self.generic_visit(node)

    def visit_If(self,node):
        # Explicitly documented bug fix: complete stop-only instructions used
        # to disappear from evaluation. Other navigation remains frozen.
        for condition in ("stop_only_instruction(ob.get('instructions', ''))",
                          "move_dir.lower() == 'stop' and stop_only_instruction(ob.get('instructions', ''))"):
            if ast.dump(node.test)==ast.dump(ast.parse(condition,mode='eval').body):
                assert not node.orelse
                return None
        for condition in ("spatial_state is not None", "spatial_mode != 'off'", "spatial_mode == 'g5'", "spatial_mode == 'g5' and not found"):
            if ast.dump(node.test)==ast.dump(ast.parse(condition,mode='eval').body):
                assert not node.orelse
                return None  # These branches are unreachable with verified default arguments.
        if ast.dump(node.test) == ast.dump(ast.parse("confirmation_review != 'off'", mode='eval').body):
            assert not node.orelse
            return None  # Explicitly default-off option validation.
        if isinstance(node.test,ast.Name) and node.test.id == 'candidate_count':
            assert not node.orelse
            return None  # Disabled branch cannot execute in the frozen baseline.
        return self.generic_visit(node)

    def visit_Call(self,node):
        if isinstance(node.func,ast.Name) and node.func.id == 'search_and_reach_destination':
            for kw in node.keywords[:]:
                if kw.arg=='spatial_state':
                    assert ast.dump(kw.value)==ast.dump(ast.parse("spatial_state if spatial_mode != 'off' else None",mode='eval').body)
                    node.keywords.remove(kw)
                if kw.arg in ('candidate_count','confirmation_review'):
                    assert isinstance(kw.value,ast.Name) and kw.value.id == kw.arg
                    node.keywords.remove(kw)
        return self.generic_visit(node)

    def visit_Expr(self,node):
        call=node.value
        if isinstance(call,ast.Call) and isinstance(call.func,ast.Attribute) and isinstance(call.func.value,ast.Name) and call.func.value.id=='_audit':
            assert call.func.attr in self.HOOKS,call.func.attr
            return None
        return self.generic_visit(node)

def policy_ast(body):
    return ast.dump(RemovePassiveHooks().visit(ast.parse(body)),include_attributes=False)

class FrozenBaseline(unittest.TestCase):
    def test_all_navigation_policy_statements_are_unchanged(self):
        original=functions(historical('PSC-AVDN/script/ANDH-Full/Search_Confirmation.py').decode('utf-8'))
        actual={}
        for name in ('runner.py','navigation.py','tools/geometry.py','tools/views.py','tools/model.py','tools/sam3.py'):
            actual.update(functions((PACKAGE/name).read_text(encoding='utf-8')))
        for name,body in original.items():
            with self.subTest(name=name):self.assertEqual(policy_ast(body),policy_ast(actual[name]))

    def test_parse_functions_and_prompt_are_verbatim_except_explicit_stop(self):
        source=historical('PSC-AVDN/script/ANDH-Full/Parse.py').decode('utf-8')
        actual={}
        for name in ('parser.py','tools/parsing.py'):actual.update(functions((PACKAGE/name).read_text(encoding='utf-8')))
        for name,body in functions(source).items():
            if name == 'request_dialogue_sequence':
                extension = ('    from .termination import stop_only_instruction, stop_step\n'
                    '    if stop_only_instruction(dialogue):\n'
                    '        steps = [stop_step(dialogue)]\n'
                    '        return steps, json.dumps(dict(steps=steps), ensure_ascii=False)\n')
                self.assertIn(extension, actual[name])
                actual[name] = actual[name].replace(extension, '', 1)
            with self.subTest(name=name):self.assertEqual(body,actual[name])
        def prompt(text):
            return next(ast.dump(n.value,include_attributes=False) for n in ast.parse(text).body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='_DIALOGUE_SEQUENCE_SYSTEM_PROMPT')
        self.assertEqual(prompt(source),prompt((PACKAGE/'tools/parsing.py').read_text(encoding='utf-8')))

    def test_sam3_service_and_category_extraction_are_byte_identical(self):
        for p in (PACKAGE/'tools/grounding').iterdir():
            if p.is_file():self.assertEqual(p.read_bytes(),historical('PSC-AVDN/script/visual_grounding/'+p.name),p.name)
        self.assertEqual((PACKAGE/'tools/category_terms.py').read_text(encoding='utf-8'),historical('PSC-AVDN/script/ANDH-Full/category_terms.py').decode('utf-8').replace('\r\n','\n'))

    def test_shared_runtime_helpers_match_baseline(self):
        for name in ('PSC-AVDN/script/util.py','src/env.py'):
            self.assertEqual((ROOT/name).read_text(encoding='utf-8'),historical(name).decode('utf-8').replace('\r\n','\n'),name)

    def test_no_dependency_on_other_navigation_packages(self):
        for p in PACKAGE.rglob('*.py'):
            tree=ast.parse(p.read_text(encoding='utf-8'))
            for node in ast.walk(tree):
                if isinstance(node,ast.ImportFrom) and node.level==0:self.assertFalse((node.module or '').startswith(('andh_core','visual_grounding','category_terms')),p)
                if isinstance(node,ast.Import):
                    for item in node.names:self.assertFalse(item.name.startswith(('andh_core','visual_grounding')),p)

if __name__=='__main__':unittest.main()
