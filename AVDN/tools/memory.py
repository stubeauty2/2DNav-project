"""Observation history, tool context presentation and evidence-cycle state."""
import copy
import hashlib
import re
from pathlib import Path
from .runtime import digest, runtime_hash
from .grounding_skill import GroundingSkill, category_query, CATEGORIES, proposal_overlay
from .visual_evidence import VisualEvidence


def observation_key(view):
    return digest(dict(image_sha256=hashlib.sha256(Path(view['image_path']).read_bytes()).hexdigest(),
                       footprint=view.get('world_corners_xy_m'),
                       orientation=view.get('image_up_heading_deg')))

def evidence_summary(result):
    """Keep fused evidence; detailed per-backend boxes stay in the fusion journal."""
    out = copy.deepcopy(result)
    if 'sources' in out:
        out['sources'] = [dict(backend=s['backend'], status=s['status'],
                               candidate_count=len(s.get('candidates', []))) for s in out['sources']]
    return out

def build_history_context(memory, observation_id):
    records = memory.get('loop_records', [])
    evidence = memory.get('observation_evidence', {}).get(observation_id, [])
    # Latest outcome is always explicit, including when the preceding loop did not move.
    return dict(previous_loop=copy.deepcopy(records[-1]) if records else None,
                recent_loops=copy.deepcopy(records[-3:]),
                current_observation=dict(observation_id=observation_id,
                    prior_loops=sum(r['observation_id']==observation_id for r in records),
                    evidence=[dict(evidence_id=e['evidence_id'], action=e['action'],
                                   result=evidence_summary(e['result']), view_ids=e['view_ids']) for e in evidence[-8:]],
                    omitted_evidence_count=max(0,len(evidence)-8)),
                latest_review=copy.deepcopy(memory.get('latest_review')),
                note='Recorded model reasons/reviews are fallible judgments; tool results are fallible observations. Old coordinates belong only to their source view.')

def tool_key(view, action, runtime_id):
    parameters = {k:v for k,v in action.items() if k not in ('reason','view_id','action')}
    return digest(dict(observation=observation_key(view), parameters=parameters, runtime=runtime_id))

def presentation(context):
    """Copy and label facts without pruning, ranking, or inventing observations."""
    out = copy.deepcopy(context)
    budget = out.get('optional_tools_remaining', 0)
    tools = list(out.get('available_grounding_tools', [])) + list(out.get('available_evidence_tools', []))
    if out.get('capabilities', {}).get('zoom'):
        tools.append('zoom')
    out['tools_callable_now'] = tools if budget > 0 else []
    out['skills_callable_now'] = list(out.get('available_skills', [])) if budget > 0 else []
    out['allowed_next_actions'] = ['locate', 'explore', 'give_up'] + (['tool'] if budget > 0 else [])
    if out['skills_callable_now']:out['allowed_next_actions'].append('skill')
    out['tool_result_contract'] = dict(coordinates='0..1000 xyxy in each result view_id',
        proposal_meaning='candidate region; identity unconfirmed',
        empty_meaning='no proposal returned for this query; not target absence',
        error_meaning='no usable tool observation', score_meaning='within-tool score, not target correctness')

    def annotate(value):
        if isinstance(value, dict):
            if 'backend' in value and 'candidates' in value:
                value['candidate_count'] = len(value['candidates'])
                value['query_scope'] = 'source image only; no unseen region searched'
                for c in value['candidates']:
                    b = c['bbox']; x=(b[0]+b[2])/2; y=(b[1]+b[3])/2
                    c['image_region'] = ('upper' if y < 333.33 else 'lower' if y > 666.67 else 'middle') + '-' + ('left' if x < 333.33 else 'right' if x > 666.67 else 'center')
            for item in value.values():
                annotate(item)
        elif isinstance(value, list):
            for item in value:
                annotate(item)
    annotate(out.get('feedback', []))
    return out

def presentation_v2(context):
    """Prioritize executed facts without gating actions or choosing candidates."""
    out = presentation(context)
    if out.get('jev_questions_remaining', 1) <= 0:
        out['tools_callable_now'] = [t for t in out['tools_callable_now'] if t != 'jev']
    views = out.get('views', [])
    main = next((v for v in views if v.get('source') == 'main'), views[0] if views else {})
    memory = out.get('memory', {})
    visits = memory.get('visits', [])
    current = dict(view_id=main.get('view_id'), image_up_heading_deg=main.get('image_up_heading_deg'),
        round=out.get('round'), rounds_remaining=out.get('rounds_remaining'),
        optional_tools_remaining=out.get('optional_tools_remaining'),
        tools_callable_now=out['tools_callable_now'], allowed_next_actions=out['allowed_next_actions'],
        recorded_visits_to_this_view=sum(v.get('view_id') == main.get('view_id') for v in visits),
        execution_errors=[{k: f[k] for k in ('error', 'rejected_action', 'invalid_response') if k in f}
                          for f in out.get('feedback', []) if 'error' in f])
    # These are model-generated opinions. Keep actions but avoid replaying their
    # persuasive prose as if it were a fresh observation. Raw requests stay intact.
    for action in memory.get('recent_decisions', []):
        action.pop('reason', None)
    memory['recent_decisions_note'] = 'Action history only; old reasons are hypotheses. Review conflicts remain below.'
    current['rejected_candidates'] = copy.deepcopy(memory.get('candidates', []))
    fields = dict(locate=['action','target','view_id','bbox','reason'],
        skill=['action','skill','view_id','query','reason','backends (optional)'],
        visual_question=['action','tool','view_id','question','regions','reason'],
        grounder=['action','tool','view_id','query','reason'],
        jev=['action','tool','view_id','questions','reason'],
        zoom=['action','tool','view_id','bbox','reason'],
        explore_point=['action','mode','view_id','point','reason'],
        explore_offset=['action','mode','turn_deg','step_scale','reason'],
        explore_landmark=['action','mode','landmark_id','reason'],give_up=['action','reason'])
    return dict(current_request=current, action_fields=fields, **out)

def presentation_adaptive(context):
    """Deliver each observation once; preserve candidates, probabilities and facts.

    Full execution history remains in trace.json. This only removes duplicate
    presentations and repeated model opinions, not alternatives or source evidence.
    """
    out=presentation(context) if context.get('mode')=='decision' else copy.deepcopy(context)
    def action(a):
        return {k:copy.deepcopy(v) for k,v in (a or {}).items() if k!='reason'}
    def candidate(c):
        return {k:copy.deepcopy(c[k]) for k in ('view_id','bbox','center_xy_m','target','status','reason') if k in c}
    def review(r):
        if not r:return r
        return {k:(candidate(v) if k in ('candidate','refined') else copy.deepcopy(v))
                for k,v in r.items() if k in ('answer','candidate','refined','review_view_id','round','target','evidence_ids')}
    def loop(r):
        if not r:return r
        item={k:copy.deepcopy(r[k]) for k in ('loop_id','view_id','start_position_xy_m','end_position_xy_m','tools','termination') if k in r}
        judgments=[a for a in r.get('judgments',[]) if a.get('action')!='review']
        item['last_judgment']=copy.deepcopy(judgments[-1]) if judgments else None
        outcome=r.get('outcome') or {}
        item['outcome']={k:copy.deepcopy(outcome[k]) for k in ('type','xy_m') if k in outcome}
        if 'answer' in outcome:item['outcome']['verdict']=outcome['answer'].get('verdict')
        return item
    history=out.get('history',{})
    if history:
        history['previous_loop']=loop(history.get('previous_loop'))
        history['recent_loops']=[loop(r) for r in history.get('recent_loops',[])[:-1]]
        history['latest_review']=review(history.get('latest_review'))
    evidence=history.get('current_observation',{}).get('evidence',[])
    if out.get('mode')=='review':
        reviewed=out.get('review_evidence',{})
        if 'previous_review' in reviewed:reviewed['previous_review']=review(reviewed['previous_review'])
        evidence=reviewed.get('current_observation',{}).get('evidence',[])
    for e in evidence:e['action']=action(e.get('action'))
    ids={e['evidence_id'] for e in evidence}
    answer_ids={a.get('evidence_id') for e in evidence for a in e.get('result',{}).get('answers',[])}
    feedback=[];errors={}
    for f in out.get('feedback',[]):
        if 'error' in f:
            item={k:copy.deepcopy(v) for k,v in f.items() if k!='rejected_action'}
            if 'rejected_action' in f:item['rejected_action']=action(f['rejected_action'])
            key=digest(item)
            previous=errors.pop(key,{})
            errors[key]=dict(item,repeated_count=previous.get('repeated_count',0)+1)
        elif f.get('tool_result',{}).get('evidence_id') in ids:
            feedback.append(dict(evidence_ref=f['tool_result']['evidence_id'],cache_hit=f['tool_result'].get('cache_hit',False)))
        else:feedback.append(f)
    out['feedback']=feedback
    out['latest_feedback']=list(errors.values())[-2:]
    out['execution_error_count']=sum(x['repeated_count'] for x in errors.values())
    memory=out.get('memory',{})
    memory['candidates']=[candidate(c) for c in memory.get('candidates',[])]
    memory['recent_decisions']=[action(a) for a in memory.get('recent_decisions',[])]
    if 'visual_evidence' in out:
        out['visual_evidence']=[a for a in out['visual_evidence'] if a.get('evidence_id') not in answer_ids]
    out.pop('current_request',None)
    out.pop('tool_attempts',None)  # Current evidence + prior loop tool refs contain executed facts.
    out.pop('action_fields',None)  # The single adaptive system contract supplies exact schemas.
    out.pop('grounding_backends',None)  # Duplicates capabilities/available tools.
    if out.get('mode')=='decision' and out.get('jev_questions_remaining',1)<=0:
        out['tools_callable_now']=[t for t in out['tools_callable_now'] if t!='jev']
    if out.get('mode')=='decision':
        # Counts this decision, an evidence call, a subsequent decision and review.
        remaining=out.get('model_requests_remaining',48)
        out['tools_callable_now']=[t for t in out['tools_callable_now']
            if remaining >= (6 if t=='grounding' else 4 if t=='visual_question' else 3)]
        if remaining<6:out['skills_callable_now']=[]
        out['allowed_next_actions']=['locate','explore','give_up']+(['tool'] if out['tools_callable_now'] else [])+(['skill'] if out['skills_callable_now'] else [])
        if 'allowed_actions' in out:out['allowed_actions']=out['allowed_next_actions']
    focus=out.get('evidence_focus')
    if out.get('mode')=='decision' and focus:
        out['tools_callable_now']=[t for t in out['tools_callable_now'] if t in focus['allowed_tools']]
        out['skills_callable_now']=[s for s in out['skills_callable_now'] if s in focus['allowed_skills']]
        out['available_evidence_tools']=[t for t in out.get('available_evidence_tools',[]) if t in focus['allowed_tools']]
        out['allowed_next_actions']=['explore','give_up']+(['tool'] if out['tools_callable_now'] else [])+(['skill'] if out['skills_callable_now'] else [])
        if 'allowed_actions' in out:out['allowed_actions']=out['allowed_next_actions']
    def compact_geometry(value):
        if isinstance(value,dict):
            if 'public_geometry' in value:
                value['public_geometry']={k:v for k,v in value['public_geometry'].items()
                    if k not in ('source','note','reference_note')}
            for v in value.values():compact_geometry(v)
        elif isinstance(value,list):
            for v in value:compact_geometry(v)
    compact_geometry(out)
    return out


def attempt_index(events):
    """Factual execution index only, without GT, rank or interpreted labels."""
    result = []
    for index, event in enumerate(events):
        if event.get('kind') not in ('tool','skill'):
            continue
        action, output = event['input'], event['output']
        item = dict(event_index=index, round=event['round'], tool=action.get('tool',action.get('skill')),
                    invocation=event['kind'],
                    view_id=action['view_id'], status=output.get('status'))
        for key in ('query', 'bbox', 'questions','question','regions','backends'):
            if key in action:
                item[key] = copy.deepcopy(action[key])
        if 'candidates' in output:
            item['candidate_count'] = len(output['candidates'])
        result.append(item)
    return result

class EvidenceCycle:
    def evidence_focus(self, remaining_tools):
        """A bounded source-change decision after review conflict or cached loops.

        This does not call a tool or choose a target. The 3.5 decision model selects
        a new evidence source/question (or explores/stops) in a restricted recovery
        step; the normal decision loop resumes after new source evidence.
        """
        if self.grounding_mode!='adaptive' or remaining_tools<=0:return None
        if self.observation_executions.get(self.observation_id,0)>=self.max_observation_tools:return None
        requests_remaining=getattr(self.model,'max_requests',48)-self.model.cursor
        if requests_remaining<3:return None
        previous=self.memory.get('latest_review')
        evidence=self.memory['observation_evidence'].get(self.observation_id,[])
        conflict=False
        if (previous and previous['observation_id']==self.observation_id and
                previous['answer']['verdict']!='confirm'):
            fresh=[e for e in evidence if e['evidence_id'] not in previous['evidence_ids']
                and e['action'].get('tool',e['action'].get('skill'))!='jev'
                and e['result'].get('status') in ('ok','partial')]
            conflict=not fresh
        recent=[]
        for r in self.memory['loop_records'][-1:]:
            if r['observation_id']==self.observation_id:recent.extend(r['tools'])
        if self.active_loop:recent.extend(self.active_loop['tools'])
        stalled=len(recent)>=2 and all(t.get('cache_hit') for t in recent[-2:])
        if not (conflict or stalled):return None
        tools=['zoom']+[n for n,s in self.tool_status.items() if s=='available']
        if getattr(self.model,'visual_model',None) and requests_remaining>=4:
            tools.insert(0,'visual_question')
        return dict(reason='Unresolved review contradiction' if conflict else 'Two consecutive cached calls added no evidence',
            allowed_tools=tools,allowed_skills=['grounding'] if requests_remaining>=6 and any(s=='available' for s in self.tool_status.values()) else [],
            preferred_tool='visual_question' if 'visual_question' in tools else 'zoom',
            instruction='Choose a NEW evidence source/question, explore, or give_up. Jev repetitions and another locate are not allowed in this recovery step. No target is selected by this policy.')

    def init_evidence(self, grounding_mode, max_observation_tools):
        if grounding_mode not in ('direct','skill','adaptive'):
            raise ValueError('Unknown grounding mode')
        if grounding_mode in ('skill','adaptive') and self.observation_grounder:
            raise ValueError('Skill starts with model viewing RGB; disable observation_grounder')
        self.grounding_mode=grounding_mode
        self.max_observation_tools=max_observation_tools
        self.tool_cache={}
        self.observation_executions={}
        self.observation_id=None
        self.runtime_id=runtime_hash()
        self.active_loop=None
        self.decision_focus=None
        self.memory.update(loop_records=[],observation_evidence={},latest_review=None,review_records=[])
        self.grounding_skill=GroundingSkill(self.grounders,self.model)
        self.visual_tool=VisualEvidence(self.model)
        if grounding_mode=='adaptive':
            self.skill_backends=tuple(self.skill_backends if self.skill_backends is not None else
                [name for name in ('dino','sam3') if self.tool_status.get(name)=='available'])
            if self.skill_backends and (len(set(self.skill_backends))!=len(self.skill_backends) or
                    set(self.skill_backends)-set(self.grounders)):
                raise ValueError('Invalid configured skill backends')
        target=self.episode['initialization']['final_target']
        matches=[(m.start(),-len(category),category) for category in CATEGORIES
                 for m in [re.search(r'\b'+re.escape(category)+r's?\b',target,re.I)] if m]
        self.search_target=dict(description=target,
            category_hint=min(matches)[2] if matches else None,
            source='First category in fallible initialization.final_target; verify against original instruction')

    def evidence_context(self, context):
        context['memory']={k:copy.deepcopy(self.memory[k]) for k in
                           ('visits','confirmed_auxiliary','candidates','recent_decisions')}
        if self.grounding_mode not in ('skill','adaptive'):return context
        context.update(grounding_mode=self.grounding_mode,grounding_categories=list(CATEGORIES),
            search_target=copy.deepcopy(self.search_target),
            available_grounding_tools=['grounding'],grounding_backends=dict(self.tool_status),
            history=build_history_context(self.memory,self.observation_id),
            observation_tools_remaining=max(0,self.max_observation_tools-
                                            self.observation_executions.get(self.observation_id,0)))
        if self.grounding_mode=='adaptive':
            context['model_requests_remaining']=max(0,getattr(self.model,'max_requests',48)-self.model.cursor)
            context['request_budget_note']='Includes this decision; reserve a later decision and independent review. Skill fusion uses 1..3 requests; visual_question uses 1; raw grounders and Jev use no decision-model requests.'
            context.update(available_grounding_tools=[n for n,s in self.tool_status.items() if s=='available'],
                available_skills=['grounding'] if self.skill_backends else [],
                skill_default_backends=list(self.skill_backends),
                skill_available_backends=[n for n,s in self.tool_status.items() if s=='available'])
            # A backend can be selected per invocation even if default sources are disabled.
            if context['skill_available_backends']:context['available_skills']=['grounding']
            if 'allowed_actions' in context and context['mode']=='decision':
                context['allowed_actions'].append('skill')
            if context['mode']=='decision':
                context['evidence_focus']=self.evidence_focus(context['optional_tools_remaining'])
                self.decision_focus=context['evidence_focus']
        return context

    def begin_loop(self, main):
        self.observation_id=observation_key(main)
        self.active_loop=dict(loop_id=self.round+1,observation_id=self.observation_id,
            view_id=main['view_id'],start_position_xy_m=self.position[:],judgments=[],tools=[],outcome=None)

    def finish_loop(self):
        if self.active_loop is None:return
        self.active_loop.update(end_position_xy_m=self.position[:],end_heading_deg=self.heading,
                                termination=self.termination)
        self.memory['loop_records'].append(copy.deepcopy(self.active_loop))
        self.active_loop=None
        self.persist()

    def build_request_views(self, main, extra=None):
        chosen={main['view_id']:main}
        entries=self.memory['observation_evidence'].get(self.observation_id,[])
        ids=[ident for item in entries[-4:] for ident in item['view_ids']]
        if extra:ids.append(extra['view_id'])
        for ident in ids:
            if ident in self.env.views:chosen[ident]=self.env.views[ident]
        if self.grounding_mode=='adaptive' and self.memory['loop_records']:
            previous=self.memory['loop_records'][-1]['view_id']
            if previous!=main['view_id'] and previous in self.env.views:
                # Retain the immediately preceding observed area after movement.
                # Its own frame is explicit; it is not a preview of unseen terrain.
                chosen.pop(previous,None)
                chosen[previous]=dict(self.env.views[previous],source='main_previous')
        return [main]+[v for k,v in chosen.items() if k!=main['view_id']][-4:]

    def execute_tool(self, action, view, tool_index):
        # Evidence tools always inspect unmarked RGB, not annotation colors.
        while view.get('source')=='grounding_overlay':view=self.env.views[view['parent_id']]
        if self.grounding_mode=='adaptive' and view.get('unmarked_image_path'):
            view=dict(view,image_path=view['unmarked_image_path'],annotation=None)
        action=copy.deepcopy(action)
        if action['action']=='skill':
            action['tool']=action.pop('skill')
        if action['tool'] in ('grounding','dino','sam3') or (
                action['tool']=='locate_anything' and self.grounding_mode=='skill'):
            action=dict(action,query=category_query(action['query']))
        if self.grounding_mode=='skill' and action['tool'] in ('dino','sam3','locate_anything'):
            raise ValueError('Use tool grounding to obtain DINO and SAM3 evidence with MLLM fusion')
        if action['tool']=='grounding' and self.grounding_mode=='adaptive':
            selected=action.get('backends',list(self.skill_backends))
            if not selected or any(self.tool_status.get(n)!='available' for n in selected):
                raise ValueError('Select available grounding skill backends')
            action['backends']=sorted(selected)
        if action['tool']=='visual_question':
            if self.grounding_mode!='adaptive' or not getattr(self.model,'visual_model',None):
                raise ValueError('Visual evidence model is disabled')
            for region in action['regions']:self.env.locate(view,region['bbox'])
        if self.grounding_mode=='adaptive' and action['tool'] in ('dino','sam3','locate_anything'):
            if self.tool_status.get(action['tool'])!='available':
                raise ValueError('Requested grounding tool is unavailable; choose a listed tool or skill')
        key=tool_key(view,action,self.runtime_id)
        if key in self.tool_cache:
            result,overlay=self.tool_cache[key]
            return dict(copy.deepcopy(result),evidence_id=key,cache_hit=True),overlay
        used=self.observation_executions.get(self.observation_id,0)
        if used>=self.max_observation_tools:
            raise ValueError('Observation evidence budget exhausted; use existing evidence, explore or give_up')
        if action['tool'] in ('grounding','visual_question') and self.model.cursor>=getattr(self.model,'max_requests',float('inf')):
            from .model import Halt
            raise Halt('model_request_limit')
        self.observation_executions[self.observation_id]=used+1
        overlay=None
        if action['tool']=='grounding':
            skill=self.grounding_skill
            raw=None
            if self.grounding_mode=='adaptive':
                skill=GroundingSkill(self.grounders,self.model,action['backends'])
                raw={}
                for name in skill.backends:
                    direct=dict(action='tool',tool=name,query=action['query'])
                    raw_key=tool_key(view,direct,self.runtime_id)
                    if raw_key in self.tool_cache:
                        raw[name]=copy.deepcopy(self.tool_cache[raw_key][0])
                    else:
                        raw[name],raw_overlay=self.ground(view,name,action['query'],str(tool_index)+'-'+name)
                        self.tool_cache[raw_key]=(copy.deepcopy(raw[name]),raw_overlay)
            result,overlay,extra=skill.run(view,action['query'],
                self.output/'tools'/f'r{self.round:02d}-{tool_index}-grounding',raw_results=raw)
            for c in result['candidates']:
                ref=key[:12]+'/'+c['candidate_id']
                c['object_ref']=ref
                self.object_refs[ref]=dict(view_id=view['view_id'],bbox=c['bbox'])
                if self.grounding_mode=='adaptive':
                    from .geometry import candidate_context
                    try:
                        located=self.env.locate(view,c['bbox'])
                        c['public_geometry']=candidate_context(located['center_xy_m'],self.episode,self.position,self.heading)
                    except ValueError:
                        c['geometry_note']='Proposal touches unavailable raster; inspect its visible portion'
            for v in extra+([overlay] if overlay else []):self.env.views[v['view_id']]=v
            self.event('grounding_fusion',request_indices=result['fusion_requests'],
                       source_views=[v['view_id'] for v in extra],output=result)
        elif action['tool']=='jev':
            result=self.inspect_visual(view,action['questions'],tool_index)
        elif action['tool']=='visual_question':
            result=self.visual_tool.inspect(view,action['question'],action['regions'],
                self.output/'tools'/f'r{self.round:02d}-{tool_index}-visual.json')
            if result.get('proposals'):
                from .geometry import candidate_context
                for index,p in enumerate(result['proposals']):
                    ref=key[:12]+'/rgb-'+str(index)
                    p.update(candidate_id='rgb-'+str(index),object_ref=ref,view_id=view['view_id'],
                             proposal_status='unverified RGB proposal; not a destination verdict')
                    self.object_refs[ref]=dict(view_id=view['view_id'],bbox=p['bbox'])
                    try:
                        located=self.env.locate(view,p['bbox'])
                        p['public_geometry']=candidate_context(located['center_xy_m'],self.episode,self.position,self.heading)
                    except ValueError:p['geometry_note']='Proposal touches unavailable raster; inspect visible portion'
                overlay=proposal_overlay(view,result['proposals'],
                    self.output/'tools'/f'r{self.round:02d}-{tool_index}-visual.png','visual')
                self.env.views[overlay['view_id']]=overlay
        elif action['tool'] in ('dino','sam3','locate_anything'):
            result,overlay=self.ground(view,action['tool'],action['query'],tool_index)
        elif action['tool']=='zoom':
            overlay=self.env.zoom(view,action['bbox'],f'r{self.round:02d}-zoom{tool_index}')
            result=dict(status='ok',tool='zoom',view_id=overlay['view_id'])
        else:
            raise ValueError('Unknown evidence tool')
        result=dict(result,evidence_id=key,cache_hit=False)
        # Errors are also reusable observations of failure, not evidence of absence.
        self.tool_cache[key]=(copy.deepcopy(result),overlay)
        item=dict(evidence_id=key,action=copy.deepcopy(action),result=copy.deepcopy(result),
                  view_ids=[v['view_id'] for v in ([view]+([overlay] if overlay else []))])
        self.memory['observation_evidence'].setdefault(self.observation_id,[]).append(item)
        return result,overlay

    def remember_review(self, answer, candidate, detail, refined=None, target='final'):
        outcome=dict(answer=answer,candidate=candidate,review_view_id=detail['view_id'],
                     observation_id=self.observation_id,round=self.round+1,target=target,
                     evidence_ids=[e['evidence_id'] for e in self.memory['observation_evidence'].get(self.observation_id,[])])
        if refined is not None:outcome['refined']=refined
        self.memory['latest_review']=outcome
        self.memory['review_records'].append(copy.deepcopy(outcome))
        return outcome

    def require_new_review_evidence(self, candidate, target='final'):
        """A near-identical proposal needs new evidence after a non-confirming review."""
        if self.grounding_mode not in ('skill','adaptive'):return
        entries=self.memory['observation_evidence'].get(self.observation_id,[])
        if self.grounding_mode=='adaptive':
            # Failed services and malformed answers do not resolve a review conflict.
            entries=[e for e in entries if e['result'].get('status') in ('ok','partial') or
                     any(a.get('status')=='ok' for a in e['result'].get('answers',[]))]
        now={e['evidence_id'] for e in entries}
        def bounds(polygon):
            return [min(p[0] for p in polygon),min(p[1] for p in polygon),
                    max(p[0] for p in polygon),max(p[1] for p in polygon)]
        for previous in self.memory['review_records']:
            if previous['answer']['verdict']=='confirm' or previous['target']!=target:continue
            if previous['observation_id']!=self.observation_id or now-set(previous['evidence_ids']):continue
            a=bounds(candidate['polygon_xy_m']);b=bounds(previous['candidate']['polygon_xy_m'])
            intersection=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
            union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection
            if union>0 and intersection/union>=.95:
                raise ValueError('This candidate already received a non-confirming review in the same observation. '
                                 'No new evidence since that review; address its conflict, inspect another candidate or explore.')
