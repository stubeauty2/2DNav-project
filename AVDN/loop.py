"""Validated action protocol and the observe/explore/locate/review state machine."""
import math
from .analysis import validate_agent_input, validate_result
from pathlib import Path
import numpy as np
from .tools.geometry import offset, bearing, project, square_bbox, normalized
from .tools.model import Halt, parse_action, safe_view
from .tools.context import candidate_context
from .tools.grounding import observation_query
from .tools.runtime import save_json, load_json, digest
from .config import MAX_ROUNDS, MAX_TOOLS_PER_ROUND


def public_episode(episode):
    if set(episode)!={'episode_id','agent_input','initialization','environment'}:
        raise ValueError('Unexpected episode fields: navigation accepts public input only')
    validate_agent_input(episode['agent_input'])
    validate_result(episode['initialization'],episode['agent_input'])
    spec=episode['environment']
    if set(spec)!={'image_path','image_size','world_to_raster','main_width_m','main_height_m'}:
        raise ValueError('Environment must contain only public raster calibration')
    return episode

def number(value):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
        raise ValueError('Coordinate or angle must be a finite number')
    return value

def validate_action(a, review=False):
    # Some providers annotate the already-required corner convention. It is
    # redundant metadata, not a coordinate conversion or a new action.
    if 'bbox_format' in a:
        if a['bbox_format']!='xyxy' or 'bbox' not in a:
            raise ValueError('bbox_format, if present, must be xyxy with a bbox; other conventions are not converted')
        a={k:v for k,v in a.items() if k!='bbox_format'}
    # An explicit tool name as action is an unambiguous spelling alias. Never
    # infer missing coordinates, accept conflicting tools, or relax other fields.
    if not review and a.get('action') in ('zoom','sam3','dino','locate_anything'):
        name=a['action']
        if a.get('tool',name)!=name:
            raise ValueError('Conflicting action/tool names')
        a=dict(a,action='tool',tool=name)
    if review and a.get('verdict') in ('reject','uncertain'):
        # A non-confirming verdict never executes its illustrative box. Retain it
        # in the raw journal, without using it as a fallback localization.
        a={k:v for k,v in a.items() if k not in ('view_id','bbox')}
    if review:
        if a.get('action')!='review' or a.get('verdict') not in ('confirm','reject','uncertain'):
            raise ValueError('Review requires action review, verdict confirm/reject/uncertain')
        keys={'action','verdict','reason'}
        if a['verdict']=='confirm':
            keys|={'view_id','bbox'}
    elif a.get('action')=='explore':
        mode=a.get('mode')
        fields={'point':{'view_id','point'},'offset':{'turn_deg','step_scale'},'landmark':{'landmark_id'}}
        if mode not in fields:
            raise ValueError('explore mode must be point, offset or landmark')
        keys={'action','mode','reason'}|fields[mode]
        if mode=='offset':
            number(a.get('turn_deg'));number(a.get('step_scale'))
            if a['step_scale']<=0:raise ValueError('step_scale must be positive')
    elif a.get('action')=='locate':
        keys={'action','target','view_id','bbox','reason'}
        if a.get('target') not in ('final','auxiliary'):raise ValueError('Unknown target role')
    elif a.get('action')=='tool':
        if a.get('tool') not in ('zoom','sam3','dino','locate_anything'):raise ValueError('Unknown tool')
        keys={'action','tool','view_id','reason'}|({'bbox'} if a['tool']=='zoom' else {'query'})
    elif a.get('action')=='give_up':
        keys={'action','reason'}
    else:
        raise ValueError('Unknown action')
    if set(a)!=keys:raise ValueError('Action fields must be exactly '+str(sorted(keys)))
    for k in ('point','bbox'):
        if k in a:
            if not isinstance(a[k],list) or len(a[k])!=(2 if k=='point' else 4):
                raise ValueError('Invalid '+k+' length')
            for v in a[k]:number(v)
    for k in ('view_id','landmark_id','query'):
        if k in a and (not isinstance(a[k],str) or not a[k].strip()):
            raise ValueError(k+' must be a nonempty string')
    return a




class Agent:
    def __init__(self, episode, environment, model, sam3, output, max_rounds=MAX_ROUNDS, max_tools=MAX_TOOLS_PER_ROUND, grounders=None, observation_grounder=None, observation_query_mode="full"):
        self.episode=public_episode(episode)
        self.env,self.model,self.sam=environment,model,sam3
        self.output=Path(output);self.output.mkdir(parents=True,exist_ok=True)
        self.max_rounds,self.max_tools=max_rounds,max_tools
        self.observation_cache={}
        self.observation_grounder=observation_grounder
        self.observation_query_mode=observation_query_mode
        source=episode['agent_input'];self.position=list(source['initial_pose']['xy_m'])
        prior=episode['initialization']['direction']
        self.heading=prior if prior is not None else source['initial_pose']['heading_deg']
        if self.heading is None:self.heading=0.
        self.memory=dict(visits=[],confirmed_auxiliary=[],candidates=[],recent_decisions=[])
        self.events=[];self.allowed=set();self.final=None;self.round=0
        self.grounders={'sam3':sam3, **(grounders or {})}
        self.tool_status={name:('available' if tool.available() else 'unavailable')
                          for name,tool in self.grounders.items()}
        for name in ('sam3','dino','locate_anything'):self.tool_status.setdefault(name,'unavailable')
        capability_file=self.output/'tool_capabilities.json'
        if capability_file.exists():self.tool_status=load_json(capability_file)
        else:save_json(capability_file,self.tool_status)
        self.trajectory=[dict(xy_m=self.position[:],heading_deg=self.heading,source='initial')]
        self.termination=None

    def event(self, kind, **details):
        self.events.append(dict(round=self.round,kind=kind,**details))
        self.persist()

    def persist(self):
        save_json(self.output/'trace.json',dict(episode_id=self.episode['episode_id'],
            events=self.events,views=list(self.env.views.values()),delivered_view_ids=sorted(self.allowed),
            memory=self.memory,trajectory=self.trajectory,final=self.final,termination=self.termination,
            model_requests=self.model.cursor))

    def context(self, mode, feedback, tools, candidate=None):
        source=self.episode['agent_input']
        return dict(mode=mode,instruction_raw=source['instruction_raw'],turns=source['turns'],
            initialization=self.episode['initialization'],initial_pose=source['initial_pose'],
            position_xy_m=self.position,heading_deg=self.heading,coordinate_frame=source['coordinate_frame'],
            named_landmarks=source['named_landmarks'],memory=self.memory,
            round=self.round+1,rounds_remaining=self.max_rounds-self.round,
            optional_tools_remaining=self.max_tools-tools,
            capabilities=dict(zoom=True,**self.tool_status),
            available_grounding_tools=[k for k,v in self.tool_status.items() if v=='available'],
            candidate=candidate,feedback=feedback)

    def ask(self, mode, views, tools, candidate=None, feedback=None):
        feedback=list(feedback or [])
        for attempt in range(3):
            record=self.model.request(self.context(mode,feedback,tools,candidate),views)
            self.allowed.update(self.model.successful_views)
            self.event('model_response',request_index=self.model.cursor-1,status=record['status'],
                       view_ids=[v['view_id'] for v in views])
            if record['status']!='ok':
                feedback.append(dict(error='API request failed; retry'))
                continue
            try:
                raw_action=parse_action(record['response']['text'])
                action=validate_action(raw_action,review=mode=='review')
                if action!=raw_action:
                    self.event('format_normalization',raw_action=raw_action,action=action,
                               note='Normalized output syntax; no movement or target confirmation inferred')
                return action
            except (ValueError,KeyError,TypeError) as exc:
                feedback.append(dict(error=str(exc),invalid_response=record['response']['text']))
                self.event('invalid_format',error=str(exc))
        raise Halt('interface_error')

    def remember(self, action):
        self.memory['recent_decisions']=(self.memory['recent_decisions']+[action])[-3:]

    def move(self, xy, heading, source):
        self.position=self.env.check_position(xy);self.heading=heading%360
        self.trajectory.append(dict(xy_m=self.position[:],heading_deg=self.heading,source=source))
        self.event('move',**self.trajectory[-1])

    def main_parent(self, view):
        while view.get('parent_id'):
            view=self.env.views[view['parent_id']]
        return view

    def review(self, proposal, main, candidate):
        detail=self.env.review(candidate,main,f'r{self.round:02d}-review')
        detail=self.env.mark_review(detail,candidate)
        self.event('review_created',candidate=candidate,view=safe_view(detail))
        footprint=np.asarray(normalized(detail,candidate['polygon_xy_m']))
        review_candidate=dict(candidate,target=proposal['target'],
            public_geometry=candidate_context(candidate['center_xy_m'],self.episode,self.position,self.heading),
            projection_in_review=dict(view_id=detail['view_id'],
                bbox=[*footprint.min(axis=0).tolist(),*footprint.max(axis=0).tolist()],
                note='Geometric projection of the unconfirmed proposal; refine or reject from RGB'))
        feedback=[]
        for invalid in range(3):
            answer=self.ask('review',[main,detail],0,candidate=review_candidate,feedback=feedback)
            self.remember(answer)
            if answer['verdict']!='confirm':
                self.memory['candidates'].append(dict(candidate,target=proposal['target'],
                    status=answer['verdict'],reason=answer['reason']))
                self.event('review_result',answer=answer,candidate=candidate)
                return
            try:
                view=self.env.view(answer['view_id'],{main['view_id'],detail['view_id']})
                refined=self.env.locate(view,answer['bbox'])
            except ValueError as exc:
                feedback.append(dict(error=str(exc)));self.event('invalid_review',error=str(exc))
                continue
            self.event('review_result',answer=answer,candidate=candidate,refined=refined)
            if proposal['target']=='auxiliary':
                self.memory['confirmed_auxiliary'].append(dict(refined,reason=answer['reason']))
                self.move(refined['center_xy_m'],bearing(self.position,refined['center_xy_m'],self.heading),'reviewed_auxiliary')
            else:
                self.final=dict(refined,reason=answer['reason'],
                    evaluation_polygon_xy_m=project(view['pixel_to_world'],square_bbox(refined['bbox_pixels'])))
                self.termination='confirmed_final'
                self.event('final_localization',localization=self.final,actual_move=False)
            return
        raise Halt('invalid_review_limit')

    def ground(self, view, name, query, tool_index):
        grounder=self.grounders.get(name)
        if grounder is None:
            return dict(status='unavailable',backend=name,candidates=[],
                error='Service not configured; this is not evidence of absence'),None
        result=grounder.candidates(view,query,self.output/'tools'/f'r{self.round:02d}-{tool_index}.json')
        result={k:v for k,v in result.items() if k not in ('raw','input','signature')}
        overlay=None
        if result['status']=='ok':
            valid=[];rejected=[]
            for candidate in result['candidates']:
                try:located=self.env.locate(view,candidate['bbox'])
                except ValueError as exc:
                    rejected.append(dict(candidate_id=candidate['candidate_id'],error=str(exc)))
                else:valid.append(dict(candidate,public_geometry=candidate_context(
                    located['center_xy_m'],self.episode,self.position,self.heading)))
            result['candidates']=valid;result['rejected_candidates']=rejected
            if valid:
                overlay=self.env.tool_overlay(view,valid,f'r{self.round:02d}-{name}{tool_index}')
        return result,overlay

    def run(self):
        try:
            for self.round in range(self.max_rounds):
                main=self.env.main(self.position,self.heading,f'r{self.round:02d}-main')
                if not self.memory['visits'] or self.memory['visits'][-1]['view_id']!=main['view_id']:
                    self.memory['visits'].append(dict(xy_m=self.position[:],heading_deg=self.heading,
                        view_id=main['view_id'],world_corners_xy_m=main['world_corners_xy_m']))
                views=[main];tools=0;feedback=[]
                if main['view_id'] in self.allowed:
                    feedback.append(dict(observation='This is an already delivered main view, not a new observation. Pose and raster content are unchanged; compare prior candidate rejections in this same view.'))
                # Optional experimental policy: inspect the current observation with a
                # local grounder before deciding to leave it. Counts toward tool limit.
                if self.observation_grounder and self.max_tools>0 and self.tool_status.get(self.observation_grounder)=='available':
                    cached=self.observation_cache.get(main['view_id'])
                    if cached is not None:
                        result,overlay=cached
                        if overlay:views=[main,overlay]
                        feedback.append(dict(tool_result=result,reused_observation=True))
                        self.event('tool_reused',backend=self.observation_grounder,view_id=main['view_id'],
                            note='Cached successful proposal result for the identical main observation; no new tool call')
                    else:
                        tools=1
                        action=dict(action='tool',tool=self.observation_grounder,view_id=main['view_id'],
                            query=observation_query(self.episode['initialization']['final_target'],self.observation_query_mode),
                            reason='Observation proposal policy; full instruction retained for candidate selection')
                        result,overlay=self.ground(main,action['tool'],action['query'],tools)
                        if overlay:views=[main,overlay]
                        feedback.append(dict(tool_result=result))
                        self.event('tool',input=action,output=result,origin='observation_policy')
                        if result['status']=='ok':
                            self.observation_cache[main['view_id']]=(result,overlay)
                while True:
                    action=self.ask('decision',views,tools,feedback=feedback)
                    self.remember(action);self.event('decision',action=action)
                    try:
                        if action['action']=='give_up':
                            self.termination='give_up';break
                        if action['action']=='tool':
                            if tools>=self.max_tools:raise ValueError('Optional tool limit reached; explore, locate or give_up')
                            view=self.env.view(action['view_id'],self.allowed)
                            tools+=1
                            if action['tool']=='zoom':
                                zoom=self.env.zoom(view,action['bbox'],f'r{self.round:02d}-zoom{tools}')
                                result=dict(status='ok',tool='zoom',view_id=zoom['view_id'])
                                views=[main,zoom]
                            else:
                                result,overlay=self.ground(view,action['tool'],action['query'],tools)
                                if overlay:views=[main,overlay]
                            feedback.append(dict(tool_result=result))
                            self.event('tool',input=action,output=result)
                            continue
                        if action['action']=='explore':
                            mode=action['mode']
                            if mode=='offset':
                                xy,h=offset(self.position,self.heading,action['turn_deg'],action['step_scale'],main['height_m'])
                            elif mode=='point':
                                xy=self.env.check_point(self.env.view(action['view_id'],self.allowed),action['point'])
                                h=bearing(self.position,xy,self.heading)
                            else:
                                matches=[l for l in self.episode['agent_input']['named_landmarks'] if l['id']==action['landmark_id']]
                                if not matches:raise ValueError('Unknown public landmark_id')
                                xy=matches[0]['xy_m'];h=bearing(self.position,xy,self.heading)
                            self.move(xy,h,'explore_'+mode)
                            break
                        view=self.env.view(action['view_id'],self.allowed)
                        candidate=self.env.locate(view,action['bbox'])
                        self.event('candidate',candidate=candidate,target=action['target'],reason=action['reason'])
                        self.review(action,self.main_parent(view),candidate)
                        break
                    except ValueError as exc:
                        feedback.append(dict(error=str(exc),rejected_action=action))
                        self.event('invalid_action',action=action,error=str(exc))
                        # Invalid choices consume the same global 48-request allowance.
                        # Do not add an earlier movement-specific termination rule.
                if self.termination:break
            if self.termination is None:self.termination='step_limit'
        except Halt as exc:
            self.termination=str(exc)
        self.persist()
        result=dict(episode_id=self.episode['episode_id'],termination=self.termination,
                    completed=self.termination in ('confirmed_final','give_up','step_limit','model_request_limit',
                        'invalid_action_limit','invalid_review_limit','interface_error'),
                    final=self.final,actual_stop_xy_m=self.position,model_requests=self.model.cursor,
                    rounds=self.round+1,observed_view_count=len(self.allowed),
                    tool_capabilities=self.tool_status,observation_grounder=self.observation_grounder,
                    observation_query_mode=self.observation_query_mode,
                    tool_calls=[dict(backend=e['input']['tool'],status=e['output']['status'])
                                for e in self.events if e['kind']=='tool'],
                    input_sha256=digest(self.episode))
        save_json(self.output/'result.json',result)
        return result
