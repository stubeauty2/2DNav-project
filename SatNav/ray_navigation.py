"""Search a language-defined final ray through the existing observation/tool loop."""
import math
import re
from AVDN.navigation import Agent as BaseAgent
from AVDN.tools.geometry import bearing, normalized
from .navigation import Agent
from .route_geometry import final_ray, project_ray


def stopping_side(text):
    matches=re.findall(r'\b(?:upper|lower|middle|top|bottom|center|centre)[\s-]+(left|right)\b',text.lower())
    return matches[-1] if matches else None


def stopping_vertical(text):
    matches=re.findall(r'\b(upper|top|lower|bottom)\b',text.lower())
    return (1 if matches[-1] in ('lower','bottom') else -1) if matches else 0


class RayAgent(Agent):
    def __init__(self, *args, route_diagnostics=None, region_stop=False, **kwargs):
        super().__init__(*args, route_diagnostics=route_diagnostics, **kwargs)
        self.ray = final_ray(route_diagnostics) if self.language_route is None else None
        self.ray_ready = False
        self.ray_side = stopping_side((route_diagnostics or {}).get('stop_condition',''))
        self.ray_rejections = 0
        self.review_stalls = 0
        self.region_stop = region_stop
        self.ray_vertical = stopping_vertical((route_diagnostics or {}).get('stop_condition',''))

    def context(self, mode, feedback, tools, candidate=None):
        context = super().context(mode, feedback, tools, candidate)
        if self.ray:
            context['partial_language_route'] = dict(self.ray, prefix_reached=self.ray_ready,
                instruction='Final distance is unknown. Approach the known prefix endpoint, then search forward along the final ray with overlapping views. Select the final landmark only from RGB. An early similar object before the prefix is provisional; do not fly to auxiliary object centers. Exploration actions follow this public-language corridor. Geometry alone never confirms identity.')
            context['partial_language_route']['required_target_side']=self.ray_side
            context['partial_language_route']['candidate_rejection']='A rejected candidate does not mean the entire view is empty. Inspect other objects in the SAME view before exploring. Opposite-side landmarks cannot satisfy the final left/right placement by moving forward.'
            if candidate and candidate.get('center_xy_m'):
                context['partial_language_route']['candidate_geometry']=self.candidate_geometry(candidate)
            if mode=='review' and candidate and candidate.get('target')=='final':
                context['verification_target']=dict(role='FINAL destination only',description=self.observation_target(),
                    instruction='An auxiliary landmark named earlier in the instruction is not the final destination. Confirm this proposal only against the final description and final-side constraint.')
        return context

    def ground(self, view, name, query, tool_index):
        result,overlay=super().ground(view,name,query,tool_index)
        if not self.ray or not self.ray_ready or result.get('status')!='ok':
            return result,overlay
        valid=[]
        for candidate in result['candidates']:
            geometry=self.candidate_geometry(self.env.locate(view,candidate['bbox']))
            enriched=dict(candidate,partial_route_geometry=geometry)
            if geometry['side_conflict'] or geometry['distance_to_ray_m']>30.:
                result['rejected_candidates'].append(dict(enriched,reason='Inconsistent with final language ray or required side; not a visual category rejection'))
            else:
                valid.append(enriched)
        result['candidates']=valid
        result['note']='Only geometric consistency was checked, not identity. Empty compatible proposals do not prove target absence. Inspect RGB or move for a better view.'
        overlay=self.env.tool_overlay(view,valid,f'r{self.round:02d}-{name}{tool_index}-ray') if valid else None
        return result,overlay

    def advance_ray(self, source):
        self.ray_rejections = 0
        self.review_stalls = 0
        step = .5*self.env.spec['main_height_m']
        anchor = self.ray['anchor_xy_m']
        if not self.ray_ready:
            distance = math.dist(self.position, anchor)
            xy = anchor[:] if distance <= step else [self.position[i]+(anchor[i]-self.position[i])*step/distance for i in (0, 1)]
            heading = bearing(self.position, anchor, self.heading)
            if distance <= step:
                self.ray_ready = True
                heading = self.ray['heading_deg']
        else:
            projected, _ = project_ray(self.ray, self.position)
            xy = [projected[i]+step*self.ray['axis'][i] for i in (0, 1)]
            heading = self.ray['heading_deg']
        self.event('partial_route_advance', next_xy_m=xy, prefix_reached=self.ray_ready,
                   source=self.ray['source'])
        return BaseAgent.move(self, xy, heading, source)

    def move(self, xy, heading, source):
        if getattr(self, 'ray', None) and source.startswith('explore'):
            return self.advance_ray(source)
        return super().move(xy, heading, source)

    def auxiliary_confirmed(self, refined, reason):
        if not self.ray:
            return super().auxiliary_confirmed(refined, reason)
        self.memory['confirmed_auxiliary'].append(dict(refined, reason=reason))
        return self.advance_ray('partial_route_auxiliary')

    def review(self, proposal, main, candidate):
        if not self.ray:
            return super().review(proposal, main, candidate)
        if not self.ray_ready:
            self.event('partial_prefix_candidate_deferred',candidate=candidate,
                       reason='Known prefix not reached; continue to the final search corridor and require a new visual review there')
            return self.advance_ray('partial_prefix_reobserve')
        self.ray_main = main
        before=self.position[:]
        BaseAgent.review(self, proposal, main, candidate)
        if self.termination is None and math.dist(before,self.position)<1e-6:
            self.review_stalls+=1
            if self.review_stalls>=2:
                self.event('partial_review_stall_recovery',
                           reason='Two unsuccessful reviews without movement; inspect the next overlapping view')
                self.advance_ray('partial_review_recovery')
        else:
            self.review_stalls=0

    def candidate_geometry(self, refined):
        target,distance=project_ray(self.ray,refined['center_xy_m'])
        dx,dy=(refined['center_xy_m'][i]-self.ray['anchor_xy_m'][i] for i in (0,1))
        right=dx*self.ray['axis'][1]-dy*self.ray['axis'][0]
        side_conflict=(self.ray_side=='left' and right>5.) or (self.ray_side=='right' and right < -5.)
        return dict(projected_stop_xy_m=target,distance_to_ray_m=distance,right_of_final_course_m=right,
                    side_conflict=side_conflict)

    def stopping_position(self, center):
        target,_=project_ray(self.ray,center)
        if self.region_stop:
            # Center of the upper/lower third in the public 3x3 image-region
            # convention. Fixed from observation scale, never fitted to goals.
            shift=self.ray_vertical*self.env.spec['main_height_m']/3.
            target=[target[i]+shift*self.ray['axis'][i] for i in (0,1)]
            target,_=project_ray(self.ray,target)
        return target

    def final_confirmed(self, refined, reason, view):
        if not self.ray:
            return super().final_confirmed(refined, reason, view)
        geometry=self.candidate_geometry(refined)
        target,distance=self.stopping_position(refined['center_xy_m']),geometry['distance_to_ray_m']
        # Public generation constraint, Appendix C.2: decision poses are within
        # 30m of their landmark. This is not an episode evaluator radius input.
        conflict = distance > 30. or geometry['side_conflict']
        in_view = all(0 <= v <= 1000 for v in normalized(self.ray_main, [target])[0])
        if not self.ray_ready or conflict or not in_view:
            self.memory['candidates'].append(dict(refined, target='final', status='reject' if conflict and self.ray_ready else 'deferred',
                geometry=geometry,reason='Require the final instruction corridor and side; inspect OTHER candidates in the same view after a geometry rejection.'))
            self.event('partial_route_deferred', distance_to_ray_m=distance,
                       prefix_reached=self.ray_ready, projected_stop_in_view=in_view,side_conflict=geometry['side_conflict'])
            if self.ray_ready and conflict:
                self.ray_rejections+=1
                if self.ray_rejections<=2:
                    return
            if self.region_stop and self.ray_ready and not conflict and not in_view:
                distance=math.dist(self.position,target)
                step=min(.5*self.env.spec['main_height_m'],distance)
                xy=[self.position[i]+(target[i]-self.position[i])*step/distance for i in (0,1)]
                self.ray_rejections=0;self.review_stalls=0
                self.event('partial_region_alignment',target_xy_m=target,next_xy_m=xy,
                           reason='Stopping camera outside this view; align then require a fresh visual review')
                BaseAgent.move(self,xy,self.ray['heading_deg'],'partial_region_reobserve')
                return
            self.advance_ray('partial_route_reobserve')
            return
        BaseAgent.final_confirmed(self, refined, reason, view)
        BaseAgent.move(self, target, self.ray['heading_deg'], 'partial_route_stop')
        final_view = self.env.main(self.position, self.heading, f'r{self.round:02d}-stop')
        self.event('stop', xy_m=self.position, heading_deg=self.heading, view_id=final_view['view_id'],
                   basis='Visually confirmed landmark and final language ray',region_stop=self.region_stop)
