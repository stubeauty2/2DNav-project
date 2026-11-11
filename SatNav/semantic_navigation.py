"""Stage-specific grounding and image-region alignment in the shared tool loop."""
import math
from AVDN.navigation import Agent as BaseAgent
from .navigation import Agent
from .semantic_plan import validate_plan, REGIONS, aligned_heading


class SemanticAgent(Agent):
    def __init__(self, *args, semantic_plan=None, **kwargs):
        super().__init__(*args, **kwargs)
        plan=validate_plan(semantic_plan) if semantic_plan is not None else None
        self.legs=plan['legs'] if plan and self.language_route is None else []
        self.phase=0
        if self.legs:
            self.heading=self.episode['agent_input']['initial_pose']['heading_deg']
            self.trajectory[0]['heading_deg']=self.heading
        self.phase_origin=self.position[:]
        self.phase_heading=self.heading
        self.local_search_steps=0
        self.phase_distance_reached=False

    def observation_target(self):
        return self.legs[self.phase]['landmark'] if self.legs else super().observation_target()

    def observation_cache_key(self, main):
        return (main['view_id'],self.phase,self.observation_target()) if self.legs else super().observation_cache_key(main)

    def context(self, mode, feedback, tools, candidate=None):
        value=super().context(mode,feedback,tools,candidate)
        if self.legs:
            value['semantic_route']=dict(phase_index=self.phase,phase_count=len(self.legs),
                current_leg=self.legs[self.phase],phase_start_xy_m=self.phase_origin,
                phase_heading_deg=self.phase_heading,completed_legs=self.legs[:self.phase],
                current_target_type='final' if self.phase==len(self.legs)-1 else 'auxiliary',
                instruction='Ground and review the CURRENT leg landmark. Future destination sightings do not complete earlier legs. Follow forward travel until its cue is confirmed; image-region alignment and specified turns are then applied from observed geometry. Do not fly to an auxiliary object center by default.')
            value['semantic_route']['metric_endpoint_xy_m']=self.metric_endpoint()
            value['semantic_route']['explicit_distance_reached']=self.phase_distance_reached
            value['semantic_route']['forward_constraint']='Before a turn, travel follows the phase forward axis. A left/right arrival cue must agree with the landmark side of that axis; lateral movement cannot manufacture a matching cue.'
            value['semantic_route']['local_search_radius_m']=.5*self.env.spec['main_height_m'] if self.metric_endpoint() is not None else None
            value['navigation_contract']['direction_role']='Follow the current leg heading; original first heading is authoritative for the first traversal. The composed final-direction prior does not replace this stage plan.'
            value['navigation_contract']['exploration']='Ground the active leg landmark; complete its arrival and turn conditions before advancing to the next leg.'
            value['navigation_contract']['unresolved']='Unknown final bearing does not erase the observed original heading or the ordered public instruction phases.'
        return value

    def metric_endpoint(self):
        distance=self.legs[self.phase]['distance_m']
        if distance is None:return None
        angle=math.radians(self.phase_heading)
        return [self.phase_origin[0]+distance*math.sin(angle),
                self.phase_origin[1]+distance*math.cos(angle)]

    def phase_coordinates(self, point):
        angle=math.radians(self.phase_heading)
        dx,dy=point[0]-self.phase_origin[0],point[1]-self.phase_origin[1]
        return dx*math.sin(angle)+dy*math.cos(angle), dx*math.cos(angle)-dy*math.sin(angle)

    def candidate_conflict(self, refined):
        region=self.legs[self.phase]['arrival_region']
        if region is None:return None
        _,right=self.phase_coordinates(refined['center_xy_m'])
        tolerance=.05*self.env.spec['main_width_m']
        side=region.split('_')[1]
        if (side=='right' and right < -tolerance) or (side=='left' and right > tolerance):
            return f'Landmark is {right:.1f}m right of the forward axis, inconsistent with the {side} arrival cue. Do not shift sideways to force the cue; inspect another candidate.'
        return None

    def accept_phase_candidate(self, refined):
        conflict=self.candidate_conflict(refined)
        if conflict is None:return True
        self.memory['candidates'].append(dict(refined,status='reject',phase=self.phase,reason=conflict))
        self.event('semantic_geometry_reject',phase=self.phase,reason=conflict)
        return False

    def bounded_local_search(self, proposed, endpoint):
        radius=.5*self.env.spec['main_height_m']
        dx,dy=proposed[0]-endpoint[0],proposed[1]-endpoint[1]
        distance=math.hypot(dx,dy)
        result=proposed if distance<=radius else [endpoint[0]+dx*radius/distance,endpoint[1]+dy*radius/distance]
        if math.dist(result,self.position)<5.:
            angle=math.radians(self.phase_heading+90*(self.local_search_steps%4))
            result=[endpoint[0]+.5*radius*math.sin(angle),endpoint[1]+.5*radius*math.cos(angle)]
            self.local_search_steps+=1
        step=math.dist(self.position,result)
        if step>radius:
            result=[self.position[i]+(result[i]-self.position[i])*radius/step for i in (0,1)]
        return result

    def move(self, xy, heading, source):
        if getattr(self,'legs',None) and source.startswith('explore'):
            endpoint=self.metric_endpoint()
            angle=math.radians(self.phase_heading)
            axis=[math.sin(angle),math.cos(angle)]
            travelled=sum((self.position[i]-self.phase_origin[i])*axis[i] for i in (0,1))
            remaining=None if endpoint is None else self.legs[self.phase]['distance_m']-travelled
            if remaining is not None and remaining<=5.:self.phase_distance_reached=True
            if remaining is None or (remaining>5. and not self.phase_distance_reached):
                proposed=math.dist(self.position,xy)
                step=min(max(proposed,.2*self.env.spec['main_height_m']),.5*self.env.spec['main_height_m'])
                if remaining is not None:step=min(step,remaining)
                xy=[self.position[i]+step*axis[i] for i in (0,1)]
                heading=self.phase_heading
                self.event('semantic_forward',phase=self.phase,next_xy_m=xy,
                           reason='Bounded forward scan along active instruction leg')
            elif endpoint is not None:
                xy=self.bounded_local_search(xy,endpoint)
                heading=self.phase_heading
                self.event('semantic_local_search',phase=self.phase,next_xy_m=xy,
                           endpoint_xy_m=endpoint,reason='Search locally after the explicit traversal distance; do not keep flying beyond it')
        return super().move(xy,heading,source)

    def review(self, proposal, main, candidate):
        if not self.legs:return super().review(proposal,main,candidate)
        self._phase_main=main
        target='final' if self.phase==len(self.legs)-1 else 'auxiliary'
        if proposal['target']!=target:
            self.event('semantic_target_scope',proposed=proposal['target'],required=target,phase=self.phase)
        return BaseAgent.review(self,dict(proposal,target=target),main,candidate)

    def desired_camera(self, refined, final=False):
        endpoint=self.metric_endpoint()
        if endpoint is not None:return endpoint
        region=self.legs[self.phase]['arrival_region']
        if region is None:return self.position[:]
        forward,right=self.phase_coordinates(refined['center_xy_m'])
        ahead=(500-REGIONS[region][1])/1000*self.env.spec['main_height_m']
        if region=='middle_center' and self.legs[self.phase]['align_region'] is not None and abs(right)<5.:
            ahead=10. # Keep an observed ray for the following rotation.
        travel=max(0.,forward-ahead)
        angle=math.radians(self.phase_heading)
        return [self.phase_origin[0]+travel*math.sin(angle),self.phase_origin[1]+travel*math.cos(angle)]

    def approach_camera(self, target, source):
        distance=math.dist(self.position,target)
        maximum=.5*self.env.spec['main_height_m']
        step=min(distance,maximum)
        if distance>1e-6:
            xy=[self.position[i]+(target[i]-self.position[i])*step/distance for i in (0,1)]
            self.move(xy,self.phase_heading,source)
        return distance<=maximum+1e-6

    def auxiliary_confirmed(self, refined, reason):
        if not self.legs:return super().auxiliary_confirmed(refined,reason)
        if not self.accept_phase_candidate(refined):return
        leg=self.legs[self.phase]
        target=self.desired_camera(refined)
        if not self.approach_camera(target,'semantic_arrival_alignment'):
            self.event('semantic_arrival_pending',phase=self.phase,target_xy_m=target)
            return
        heading=self.phase_heading
        if leg['turn_deg'] is not None:
            heading=(heading+leg['turn_deg']*(1 if leg['turn_direction']=='right' else -1))%360
        elif leg['align_region'] is not None:
            heading=aligned_heading(self.position,refined['center_xy_m'],leg['align_region'])
        self.memory['confirmed_auxiliary'].append(dict(refined,reason=reason,phase=self.phase))
        self.event('semantic_phase_complete',phase=self.phase,observed_landmark=refined['center_xy_m'],next_heading_deg=heading)
        self.phase+=1
        self.local_search_steps=0
        self.phase_distance_reached=False
        self.phase_origin=self.position[:];self.phase_heading=heading
        self.move(self.position[:],heading,'semantic_turn')

    def final_confirmed(self, refined, reason, view):
        if not self.legs:return super().final_confirmed(refined,reason,view)
        if not self.accept_phase_candidate(refined):return
        target=self.desired_camera(refined,final=True)
        if math.dist(self.position,target)>5.:
            self.final=None;self.termination=None
            self.approach_camera(target,'semantic_final_reobserve')
            self.event('semantic_stop_deferred',target_xy_m=target,
                       reason='Fresh visual review required after stopping-position alignment')
            return
        BaseAgent.final_confirmed(self,refined,reason,view)
        self.move(target,self.phase_heading,'semantic_language_stop')
        final_view=self.env.main(self.position,self.heading,f'r{self.round:02d}-stop')
        self.event('stop',xy_m=self.position,heading_deg=self.heading,view_id=final_view['view_id'],
                   basis='Completed public instruction phases and fresh visual confirmation')
