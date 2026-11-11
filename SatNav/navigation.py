"""Reuse exploration/tool/review loop and add a language-conditioned stop pose."""
from AVDN.navigation import Agent as BaseAgent
from AVDN.tools.geometry import bearing, offset
from AVDN.tools.llm import Halt
from AVDN.tools.geometry import normalized
from .route_geometry import numeric_route, guidance
import math


class Agent(BaseAgent):
    def __init__(self, *args, route_diagnostics=None, route_guard=False, route_stop_gate=False, bounded_exploration=False, **kwargs):
        self.language_route = numeric_route(route_diagnostics)
        self.route_guard = route_guard
        self.route_stop_gate = route_stop_gate
        self.bounded_exploration = bounded_exploration
        self.route_area_visited = False
        super().__init__(*args, **kwargs)

    def move(self, xy, heading, source):
        if self.bounded_exploration and self.language_route is None and source.startswith('explore'):
            distance=math.dist(self.position,xy)
            maximum=.5*self.env.spec['main_height_m']
            if distance>maximum:
                proposed=xy[:]
                xy=[self.position[i]+(xy[i]-self.position[i])*maximum/distance for i in (0,1)]
                self.event('exploration_stride_clipped',proposed_xy_m=proposed,executed_xy_m=xy,
                           reason='Overlapping observations; a long instruction leg is not one camera hop')
        if self.route_guard and self.language_route is not None and source.startswith('explore') and not self.route_area_visited:
            endpoint=self.language_route['endpoint_xy_m']
            distance=math.dist(self.position,endpoint)
            if distance<=1.:
                self.route_area_visited=True
            else:
                step=math.dist(self.position,xy)
                after=math.dist(xy,endpoint)
                maximum=min(.7*self.env.spec['main_height_m'],distance)
                if after>=distance or step>maximum+1e-6:
                    original=dict(xy_m=xy,heading_deg=heading,source=source)
                    heading=bearing(self.position,endpoint,self.heading)
                    xy=[self.position[i]+(endpoint[i]-self.position[i])*maximum/distance for i in (0,1)]
                    self.event('language_motion_correction',proposed=original,executed_xy_m=xy,
                               reason='Approach complete numeric-language endpoint with overlapping views; never use GT')
                if math.dist(xy,endpoint)<=1.:
                    self.route_area_visited=True
        return super().move(xy,heading,source)

    def context(self, mode, feedback, tools, candidate=None):
        context = super().context(mode, feedback, tools, candidate)
        context['navigation_contract'] = dict(
            direction_role='initialization.direction estimates start-to-final displacement; NOT the first narrated route segment',
            original_first_heading_deg=self.episode['agent_input']['initial_pose']['heading_deg'],
            exploration='Use the final-direction prior to search for the final target. Choose movement distances for observing new areas; do not attach original first-segment distance to a different composed heading.',
            unresolved='If direction is null, current heading is only a fallback exploration heading, not a known final bearing.',
            stop='Target identity confirmation is followed by a separate language-conditioned physical stopping action.')
        if self.bounded_exploration and self.language_route is None:
            context['navigation_contract']['max_exploration_step_m']=.5*self.env.spec['main_height_m']
            context['navigation_contract']['distance_progress']='Original instruction distances describe whole route segments, not a fresh distance to repeat every round. Follow executed poses and inspect overlapping views; actual exploration hops are capped at half a footprint.'
        if self.language_route is not None:
            context['language_route_guidance'] = guidance(self.language_route,self.position,self.heading)
            context['language_route_guidance']['motion_guard_enabled']=self.route_guard
            context['language_route_guidance']['predicted_area_visited']=self.route_area_visited
            if self.route_stop_gate:
                context['language_route_guidance']['stop_verification']='If the language stopping area is outside the current view, a confirmed candidate remains provisional. Approach with a bounded move and inspect the next view before stopping.'
        return context

    def review(self, proposal, main, candidate):
        super().review(proposal, main, candidate)
        if self.termination != 'confirmed_final':
            return
        # Base loop confirms identity, not a physical stopping location.
        self.termination = None
        if self.language_route is not None:
            endpoint = self.language_route['endpoint_xy_m']
            image_point = normalized(main,[endpoint])[0]
            if all(0<=v<=1000 for v in image_point):
                # Only use an observed map location after visual identity review.
                xy = self.env.check_point(main,image_point)
                self.event('stop_pose_from_language', endpoint_xy_m=xy,
                           source=self.language_route['source'], image_point=image_point)
                self.move(xy,self.language_route['final_heading_deg'],'numeric_language_stop')
                final_view=self.env.main(self.position,self.heading,f'r{self.round:02d}-stop')
                self.termination='confirmed_final'
                self.event('stop',xy_m=self.position,heading_deg=self.heading,
                           view_id=final_view['view_id'],basis='complete numeric language route plus confirmed visual target')
                return
            if self.route_stop_gate:
                # An early visual match must not erase contradictory route evidence.
                # Move only one overlapping-view step, then let the ordinary loop
                # ground and review again. Never STOP solely on a predicted coordinate.
                provisional = self.final
                self.final = None
                self.memory['candidates'].append(dict(candidate, target='final', status='deferred',
                    reason='Visual match is provisional: language stopping area is outside current view. Reobserve after approaching.'))
                distance = math.dist(self.position, endpoint)
                step = min(.7*self.env.spec['main_height_m'], distance)
                heading = bearing(self.position, endpoint, self.heading)
                xy = [self.position[i]+(endpoint[i]-self.position[i])*step/distance for i in (0,1)]
                self.event('stop_deferred_for_route', provisional=provisional,
                           endpoint_xy_m=endpoint, next_xy_m=xy,
                           reason='Require fresh visual confirmation near language stopping area')
                self.move(xy, heading, 'language_stop_reobserve')
                return
        feedback = [dict(instruction=(
            'Final target identity has been confirmed. Now select the UAV stopping '
            'position from the original instruction, including its final image-region '
            'condition. Do not automatically fly to the object center. Return an '
            'explore action in point mode for the stopping position within this main '
            'view, or offset mode for a position outside it. This move is followed by '
            'STOP. To stop at the current position use point [500,500] in this view.'),
            confirmed_target=self.final)]
        for _ in range(3):
            action = self.ask('decision', [main], 0, feedback=feedback)
            self.event('stop_pose_proposal', action=action)
            try:
                if action['action'] != 'explore' or action.get('mode') not in ('point', 'offset'):
                    raise ValueError('Stopping pose requires explore point or offset')
                if action['mode'] == 'point':
                    if action['view_id'] != main['view_id']:
                        raise ValueError('Use the supplied main view')
                    xy = self.env.check_point(main, action['point'])
                    # Image-region condition is interpreted in current camera heading.
                    heading = self.heading
                else:
                    xy, heading = offset(self.position, self.heading, action['turn_deg'],
                                         action['step_scale'], main['height_m'])
                self.move(xy, heading, 'language_conditioned_stop')
                final_view = self.env.main(self.position, self.heading, f'r{self.round:02d}-stop')
                self.termination = 'confirmed_final'
                self.event('stop', xy_m=self.position, heading_deg=self.heading,
                           view_id=final_view['view_id'], basis='language and confirmed visual target only')
                return
            except ValueError as exc:
                feedback.append(dict(error=str(exc)))
        raise Halt('invalid_stop_pose_limit')
