"""Use an uncertain search proposal without converting it into exact route truth."""
import math
from AVDN.navigation import Agent as BaseAgent
from AVDN.tools.geometry import bearing
from .ray_navigation import RayAgent
from .search_prior import spiral_offsets


class PriorSearchAgent(RayAgent):
    def __init__(self,*args,search_prior=None,**kwargs):
        super().__init__(*args,**kwargs)
        self.search_prior=search_prior if self.language_route is None and self.ray is None else None
        self.prior_area_reached=False;self.prior_scan_index=0;self.prior_review_stalls=0
        if self.search_prior:
            self.heading=self.search_prior['bearing_deg'] if self.search_prior['bearing_deg'] is not None else self.heading
            self.trajectory[0]['heading_deg']=self.heading

    def context(self,mode,feedback,tools,candidate=None):
        value=super().context(mode,feedback,tools,candidate)
        if self.search_prior:
            value['uncertain_search_prior']=dict(self.search_prior,area_reached=self.prior_area_reached,
                scan_index=self.prior_scan_index,policy='Approach the statistical search area, then inspect overlapping nearby views. This prior is not proof of target identity. Locate the FINAL described object from RGB; do not reenact the first distance repeatedly.')
            value['navigation_contract']['exploration']='Search around the uncertain language/training-text proposal. The movement controller covers neighboring views; identify the final target visually.'
            if mode=='review' and candidate and candidate.get('target')=='final':
                value['verification_target']=dict(role='FINAL destination only',description=self.observation_target(),
                    instruction='An earlier auxiliary landmark is not the destination. The statistical search prior does not certify this object; compare its visible attributes and relations with the final description.')
        return value

    def advance_search(self,source):
        center=self.search_prior['search_center_xy_m'];maximum=.7*self.env.spec['main_height_m']
        if not self.prior_area_reached:
            target=center
        else:
            offsets=spiral_offsets(.5*self.env.spec['main_height_m'])
            offset=offsets[self.prior_scan_index%len(offsets)]
            target=[center[i]+offset[i] for i in (0,1)]
        distance=math.dist(self.position,target)
        xy=target[:] if distance<=maximum else [self.position[i]+(target[i]-self.position[i])*maximum/distance for i in (0,1)]
        if distance<=maximum:
            if self.prior_area_reached:self.prior_scan_index+=1
            else:self.prior_area_reached=True
        heading=self.search_prior['estimated_final_heading_deg'] if self.prior_area_reached else bearing(self.position,target,self.heading)
        self.prior_review_stalls=0
        self.event('uncertain_prior_search',target_xy_m=target,next_xy_m=xy,area_reached=self.prior_area_reached,
                   note='Search motion only; this location never authorizes STOP')
        return BaseAgent.move(self,xy,heading,source)

    def move(self,xy,heading,source):
        if getattr(self,'search_prior',None) and source.startswith('explore'):
            return self.advance_search(source)
        return super().move(xy,heading,source)

    def auxiliary_confirmed(self,refined,reason):
        if not self.search_prior:return super().auxiliary_confirmed(refined,reason)
        self.memory['confirmed_auxiliary'].append(dict(refined,reason=reason))
        return self.advance_search('prior_auxiliary_reobserve')

    def review(self,proposal,main,candidate):
        if not self.search_prior:return super().review(proposal,main,candidate)
        if not self.prior_area_reached:
            self.event('prior_early_candidate_deferred',candidate=candidate,
                       reason='Obtain a view near the uncertain search area before final verification')
            return self.advance_search('prior_approach')
        before=self.position[:]
        BaseAgent.review(self,proposal,main,candidate)
        if self.termination is None and math.dist(before,self.position)<1e-6:
            self.prior_review_stalls+=1
            if self.prior_review_stalls>=2:self.advance_search('prior_review_recovery')

    def final_confirmed(self,refined,reason,view):
        if not self.search_prior:return super().final_confirmed(refined,reason,view)
        # STOP uses an observed object, never the statistical center.
        BaseAgent.final_confirmed(self,refined,reason,view)
        BaseAgent.move(self,refined['center_xy_m'],self.heading,'observed_target_stop')
        stopped=self.env.main(self.position,self.heading,f'r{self.round:02d}-stop')
        self.event('stop',xy_m=self.position,heading_deg=self.heading,view_id=stopped['view_id'],
                   basis='Current visual confirmation and observed object center; not imputed route endpoint')
