"""Uncertain search-area proposal from public text, never a stopping coordinate."""
import math
from .semantic_plan import validate_plan
from .initialization.prior import estimate_bearing


def search_region(initial_heading, plan, statistics):
    legs=validate_plan(plan,allow_qualitative=True)['legs']
    if not legs:return None
    positional=statistics.get('positional_priors',{}).get(str(len(legs)),{})
    distances=positional.get('distance_medians_m',[statistics['distance_median_m']]*len(legs))
    turns=positional.get('turn_medians_deg',[statistics['angle_median_deg']]*(len(legs)-1))
    prior=estimate_bearing(initial_heading,legs,distances,turns)
    heading=initial_heading;center=[0.,0.]
    for i,leg in enumerate(legs):
        distance=leg['distance_m'] if leg['distance_m'] is not None else distances[i]
        center[0]+=distance*math.sin(math.radians(heading))
        center[1]+=distance*math.cos(math.radians(heading))
        if i<len(legs)-1 and leg['turn_direction']:
            turn=leg['turn_deg'] if leg['turn_deg'] is not None else turns[i]
            heading=(heading+turn*(1 if leg['turn_direction']=='right' else -1))%360
    return dict(search_center_xy_m=center,bearing_deg=prior['direction'],estimated_final_heading_deg=heading,
        uncertain=True,imputed=prior['imputed'],source=prior['source'],
        warning='Statistical search proposal, not an exact endpoint. Never STOP at this coordinate without observed target grounding.')


def spiral_offsets(spacing):
    # A connected square spiral: adjacent observations remain overlapping.
    x=y=0;result=[]
    for length in range(1,5):
        for dx,dy in ([(1,0),(0,1)] if length%2 else [(-1,0),(0,-1)]):
            for _ in range(length):
                x+=dx;y+=dy;result.append([x*spacing,y*spacing])
    return result
