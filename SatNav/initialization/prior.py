"""Uncertain bearing prior from public instruction legs and training text statistics.

This estimates a SEARCH BEARING only. Imputed lengths/turns must never replace
the explicit-only route diagnostics or become a precise stopping coordinate.
"""
import math


def estimate_bearing(initial_heading, legs, distance_prior_m, turn_prior_deg):
    if not legs:raise ValueError('Public semantic legs required')
    distances=list(distance_prior_m) if isinstance(distance_prior_m,(list,tuple)) else [distance_prior_m]*len(legs)
    turns=list(turn_prior_deg) if isinstance(turn_prior_deg,(list,tuple)) else [turn_prior_deg]*(len(legs)-1)
    if len(distances)!=len(legs) or len(turns)!=len(legs)-1:
        raise ValueError('Prior count must match public leg count')
    for value in [initial_heading]+distances+turns:
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
            raise ValueError('Finite numeric prior parameters required')
    if any(v<=0 for v in distances) or any(not 0<v<=180 for v in turns):
        raise ValueError('Invalid prior range')
    heading=initial_heading
    east=north=0.
    imputed=[]
    for index,leg in enumerate(legs):
        distance=leg['distance_m']
        if distance is None:
            distance=distances[index]
            imputed.append(dict(leg=index,field='distance_m',value=distance))
        east+=distance*math.sin(math.radians(heading))
        north+=distance*math.cos(math.radians(heading))
        if index<len(legs)-1 and leg['turn_direction'] is not None:
            turn=leg['turn_deg']
            if turn is None:
                turn=turns[index]
                imputed.append(dict(leg=index,field='turn_deg',value=turn))
            heading=(heading+turn*(1 if leg['turn_direction']=='right' else -1))%360
    direction=math.degrees(math.atan2(east,north))%360 if math.hypot(east,north)>1e-6 else None
    return dict(direction=direction,imputed=imputed,
        source='training instruction text statistics and public episode language; NOT ground truth',
        uncertain=True,warning='Search direction only; no precise stopping coordinate or guaranteed 90-degree bound')
