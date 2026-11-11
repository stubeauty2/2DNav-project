"""Explicit stage policy; preserve base prompts for other datasets/variants."""
from AVDN.agent.prompts import SYSTEM, REVIEW_SYSTEM

SEMANTIC_SYSTEM = SYSTEM.replace(
    'Choose observations and tools yourself; the narrated route is evidence for identifying\n'
    'the destination, not a compulsory sequence of movements to reenact.',
    'Execute the ordered public semantic_route. Its current leg defines the object to\n'
    'ground now; completing earlier landmarks and turns is required before the final leg.')
SEMANTIC_SYSTEM = SEMANTIC_SYSTEM.replace(
    'DECISION: inspect the current RGB for plausible final targets before moving away.',
    'DECISION: inspect RGB for semantic_route.current_leg.landmark. Use target auxiliary\n'
    'until semantic_route.current_target_type is final. Future destinations do not\n'
    'complete an earlier leg. Propose the matching current landmark when visible.')
SEMANTIC_SYSTEM = SEMANTIC_SYSTEM.replace(
    'inspect the new view, and update the prior. No fixed stage order is required.',
    'inspect the new view, and follow the active leg heading and stage order.')
SEMANTIC_SYSTEM = SEMANTIC_SYSTEM.replace(
    'A destination already identifiable in the current view can be located immediately.',
    'The final destination may be proposed only when the final leg is active.')
SEMANTIC_SYSTEM += '''
The controller handles stopping-position/turn alignment from a visually verified
landmark. Arrival region and post-turn region are not appearance attributes. An
object not yet in its requested region can be proposed when its identity is clear;
the controller may move FORWARD along the phase axis and require another review.
Left/right arrival cues must match the object's side of that forward axis. An
opposite-side object is a contradiction, not a reason to shift the flight path
sideways. Compare alternative landmarks on the instructed side. Do not ignore
semantic_geometry_reject feedback by proposing the same object again.
Do not claim phase completion yourself. Read the current phase from semantic_route.
Continue forward along phase_heading_deg with overlapping observations when the
current landmark is not visible. Once an explicit leg distance is reached, inspect
the local area rather than repeating that full distance from the new pose. Use
metric_endpoint_xy_m and local_search_radius_m as the bounded search area.
'''
SEMANTIC_REVIEW_SYSTEM = REVIEW_SYSTEM.replace(
    'or moves to an auxiliary center; no fallback box is used when rejecting or uncertain.',
    'or aligns the view/turn for a verified intermediate landmark; no fallback box is used when rejecting or uncertain.')
SEMANTIC_REVIEW_SYSTEM += '''
For this request, verify ONLY semantic_route.current_leg.landmark. The target type
is semantic_route.current_target_type. A correct future target is not a match to
the current landmark. Use the full instruction for disambiguating appearance and
relationships, but do not demand completion of later legs before confirming an
intermediate object. Region alignment occurs after identity review; a different
current vertical image region alone does not mean the object identity is wrong.
However, left/right disagreement with the active forward course is a spatial
contradiction: forward travel cannot move an object across that flight axis.
'''
