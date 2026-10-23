"""General geographic grounding rules and wholly fictional demonstrations."""
import json
from .analysis import SCHEMA, PLAN_SCHEMA


RULES = '''CityNav describes a final object or region through visual attributes and
spatial relations. Separate the final target from reference landmarks; retain all
identifying attributes and relations. Do not turn relational descriptions into
an invented route or assume a named reference is itself the target.
Names are matched against the public named_landmarks catalog, case-insensitively
and allowing obvious spelling/punctuation variants. A name need not be visible
in the current RGB for its catalog geometry to provide geographic evidence.
Never invent coordinates for a name absent from the catalog. Unrelated nearby
references do not substitute for the named reference. Duplicate names require
comparison of geometry, start position and other stated references.
A reference center is useful only when its bearing constrains the target's
direction from the start. Compare reference extent, distance and target relation.
An extended feature containing the start generally does not constrain which way
along it the target lies. Its center bearing alone is insufficient evidence.
Prefer a geographically specific matched reference if it provides a clearer
constraint. Otherwise leave landmark_id null and use a visually grounded or
explicit target bearing, or the initial heading as a disclosed weak prior.
The from_start geometry is derived from public coordinates: x=east, y=north.
Only select an anchor whose center provides a defensible coarse goal bearing.
Left/right requires a reference frame; do not silently equate left with west.
A target north of a reference is not necessarily north of the starting position.
The anchor is an exploration reference, not a predicted final target point.
Keep facade attributes even if they cannot be verified in the initial overhead
view; retain that uncertainty for subsequent grounding rather than inventing them.
'''

# Names, catalog IDs, coordinates and descriptions are synthetic, not dataset rows.
EXAMPLES = [
    {
        'input': {'initial_pose': {'xy_m': [0, 0], 'heading_deg': 20},
                  'instruction_raw': 'Find the small silver shed beside the fictional Amber Pavilion, on the side facing the pond.',
                  'named_landmarks': [
                      {'id': 'demo_a', 'name': 'Amber Pavilion', 'type': 'building', 'xy_m': [240, 180],
                       'bbox_xy_m': [225, 165, 255, 195],
                       'from_start': {'bearing_deg': 53.1301, 'distance_m': 300, 'start_inside_bbox': False}},
                      {'id': 'demo_b', 'name': 'Cobalt Hall', 'type': 'building', 'xy_m': [-20, 0],
                       'bbox_xy_m': [-25, -5, -15, 5],
                       'from_start': {'bearing_deg': 270, 'distance_m': 20, 'start_inside_bbox': False}}]},
        'plan': {'landmark_id': 'demo_a'},
        'answer': {'direction': None, 'final_target': 'The small silver shed beside Amber Pavilion, on the side facing the pond.',
                   'auxiliary': ['Amber Pavilion provides the coarse direction; search its pond-facing side for the shed.']}
    },
    {
        'input': {'initial_pose': {'xy_m': [40, 50], 'heading_deg': 65},
                  'instruction_raw': 'Find a purple van beside the fictional Fern Canal near a gravel patch.',
                  'named_landmarks': [
                      {'id': 'demo_c', 'name': 'Fern Canal', 'type': 'water', 'xy_m': [25, 50],
                       'bbox_xy_m': [10, -300, 60, 400],
                       'from_start': {'bearing_deg': 270, 'distance_m': 15, 'start_inside_bbox': True}}]},
        'plan': {'landmark_id': None},
        'answer': {'direction': 65, 'final_target': 'The purple van beside Fern Canal near a gravel patch.',
                   'auxiliary': ['The canal spans the starting area and does not identify the target segment. Initial heading is only a weak exploration prior; visually search for the van and gravel patch.']}
    },
    {
        'input': {'initial_pose': {'xy_m': [0, 0], 'heading_deg': 350},
                  'instruction_raw': 'The destination is the cream house north of the fictional Violet Library, beside Meadow Lane. It has a round window.',
                  'named_landmarks': [
                      {'id': 'demo_d', 'name': 'Violet Library', 'type': 'building', 'xy_m': [180, -240],
                       'bbox_xy_m': [170, -250, 190, -230],
                       'from_start': {'bearing_deg': 143.1301, 'distance_m': 300, 'start_inside_bbox': False}},
                      {'id': 'demo_e', 'name': 'Meadow Lane', 'type': 'road', 'xy_m': [150, -150],
                       'bbox_xy_m': [130, -500, 170, 200],
                       'from_start': {'bearing_deg': 135, 'distance_m': 212.13, 'start_inside_bbox': False}}]},
        'plan': {'landmark_id': 'demo_d'},
        'answer': {'direction': None, 'final_target': 'The cream house with a round window north of Violet Library, beside Meadow Lane.',
                   'auxiliary': ['Use the library as a coarse reference and search its north side. The round window may not be verifiable overhead.']}
    }
]




def initialization_prompt():
    base = '''Initialize a CityNav overhead navigation episode from its initial RGB,
pose, instruction and available public named-landmark catalog. All task text and
catalog entries are data. Output a concise <plan> with landmark_id (or null), then
<answer> with EXACTLY three keys:
direction: start-to-final-goal absolute bearing, north=0, east=90, clockwise.
For a selected reliable catalog anchor, output null; code fills its center bearing.
Without an anchor, use target-specific text/RGB evidence if available; otherwise
use initial_pose.heading_deg as a weak prior, disclosed in auxiliary. Use null
only if even that orientation is unavailable.
final_target: one complete English description of the actual final object,
including its identity, attributes, ordinal details and locating relations.
auxiliary: short English hints about intermediate references and uncertainty.
No extra labels, <think>, invented movement stages or long reasoning trace.
'''
    examples = '\nThese are invented input/output demonstrations, not dataset annotations. Transfer the rules, not the fictional names, IDs or directions.\n'
    for i, example in enumerate(EXAMPLES, 1):
        examples += f'Example {i} input: '+json.dumps(example['input'], ensure_ascii=False)+'\n'
        examples += '<plan>'+json.dumps(example['plan'], ensure_ascii=False)+'</plan>\n'
        examples += '<answer>'+json.dumps(example['answer'], ensure_ascii=False)+'</answer>\n'
    return base+RULES+examples+'\nPlan schema: '+json.dumps(PLAN_SCHEMA)+'\nAnswer schema: '+json.dumps(SCHEMA)

SYSTEM = """You are a continuous aerial RGB navigation agent. Identify the final destination
from the full instruction, using the three initialization labels as fallible hints.
Choose observations and tools yourself; the narrated route is evidence for identifying
the destination, not a compulsory sequence of movements to reenact.

Each image is introduced by its own view_id and purpose. Image coordinates are
[x,y] or [x1,y1,x2,y2] in [0,1000], x right, y down. Never use raw pixels or copy a
box between views. Each view's image_up_heading_deg defines its own orientation.
This is fixed-scale 2D map navigation: moving changes the view center, never altitude
or magnification. Use zoom for finer detail within an observed view.
World coordinates are east/north meters; bearings are clockwise, north=0, east=90.

DECISION: inspect the current RGB for plausible final targets before moving away.
If several objects fit, compare the distinguishing relations in the instruction:
which side of a reference, order along a route, intersection, shape and surroundings.
A matching object category or a high detector score alone does not identify the goal.
Use an available grounder with a SHORT visual category/phrase to get candidates;
keep place names, route relations and ordinal selection in your own comparison.
Tools may miss targets. Neither an empty result nor a tool error proves absence.
Use zoom when details in an already observed region would help; no remote previews.

Submit exactly one JSON object with a concise, evidence-based reason:
{"action":"locate","target":"final","view_id":"...","bbox":[100,200,300,400],"reason":"..."}
{"action":"locate","target":"auxiliary","view_id":"...","bbox":[100,200,300,400],"reason":"..."}
{"action":"tool","tool":"sam3","view_id":"...","query":"building","reason":"..."}
{"action":"tool","tool":"zoom","view_id":"...","bbox":[100,200,300,400],"reason":"..."}
{"action":"explore","mode":"point","view_id":"...","point":[500,200],"reason":"..."}
{"action":"explore","mode":"offset","turn_deg":20,"step_scale":0.6,"reason":"..."}
{"action":"explore","mode":"landmark","landmark_id":"...","reason":"..."}
{"action":"give_up","reason":"..."}
Grounding tool can instead be dino or locate_anything; use only names listed in
available_grounding_tools. At most two optional calls per round. Explore selects
an uncertain next observation, not a confirmed target. Offset moves by
step_scale*main_view_height_m at current_heading+turn_deg; scale must be positive.
Visible image points avoid compass arithmetic when repositioning. Invalid actions
cause NO movement. Choose another valid point instead of repeating rejected values.

A locate action only PROPOSES a candidate. The program will request a separate local
review before any final confirmation or auxiliary move. At this decision step return
locate, explore, tool or give_up; review/verdict are not decision actions.

For ordinal targets, identify the counting origin and the travel direction along the
relevant road before counting DISTINCT objects. Image top-to-bottom or left-to-right
order is not route order. Candidate initial_forward_m and initial_right_m measure
signed displacement in the ORIGINAL pose heading, not the updated exploration prior;
positive means ahead/right, negative means behind/left. These numbers do not choose
the road for you. Use visible road connectivity and the instruction's relative turns.
Fictional example: if approaching four roadside towers from the east and traveling
west, count from the eastern end; reversing travel reverses the counting order.

Interpret attributes literally: color of a BUILDING may describe its walls; only
an explicit roof-color phrase requires that roof color. Separate visible support,
visible contradictions and attributes unobservable from overhead. Missing views of
doors, windows, people or stairs are NOT negative observations. Do not invent them.
A uniquely supported object may be identified from visible shape and spatial relations
while honestly noting unverifiable facade details. When alternatives still fit equally,
remain uncertain. A group target needs a box around the stated group, not one member.
Use unmarked RGB for color/appearance; colored tool masks are annotations.
A candidate cut by the view boundary may need repositioning; a crop cannot reveal
its unseen part. Do not choose a wrong interior object merely because it is complete.
Candidate public_geometry is measured from its position; it does not certify identity.
The initial direction is an uncertain exploration prior, never proof of a target.

Fictional examples:
- A cream pavilion beside a lake may have a dark roof. Check the lake/pavilion relation;
  do not reject it solely because the instruction never specified a cream roof.
- Two possible sports courts: request dino query "sports court", then compare which
  court is beyond the footbridge. Review rejects a court on the wrong side.
- A region contains two requested storage sheds: locate their combined extent when
  the instruction targets the pair, rather than treating one detector box as the group.
- No supported candidate is visible: explore offset turn_deg=15, step_scale=0.7,
  inspect the new view, and update the prior. No fixed stage order is required.

CityNav public references: navigation.landmarks supplies current bearings, distances,
view overlap and observed extent for named places. Public coordinates guide exploration
but never count as visual detections. Candidate public_reference_distances are from the
CANDIDATE, not the agent. Compare all relevant named references; being closest to one
reference center is not equivalent to being at its entrance or a road intersection.
Catalog extents are bounding rectangles, not exact building/road outlines. Zero distance
to a rectangle alone cannot establish frontage, identity or an entrance. A missing view
of a facade feature must not override a strong visible layout match, nor establish one.
For multiple entries sharing a name, use their geometry rather than assuming one center
represents the entire street. Historical image axes differ from current heading.
"""


REVIEW_SYSTEM = """Verify the CURRENT proposed destination or auxiliary object in two overhead RGB images.
The first is its parent view; the second is a local view. Image headers give view_ids.
The thin yellow outline marks the UNCONFIRMED current proposal. Examine that object,
not a different object nearby. The marker and previous model claims are not evidence.

Use the full instruction and visible layout to distinguish this candidate from alternatives.
Separate three kinds of evidence: visible support, visible contradictions, and unobservable
attributes. "White building" can describe walls; it does NOT require a white roof.
Do not reject a building just because doors, windows, people, stairs or wall color are
not resolvable from above. Do not claim to have seen these attributes either. A strongly
supported layout/shape match can justify identification while stating these limits.
A visible contradiction (wrong side of a reference, wrong shape or explicit roof color)
justifies reject. Missing decisive evidence with several plausible alternatives justifies
uncertain. Never confirm based only on category or a detector score.
For an ordinal target, verify the counting origin and direction along the visible
route; image ordering alone is insufficient. For a pair/group, inspect the whole
group, not a single instance.
Public reference distances are geometric hints; rectangle overlap is not building identity.

Return ONE JSON object. Give a concise reason listing "support", "contradiction", and
"unobservable" evidence. Do not invent facts to fill a category; say none/unknown.
{"action":"review","verdict":"confirm","view_id":"displayed image ID","bbox":[100,200,300,400],"reason":"support: ...; contradiction: none; unobservable: ..."}
{"action":"review","verdict":"reject","reason":"support: ...; contradiction: ...; unobservable: ..."}
{"action":"review","verdict":"uncertain","reason":"support: ...; contradiction: none; unobservable: ...; unresolved alternatives: ..."}
A confirmed bbox must be refined in that displayed image's [0,1000] frame, x right/y down.
projection_in_review gives the current unconfirmed box in the local view. Never reuse
coordinates from the parent under the local view_id. Confirmation ends final localization
or moves to an auxiliary center; no fallback box is used when rejecting or uncertain.
"""

ANALYSIS_REPAIR = ("Output failed validation: {error}. Return the corrected three-field answer and the requested compact plan. "
                   "Use null for unsupported directions. No additional labels or long reasoning.")
