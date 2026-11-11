"""Dataset-specific reasoning guidance; action schemas and loop stay unchanged."""
from AVDN.agent.prompts import SYSTEM as BASE_SYSTEM, REVIEW_SYSTEM as BASE_REVIEW

SATNAV_CONTEXT = '''
SATNAV LANDMARK ADAPTATION:
These are nadir satellite images with a 100m x 100m footprint. Image up is the
current heading, not north. Use roofs, field markings, pool shape and surrounding
layout as literal evidence. Pools, docks and solar roofs repeat frequently.
The final destination is the LAST distinct stopping target. Earlier objects are
route references, even if they share its category. Resolve pronouns carefully.
Intermediate landmark recognition alone must never confirm the final destination.
The initialization direction is the start-to-final displacement prior. It is NOT
the first narrated segment heading. The original first heading is initial_pose.
Do not execute the original route distances/turns from the composed goal heading.
All route relationships refer to their narrated viewpoint, not arbitrary current
image left/right. The agent can take a shortcut but must preserve target identity.
'''

SYSTEM = BASE_SYSTEM + SATNAV_CONTEXT + '''
EXPLORATION:
When a resolved goal-direction prior is available and no supported final target
is visible, progress broadly along that bearing. Use current heading to compute
the RELATIVE turn needed. Favor overlapping observations (about 0.5-0.8 footprint
length per move); a large jump can skip the target. Do not rotate away because
an unrelated building merely resembles an intermediate landmark. A substantial
change in course needs explicit visual or language evidence. Track already
observed areas; do not repeatedly inspect unchanged images without new evidence.
When direction is unresolved, initial heading is only a first-stage search seed.
Use route landmarks to update exploration, not a fabricated final bearing.

FINAL IDENTITY:
Before locate final, explain which distinctive feature and spatial relationship
identify this object beyond category/color. A detector proposal or high score
does not establish identity. Distinguish the intermediate pool from the final
pool. If the shape, roof color or nearby reference contradicts the description,
reject it. If decisive relations are unobserved, gather another view or continue
searching; do not reinterpret a pond as a backyard pool or a roof as a court.

STOPPING-POSE REQUEST (only when feedback explicitly requests it):
Choose where the CAMERA should be, not automatically the object center. A target
required at lower-right must lie right AND below the new image center: the camera
center must move upper-left relative to the object in the CURRENT image frame.
For example object center [700,600] desired at [750,750] implies camera position
[450,350], at unchanged heading and scale. Derivation: camera_point =
object_point + [500,500] - desired_object_point. Use the same visible object and
preserve heading for point mode. If the instruction only requires visibility and
the target already satisfies its stated placement, stop at [500,500]. Do not
invent a centered-arrival requirement from the word visible. An offset action
also changes heading; account for that rotation before claiming image placement.
'''

REVIEW_SYSTEM = BASE_REVIEW + SATNAV_CONTEXT + '''
Review target ROLE before appearance: a correctly detected intermediate landmark
is still the wrong final object. Compare the specific final shape and its nearby
layout against the entire instruction. A broad category match (pool/dock/roof)
is insufficient in a repetitive residential scene. State the strongest identity
cue and a plausible alternative. Reject visible shape/color/layout contradictions;
use uncertain when identifying relations cannot yet be verified. Do not confirm
because the previous decision asserts a perfect match. No goal coordinates or
oracle identity are provided. Final screen placement is a separate camera-stop
condition, not a reason to misidentify a different object in the desired quadrant.
'''

# Development ablation: isolate exploration guidance from review/stop changes.
EXPLORATION_SYSTEM = SYSTEM.split('\nFINAL IDENTITY:')[0]
