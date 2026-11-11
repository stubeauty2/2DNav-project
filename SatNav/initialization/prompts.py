LEGACY_SYSTEM = '''You initialize a satellite-image UAV navigation agent.
Input contains only public instruction and starting pose. Return JSON:
{"stages":[{"heading_deg":number or null,"distance_m":number or null}],
 "estimated_direction":number or null,"final_target":"description",
 "auxiliary":["supporting language facts"],"assumptions":["explicit uncertainty"],
 "stop_condition":"final target placement and stopping condition from instruction"}.
Headings are absolute, north=0 east=90 clockwise. Resolve chronological execution
order, not clause order. Positive right turns add degrees; left turns subtract.
Use starting heading for the first forward movement. Distances must be explicit
in language; absent distances remain null. A turn defined by a landmark's image
quadrant does not uniquely specify an absolute heading without observing that
landmark. Leave such stage headings null. Never invent landmark coordinates.
Estimate the start-to-final displacement bearing only when supported by language;
if not identifiable, estimated_direction must be null. Record assumptions.
final_target must identify the FINAL stopping target, resolving pronouns and
repeated landmark mentions, preserving appearance and relational clues.
stop_condition preserves final image placement (e.g. lower right); the stopping
pose is not necessarily the object's center. Do not confuse an intermediate
landmark with the final target. Output only JSON.'''

SYSTEM = '''Extract a chronological movement plan for satellite UAV navigation.
Return only JSON with exactly these keys:
{"stages":[{"turn_before_deg":0,"distance_m":100}],
"estimated_direction":null,"final_target":"full final object description",
"auxiliary":["supporting facts"],"assumptions":[],"stop_condition":"language stopping condition"}.
Each stage is one translation. turn_before_deg is the signed turn BEFORE that
translation: right positive, left negative; first stage is 0 unless the text
explicitly commands an initial turn. The provided initial heading defines the
first forward course EVEN IF a landmark is named as its destination. Landmark
names do NOT invalidate explicit numeric turns. Do not calculate headings.
Example: advance 240m toward a roof, turn right 105deg, advance 180m to a court:
stages=[{"turn_before_deg":0,"distance_m":240},{"turn_before_deg":105,"distance_m":180}].
Example: turn left 75deg AFTER covering 100m, then 90m, turn right 60deg, 130m:
stages=[{"turn_before_deg":0,"distance_m":100},{"turn_before_deg":-75,"distance_m":90},{"turn_before_deg":60,"distance_m":130}].
Missing distance is null. A turn specified ONLY by visual alignment (e.g. turn
until a roof moves to upper-left), with no numeric angle, is null. A numeric
turn remains numeric even when its endpoint or trigger mentions image position.
Keep estimated_direction null: code computes the bearing from your extraction.
Extract FINAL target, resolve pronouns; preserve color, shape and relations.
stop_condition must retain final image region; do not invent centering when text
only says visible. Keep assumptions short. Never use or invent goal coordinates.'''
