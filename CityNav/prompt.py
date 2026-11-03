"""General geographic grounding rules and wholly fictional demonstrations."""

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
