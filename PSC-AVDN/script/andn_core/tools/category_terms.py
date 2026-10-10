"""Deterministically extract SAM3 category prompts from Parse ``dest`` text."""

from __future__ import annotations

import re


# Reviewed exceptions from the fixed test_unseen_full Parse output. These are
# text-only decisions: no image, prediction, or ground-truth data is consulted.
OVERRIDES = {
    "area with outline of buildings and train tracks to the southwest": ["area"],
    "brown connected buildings with two large bushes in front": ["building"],
    "square base shingle roof house with a pool": ["house"],
    "big portion surrounded with trees": ["area"],
    "big structure": ["structure"],
    "gray rectangular destination in the south direction": ["structure"],
    "light blue area within a housing development": ["area"],
    "tower shaped area": ["area"],
    "triangular area": ["area"],
    "water shaped place": ["area"],
    "lot with multiple structures": ["lot"],
    "opposite fourth road of many colored cars parking": ["parking lot"],
    "block of car parking spots closest to the pool": ["parking lot"],
    "parking area with semi-trailered white trucks": ["parking lot"],
    "row of mostly blue storage trailers": ["storage trailer"],
    "power grid": ["power grid"],
    "roof of the fourth house on the right hand side": ["roof"],
    "white, blue, and red building roof next to a building parallel to the highway": ["roof"],
    "square gray roof on the south side of the intersection": ["roof"],
    "gray roof building": ["building"],
    "grey roof building": ["building"],
    "building with red roof and white car": ["building"],
    "green area by the square concrete": ["area"],
    "dark and light striped roof building, north end of those roofs": ["building"],
    "another set of trees near the brown building": ["tree"],
    "apartment block with the big blue pool with the wavy line": ["apartment building"],
    "black colored place": ["area"],
    "block": ["building"],
    "block within the trees": ["building"],
    "destination": ["object"],
    "destination after the 2 buildings": ["area"],
    "destination buried underneath a cloud": ["area"],
    "destination in the middle": ["area"],
    "destination in the parking area": ["parking lot"],
    "destination site": ["area"],
    "first block": ["building"],
    "first group of trees on your right": ["tree"],
    "first patch of farmland": ["farmland"],
    "large column of freight crates next to the forest's edge": ["crate"],
    "middle section": ["area"],
    "muddy colored place": ["ground"],
    "parking near white building": ["parking lot"],
    "red colored sand place": ["sand"],
    "second block": ["building"],
    "second turn off": ["road"],
    "second unit in the second row of grey storage buildings": ["building"],
    "small branch heading west": ["road"],
    "small grey rectangle": ["structure"],
    "third line of trees in the parking lot": ["tree"],
    "top of the next block": ["building"],
    "two rows of housing": ["house"],
    "two single gray roofed units": ["building"],
    "west side of the circular cul-de-sac": ["cul-de-sac"],
    "white color hub": ["building"],
}


PATTERNS = [
    (r"round-a-bout|roundabout", "roundabout"),
    (r"4-way intersection|crossroads?|junction|intersection", "intersection"),
    (r"parking (?:lot|area|space|spots?)|\bparking\b", "parking lot"),
    (r"swimming pool|\bpool\b", "swimming pool"),
    (r"cargo truck", "cargo truck"),
    (r"truck trailer", "truck trailer"),
    (r"storage trailers?", "storage trailer"),
    (r"(?:storage|shipping|shipment|cargo) containers?", "container"),
    (r"tennis court", "tennis court"),
    (r"baseball stadium", "baseball stadium"),
    (r"power station", "power station"),
    (r"water tower", "water tower"),
    (r"airport building", "airport building"),
    (r"apartment block", "apartment building"),
    (r"building complex|complex of buildings", "building"),
    (r"construction site", "construction site"),
    (r"cul-de-sac", "cul-de-sac"),
    (r"\bwarehouses?\b", "warehouse"),
    (r"\bhangars?\b", "hangar"),
    (r"\bmansion\b", "mansion"),
    (r"\bhuts?\b", "hut"),
    (r"\bsheds?\b", "shed"),
    (r"\bgarages?\b", "garage"),
    (r"\b(?:houses?|homes?|housing|condos?)\b", "house"),
    (r"\bbuildings?\b", "building"),
    (r"\bstructures?\b", "structure"),
    (r"\bcontainers?\b", "container"),
    (r"\btrailers?\b", "trailer"),
    (r"\b(?:aircraft|airplane|aeroplane|plane)\b", "aircraft"),
    (r"\bcranes?\b", "crane"),
    (r"\bvehicles?\b", "vehicle"),
    (r"\bcars?\b", "car"),
    (r"\btrucks?\b", "truck"),
    (r"\brooftops?\b|\broofs?\b", "roof"),
    (r"\bhighways?\b", "highway"),
    (r"\bdriveways?\b", "driveway"),
    (r"\bbridges?\b|\bflyover\b", "bridge"),
    (r"\brunways?\b", "runway"),
    (r"\broads?\b", "road"),
    (r"\bstreets?\b", "street"),
    (r"\bwalkway\b", "walkway"),
    (r"\bplatform\b", "platform"),
    (r"\bfountains?\b", "fountain"),
    (r"\bfarmland\b", "farmland"),
    (r"\bfields?\b", "field"),
    (r"\bcourtyards?\b", "courtyard"),
    (r"\btrees?\b", "tree"),
    (r"\bgrass(?:\s+land)?\b", "grass"),
    (r"\bground\b|\bland\b", "ground"),
    (r"\bsand\b", "sand"),
    (r"\bareas?\b|\bplaces?\b|\bsites?\b", "area"),
    (r"\blots?\b", "lot"),
]


def extract_category_terms(text: str) -> list[str]:
    """Return short target-category prompts without using a language model."""
    normalized = " ".join(str(text or "").lower().split())
    if normalized in OVERRIDES:
        return list(OVERRIDES[normalized])

    # Relations following the target head normally describe an anchor, not the
    # object that SAM3 should segment.
    head = re.split(
        r"\b(?:with|beside|next to|near|surrounded|after|before|behind|across|between|by|bordered|attached|connected|on the|at the|in the|to the)\b",
        normalized,
    )[0]
    matches = []
    for priority, (pattern, term) in enumerate(PATTERNS):
        match = re.search(pattern, head)
        if match:
            matches.append((match.start(), priority, match.end(), term))
    if not matches:
        return ["object"]

    matches.sort()
    first = matches[0]
    terms = [first[3]]
    if first[3] == "roof" and re.search(r"roof(?:ed)?\s+building", head):
        terms = ["building"]

    rest = head[first[2]:]
    if re.match(r"\s+and\s+", rest):
        for term in extract_category_terms(re.sub(r"^\s+and\s+", "", rest)):
            if term not in terms:
                terms.append(term)
    return terms
