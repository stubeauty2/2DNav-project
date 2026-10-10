"""Stage-scoped evidence memory. Model summaries never mutate geometry."""
from dataclasses import dataclass, field
from .tools.geometry import polygon_iou


@dataclass
class TrajectoryMemory:
    observations: dict = field(default_factory=dict)
    stages: dict = field(default_factory=dict)

    def stage(self, stage_id, destination=None):
        stage = self.stages.setdefault(stage_id, {
            "destination": destination, "observation_ids": [], "candidates": {}, "events": []})
        if destination is not None:
            stage["destination"] = destination
        return stage

    def observe(self, observation):
        self.observations[observation.observation_id] = observation.metadata()
        stage = self.stage(observation.stage_id)
        stage['observation_ids'].append(observation.observation_id)

    def event(self, stage_id, kind, **data):
        stage = self.stage(stage_id)
        event = {"event_id": f"{stage_id}:e{len(stage['events']) + 1}",
                 "kind": kind, **data}
        stage['events'].append(event)
        return event

    def candidate(self, stage_id, observation, bbox, polygon, center, result):
        candidates = self.stage(stage_id)['candidates']
        cid = f"{stage_id}:c{len(candidates) + 1}"
        candidates[cid] = {
            "candidate_id": cid, "stage_id": stage_id,
            "source_observation_id": observation.observation_id,
            "source_view_id": observation.observation_id + ":rgb",
            "source_bbox_2d": list(bbox), "map_polygon": polygon, "map_center": center,
            "evidence_view_ids": result.get('evidence_view_ids', []),
            "support": result.get('support', []), "conflicts": result.get('conflicts', []),
            "status": "pending", "attempt_count": 0,
            "first_seen_round": observation.search_step, "last_seen_round": observation.search_step,
            "confirmation_observations": [], "confirmation_history": [],
            "observations": [{"round": observation.search_step,
                              "source_view_id": observation.observation_id + ":rgb",
                              "bbox_2d": list(bbox),
                              "support": list(result.get('support', [])),
                              "conflicts": list(result.get('conflicts', []))}],
        }
        return candidates[cid]

    def merge_candidate(self, stage_id, observation, bbox, polygon, center, result, threshold=0.5):
        candidates = self.stage(stage_id)['candidates']
        matches = [(polygon_iou(polygon, c['map_polygon']), c) for c in candidates.values()]
        overlap, existing = max(matches, key=lambda item: item[0]) if matches else (0, None)
        if overlap < threshold:
            return self.candidate(stage_id, observation, bbox, polygon, center, result), True
        evidence = {"round": observation.search_step,
                    "source_view_id": observation.observation_id + ":rgb", "bbox_2d": list(bbox),
                    "support": list(result.get('support', [])), "conflicts": list(result.get('conflicts', []))}
        if evidence not in existing['observations']:
            existing['observations'].append(evidence)
        existing['last_seen_round'] = observation.search_step
        for key in ('support', 'conflicts', 'evidence_view_ids'):
            existing[key] = list(dict.fromkeys(existing[key] + result.get(key, [])))
        # Prefer a non-truncated, recent observation without clearing a rejection.
        def margin(box):
            return min(box[0], box[1], 767-box[2], 767-box[3])
        if margin(bbox) > 0 or margin(existing['source_bbox_2d']) <= 0:
            existing.update(source_observation_id=observation.observation_id,
                            source_view_id=observation.observation_id + ':rgb',
                            source_bbox_2d=list(bbox), map_polygon=polygon, map_center=center)
            existing['support'] = list(result.get('support', []))
            existing['conflicts'] = list(result.get('conflicts', []))
        return existing, False

    def summary(self, stage_id, confirmation_views=None):
        """Semantic evidence only: never leak old pixel boxes or stale image IDs."""
        candidates = []
        for c in self.stage(stage_id)['candidates'].values():
            item = {k: c.get(k) for k in ('candidate_id', 'status', 'support', 'conflicts', 'rejection_reason')}
            item['sightings'] = len(c.get('observations', []))
            item['last_confirmation'] = c.get('confirmation_history', [])[-1:]
            # History stores geometry internally; expose only its verbal assessment.
            item['last_confirmation'] = [{k: h.get(k) for k in ('round', 'status', 'reason')}
                                         for h in item['last_confirmation']]
            if confirmation_views is not None:
                item['current_confirmation_view_id'] = confirmation_views.get(c['candidate_id'])
            candidates.append(item)
        return {'destination': self.stage(stage_id)['destination'], 'candidates': candidates,
                'recent_search_and_confirmation': [
                    {'kind': e['kind'], 'round': e.get('round')}
                    for e in self.stage(stage_id)['events']
                    if e['kind'] in ('searched_no_match', 'uncertain', 'confirmation_passed')][-3:]}

    def record_confirmation(self, stage_id, result, round_n):
        for assessment in result.get('candidate_assessments', []):
            candidate = self.stage(stage_id)['candidates'][assessment['candidate_id']]
            candidate.setdefault('confirmation_history', []).append({'round': round_n, **assessment})
            candidate['status'] = assessment['status']
            candidate['conflicts'] = list(assessment.get('unresolved', []))
            if assessment['status'] == 'rejected':
                candidate['rejection_reason'] = assessment['reason']
        if result['decision'] == 'confirmed':
            candidate = self.stage(stage_id)['candidates'][result['candidate_id']]
            candidate['status'] = 'confirmed'
            candidate['confirmation_result'] = dict(result)

    def serialize(self):
        return {"observations": self.observations, "stages": self.stages}
