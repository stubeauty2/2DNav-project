Current implementation and online pilot: see [RUNNING.md](RUNNING.md) and [RESULTS.md](RESULTS.md). The feasibility notes below predate implementation.

# SatNav initialization feasibility audit

Scope: annotation-based feasibility, not a completed navigation port.

Data: all episode JSON/Parquet splits, scene list, and license files are downloaded to datasets/SatNav. Official hashes pass for 11/12 manifest entries; README alone differs. Scene access is now authorized. Full scene download is complete: all 59 GeoTIFFs independently passed SHA256 checks against the pinned release manifest. The manifest path set matches the downloaded files exactly. See datasets/SatNav/final_verification.json (70/71 entries pass overall; only the annotation README differs). Auckland-1.tif has passed SHA256 and its coordinate metadata has been inspected.

Existing AVDN initialization extracts semantic movement stages and composes their vectors. CityNav selects a public catalog landmark anchor. SatNav episode waypoints, reference_path and goals are demonstration/evaluation annotations, not a public landmark catalog; never expose them to the initialization policy. aux_info.landmarks contains category strings, not a usable coordinate catalog.

Recommended inference inputs: instruction_text, start_position, start_rotation, and initial rendered RGB when available. Outputs can retain direction, final_target, auxiliary. Keep a separate diagnostic plan with ordered movement distances, turn constraints, final view-region condition, provenance and uncertainty.

Numeric instructions: extract chronological movements and signed turns from language. Compose east/north displacement using the initial heading. Do not sort actions solely by mention order: instructions such as "Execute a turn once you have flown ..." reverse textual and execution order. Unknown distances and angles must remain unknown, not be silently filled from reference_path.

Visual-relative instructions: image quadrants constrain a landmark bearing relative to the drone, not the displacement toward the next or final target. With no visible landmark or known coordinates, a single reliable start-to-final direction is not identifiable. The initial heading is an exploration prior only. Either preserve intermediate landmark turn stages, or accept a weak-prior baseline with explicit uncertainty. This is a method-level change for the full split.

Target extraction: resolve the final stopping clause, including pronouns and repeated objects. Preserve appearance, reference relations, and the requested image region. A final target being lower-right is not equivalent to flying to its center. Separate object grounding from the stopping-position condition. Do not use aux_info.landmarks[-1] as a substitute for the full language description.

Evidence: run audit_annotations.py and probe_numeric_routes.py. The latter is deliberately a regex feasibility probe, NOT a validated production parser. It only accepts alternating numeric move/turn events and may misparse narrative order; failures are retained in numeric_route_probe.json. Coverage is not the proportion of all semantically solvable episodes. Ground truth is used only after prediction for offline scoring.

Evaluation results: seen 470/1509 numeric parses (31.15%), median bearing error 0.059 deg, 95.74% within 15 deg. Unseen 766/2872 (26.67%), median 0.104 deg, 87.86% within 15 deg. Initial-heading baseline on unique trajectory IDs: seen median 31.72 deg, unseen 45.42 deg. These two evaluations use different populations and must not be presented as a paired improvement.

Conclusion: numeric Landmark instructions support AVDN-style initialization; a CityNav-style coordinate anchor is not provided by the episode schema. The entire Landmark split cannot be reliably reduced to a single initial direction without additional visual evidence or staged exploration. Automated target extraction and navigation success remain untested. One north-up start/goal image pair has been manually inspected after authorized scene download.

## Geospatial validation

Auckland-1 is a 21793 x 12225, four-band uint8 GeoTIFF in EPSG:3857. All 2484 start/goal points for its 1242 Landmark evaluation episodes fall inside its bounds after WGS84-to-map transformation. This validates coordinate coverage, not target identity or navigation success. Web Mercator scale varies with latitude: a local ground displacement should be projected with the local scale (approximately 1/cos(latitude)), or generated geodesically before projecting. Do not crop 100 projected meters and assume a 100-meter ground footprint. Heading rotation and forward-up image coordinates must match SatSim. Auckland color interpretation explicitly identifies bands as red, green, blue and alpha.

Auckland example episode 246: a north-up 100 m ground-footprint goal crop shows the described dirt baseball diamond near the upper edge, whereas the crop center lies over residential roofs. This illustrates why goal coordinates are not object-center annotations. The preview is an offline diagnostic, not the heading-aligned observation fed to a policy. See datasets/SatNav/auckland_start_goal_preview.png.

## Landmark counts

Release val_seen (paper test_seen): 1509 episodes, 529 unique (scene_id, trajectory_id) pairs. Release val_unseen (paper test_unseen): 2872 episodes, 981 unique pairs. Train: 40343 episodes, 14253 unique pairs. Multiple instruction episodes may share a trajectory.
