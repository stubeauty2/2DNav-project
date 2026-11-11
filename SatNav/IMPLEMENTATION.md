# SatNav unified navigation implementation

Branch: navAgent-SatNav, created from navAgent.

## Required implementation and evidence

- Public preparation: LandmarkSet instruction, start position/heading and raster calibration only. Write goals into a separate evaluator file; never forward waypoints or reference paths to the agent.
- Initialization: online language parsing of chronological movements, final target and stopping constraints. Compose bearings in local east/north coordinates; record unresolved direction and assumptions explicitly.
- Environment: local metric coordinates and georeferenced crops with heading-up orientation, respecting Web Mercator scale. Preserve the existing observation/tool/localization loop.
- Navigation: reuse existing grounding services and model request journals. Final localization must lead to an explicit stopping pose; object center and navigation endpoint are not interchangeable.
- Evaluation: deterministic seen/unseen sample selected before predictions. Measure initialization angular error and fraction below 90 degrees, then navigation action traces, target visibility evidence, localization and final stop distance (30 m threshold). Do not equate visibility of the GT stopping point with visibility of the target object.
- Online tests: retain API requests, outputs, usage, images and failures. Report all sampled episodes, including initialization failures; any direction-qualified subset must be reported separately.

## Inspection findings

AVDN/navigation.py implements the existing loop. Its confirmed-final branch records localization with actual_move=False and terminates. This requires a SatNav stopping-pose adaptation before claiming navigation success.

AVDN/tools/views.py uses public raster calibration, 768-pixel crops and continuous movement. SatNav's official observation/action settings differ; the adapter must document whether a run uses native discrete actions or the project's continuous action abstraction.

AVDN initialization uses exactly direction, final_target, auxiliary (an array of strings). Preserve this contract; store structured route diagnostics separately instead of changing auxiliary into an object.
