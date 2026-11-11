# SatNav prompt adaptation without changing the navigation loop

## Conclusion

The previous version already adapted initialization, raster calibration and physical
stopping, but used AVDN decision/review system prompts with a small SatNav context
addition. Dataset-specific system prompts were therefore implemented and tested.
Neither tested prompt variant improved generalization. Keep baseline as the default;
the alternatives remain available for reproducible ablations, not as recommended upgrades.

## Controlled comparison

Thirty unique trajectory samples: ten previous development episodes and twenty
additional episodes (ten seen, ten unseen), excluding all development trajectories.
Every variant receives identical frozen initialization and public inputs. Models,
SAM3, maximum rounds/requests, view footprint, continuous actions and 30m success
threshold remain unchanged. Main navigation, SatNav stop adapter, view adapter,
action contracts and grounding interface hashes match the pre-experiment snapshot.
Only prompt selection was added to the shared API client; existing AVDN defaults
remain unchanged. No GT is used in the policy.

| Prompt | Development 10 | Additional seen 10 | Additional unseen 10 | Additional total 20 |
|---|---:|---:|---:|---:|
| Baseline | 2/10 | 4/10 | 2/10 | 6/20 |
| Full SatNav adaptation | 2/10 | 2/10 | 0/10 | 2/20 |
| Exploration-only adaptation | 3/10 | 3/10 | 0/10 | 3/20 |

Full adaptation changes exploration, review and stopping guidance. Exploration-only
retains the original review and stop guidance but adds dataset context and exploration
instructions. On the additional twenty episodes, full adaptation gains two successes
and loses six; exploration-only gains two and loses five. Development-only improvements
did not transfer. These small samples and single model rollouts do not establish a
statistically reliable universal ranking.

Full adaptation was frozen before inspecting the additional sample outcomes.
Exploration-only was fixed on development data but tested on the same additional
sample after partial full-adaptation outcomes were observed. It is secondary validation,
not a new blinded holdout. No test-specific coordinates or answers were inserted.

## Evidence and failure mechanisms

- Paris-1107: baseline first turned -120 degrees and jumped 350m despite a correct
  goal-direction prior; adapted exploration advanced 80m along the prior. Full
  adaptation improved endpoint error from 864.85m to 18.95m. This supports the
  direction/route-confusion diagnosis, but is one development case.
- Minneapolis-922 in the extension improved from 1146.70m to 25.04m under full
  adaptation. Shorter overlapping moves can help some searches.
- TheBayArea-1432: final JSON point contradicted the model's own stopping calculation.
  Explicit geometry prose did not reliably produce a geometrically correct action.
- LosAngeles-1204: the model inverted camera translation when centering the target;
  endpoint error increased from 11.85m to 59.83m.
- TheBayArea-1589: visibility-only stopping guidance was overgeneralized to a route
  with an arrival requirement, causing premature stop (6.76m to 72.04m error).
- Full adaptation eliminated >1-footprint offset steps (78 baseline vs 0 adapted),
  but large relative turns increased from 3 to 7. Prompt guidance is not an enforced
  motion policy. Simply reducing step size did not solve identity and stopping errors.

The extension initialization resolves 12/20 headings; all twelve are below 90deg
error. Unresolved directions remain explicitly marked, with initial heading used
only as an exploration fallback. Prompt arms share these outcomes.

Network errors are retained, not silently retried into better results. One full-
adaptation episode terminates with interface_error after repeated ProxyError.
Removing this pair gives baseline 6/19 versus full adaptation 2/19, so that failure
does not explain the regression. Other transient errors are present in journals.

## Deliverables and reproducibility

- `SatNav/prompts.py`: full and exploration-only variants.
- `SatNav.cli run --prompt-variant baseline|satnav|exploration`; default baseline.
- `SatNav/compare.py`: paired inputs must match; incomplete episodes are not failures.
- `out/satnav/prompt_*comparison.json`: complete paired results.
- `out/satnav/prompt_loop_invariant_audit.json`: unchanged core file checks.
- `out/satnav/prompt_final_audit.json`: 80 new online navigation runs and journal checks
  (development 10+10, extension 20+20+20); baseline development results were reused.
- `out/satnav/prompt_regression_diagnosis.json` and `prompt_behavior_audit.json`:
  trace-based failure evidence. All experiment folders contain exact source/prompt
  snapshots, API journals, tool outputs and views.

Two offline regression tests pass: prompt injection preserves old defaults, and the
real-raster exploration/localization/physical-stop control flow remains functional.
No goal/waypoint/reference-path keys were found in audited model contexts. These
checks do not substitute for online performance results.

The next evidence-driven direction is structured use of already extracted public
route/stop constraints, plus better target-relation verification, rather than adding
more generic prompt prohibitions. That further algorithmic adaptation was not claimed
as implemented in this prompt-only comparison. Native SatSim parity and a large-scale
benchmark remain outside the conclusions of these continuous-action pilot tests.
