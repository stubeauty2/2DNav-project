# SatNav online pilot results

Branch: navAgent-SatNav. Fixed sample: five distinct trajectories per split; same ten episodes in both runs. No performance-based selection.

## Results

| Run | Resolved directions | Below 90 degrees | Navigation successes |
|---|---:|---:|---:|
| Initial baseline | 1/10 | 1/10 | 1/10 |
| Relative-turn extraction + code composition | 7/10 | 7/10 | 2/10 |

V2 seen success: 1/5. Unseen: 1/5. Among seven direction-qualified episodes, one succeeded. Four fully numeric routes have direction errors below 0.2 degrees. Three heading-cone estimates have errors below 14.2 degrees. Three outputs remain unresolved (one is an extraction failure on the initial zero turn, two include visual-relative turns).

| Episode | Direction error (deg) | Stop distance (m) | Success |
|---|---:|---:|---|
| satnav-val_seen-Geneva-5-291 | 0.028 | 40.43 | False |
| satnav-val_seen-LosAngeles-1-709 | unresolved | 31.51 | False |
| satnav-val_seen-Paris-4-873 | unresolved | 27.61 | True |
| satnav-val_seen-Paris-4-1107 | 0.169 | 864.85 | False |
| satnav-val_seen-Minneapolis-4-968 | 14.111 | 82.40 | False |
| satnav-val_unseen-Orlando-1-933 | unresolved | 112.94 | False |
| satnav-val_unseen-Orlando-1-997 | 7.643 | 178.02 | False |
| satnav-val_unseen-Rotterdam-1-1118 | 13.100 | 549.75 | False |
| satnav-val_unseen-Orlando-1-1040 | 0.098 | 15.08 | True |
| satnav-val_unseen-Auckland-1-818 | 0.015 | 177.16 | False |

## Interpretation

Four episodes have visually supported final target matches in recorded views (Geneva-291, LosAngeles-709, Paris-873, Orlando-1040). Two of those fail the endpoint threshold despite localization. Six other contact-sheet reviews do not establish the intended target identity; this is not proof the object was absent in every pixel. Review evidence is stored in offline_visual_audit/reviewed.json. There are no official target object boxes in this audit.

Correct initial direction is insufficient: the agent can change course, confuse similar pools/docks, or confirm a false target. All ten V2 episodes self-confirmed; only two passed the independent endpoint metric. This demonstrates why agent confirmation cannot be used as the success label.

## Protocol and verification

The existing AVDN exploration/tool/review loop is reused with a SatNav raster adapter and an explicit final stopping-pose call. This is the project continuous-action protocol (768px, 100m footprint), not native SatSim discrete-action parity. Stops use only instruction and visual observations. The 30m endpoint metric uses GT only offline.

Both runs have ten terminal result files. Structural journal checks found no goals, goal_xy_m, waypoints or reference_path fields in model contexts. The real-raster scripted control-flow test passes; it is distinct from the online results above. Source snapshots, API usage, tool results, actions and images are retained under each experiment directory. Online initialization/review uses qwen3.8-flash; decisions use qwen3.7-plus; local SAM3 provides proposals. Thinking disabled.

Artifacts: out/satnav/pilot_20261109 and out/satnav/pilot_v2_20261109. See RUNNING.md for reproduction. No credentials are stored in source or journals.

Conclusion: implementation and online sampling are complete; current performance does not establish reliable SatNav navigation. Further improvements should target extraction repair, maintaining the goal-direction prior, relational target verification, and stopping-position estimation.
