# 2DNav-project

## Same-heading INS execution

The ANDH-Full runner supports the legacy one-step-per-INS path and the
event-aware same-heading window path:

```powershell
python PSC-AVDN/script/ANDH-Full/Search_Confirmation.py `
  --execution-mode ins-window `
  --max-routes 1
```

Confirmation is enabled by default. To run Search-only window execution and
commit Search's contiguous completed event prefix directly, use:

```powershell
$env:QWEN_API_KEY = "<configured API key>"
conda run --no-capture-output -n 2DNav python `
  PSC-AVDN/script/ANDH-Full/Search_Confirmation.py `
  --execution-mode ins-window `
  --no-confirmation `
  --out-dir out/preds/andh_full_sample2_windowed_no_confirmation/search_output
```

Use `--confirmation` to turn both the secondary visual Confirmation call and
the window-level progress Confirmation back on.

Windowed results are written to
`out/preds/andh_full_sample2_windowed/search_output` by default.  Each route
contains `ins_sessions`, `execution_windows`, `search_runs`,
`confirmation_runs`, `event_progress`, `observation_views`, and a committed
`physical_trajectory`.  Use `--execution-mode legacy` for the original
baseline-compatible execution path.
