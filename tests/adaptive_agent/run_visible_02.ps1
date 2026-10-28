$ErrorActionPreference = 'Stop'
$workspace = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
Set-Location -LiteralPath $workspace
$Host.UI.RawUI.WindowTitle = '2DNav - hidden API key input and paired sample test'
Write-Host '2DNav: 3 AVDN + 3 CityNav paired visible-target cases.'
Write-Host 'Decision/fusion/review: qwen3.5-flash. Optional visual evidence: qwen3.8-flash.'
Write-Host 'The Python prompt hides your API key. Do not paste it into chat.'
& 'C:/Users/admin/anaconda3/python.exe' -u -B 'out/16_adaptive_agent/20261027/visible_02/harness.py' run --batch visible_02
$runExitCode = $LASTEXITCODE
Write-Host "Experiment exited with code $runExitCode. Logs are in out/16_adaptive_agent/20261027/visible_02."
Read-Host 'Press Enter to close'
