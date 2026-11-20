# Runs the remaining Paper A experiments outside the Claude Code session, low priority,
# 6 workers (each ~90 MB; CPU, not RAM, was the bottleneck). Both scripts checkpoint per pool and resume.
Set-Location $PSScriptRoot
$env:WORKERS = "6"
$env:PYTHONIOENCODING = "utf-8"
$py = "..\..\.venv\Scripts\python.exe"
$log = "logs\run_rest.status"
foreach ($s in @("vams_sweep", "headroom")) {
    "$s start $(Get-Date -Format HH:mm:ss)" | Out-File -Append -Encoding utf8 $log
    & $py "$s.py" *> "logs\$s.log"
    "$s exit=$LASTEXITCODE $(Get-Date -Format HH:mm:ss)" | Out-File -Append -Encoding utf8 $log
}
"ALL DONE $(Get-Date -Format HH:mm:ss)" | Out-File -Append -Encoding utf8 $log
