# Runs the Paper B edge-extension follow-up outside the Claude Code session, low priority,
# 6 workers (see run_rest.ps1). Needs a finished paperB_baselines_results.json; resumes.
Set-Location $PSScriptRoot
$env:WORKERS = "6"
$env:PYTHONIOENCODING = "utf-8"
$py = "..\..\.venv\Scripts\python.exe"
$log = "logs\run_paperB.status"
foreach ($s in @("paperB_edge")) {
    "$s start $(Get-Date -Format HH:mm:ss)" | Out-File -Append -Encoding utf8 $log
    & $py -u "$s.py" *> "logs\$s.log"
    "$s exit=$LASTEXITCODE $(Get-Date -Format HH:mm:ss)" | Out-File -Append -Encoding utf8 $log
}
"EDGE DONE $(Get-Date -Format HH:mm:ss)" | Out-File -Append -Encoding utf8 $log
