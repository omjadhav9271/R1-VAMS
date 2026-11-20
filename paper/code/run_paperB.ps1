# Runs the Paper B baseline experiment outside the Claude Code session, low priority,
# 6 workers (see run_rest.ps1). The script checkpoints per pool x drift cell and resumes.
Set-Location $PSScriptRoot
$env:WORKERS = "6"
$env:PYTHONIOENCODING = "utf-8"
$py = "..\..\.venv\Scripts\python.exe"
$log = "logs\run_paperB.status"
foreach ($s in @("paperB_baselines")) {
    "$s start $(Get-Date -Format HH:mm:ss)" | Out-File -Append -Encoding utf8 $log
    & $py "$s.py" *> "logs\$s.log"
    "$s exit=$LASTEXITCODE $(Get-Date -Format HH:mm:ss)" | Out-File -Append -Encoding utf8 $log
}
"ALL DONE $(Get-Date -Format HH:mm:ss)" | Out-File -Append -Encoding utf8 $log
