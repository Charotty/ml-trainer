# Finish На спине extract + NaTrendStore; keep window open.
$ErrorActionPreference = 'Continue'
$host.UI.RawUI.WindowTitle = 'ml-trainer FINISH NA SPINE'

# Pre-create DrvFS mirror from Windows (WSL often cannot create new files under /mnt/e).
New-Item -ItemType Directory -Force -Path 'E:\ml\ml-trainer\results' | Out-Null
New-Item -ItemType File -Force -Path 'E:\ml\ml-trainer\results\wsl_na_spine_finish.log' | Out-Null

Write-Host '=== WSL: finish F:/Na spine + NaTrendStore ===' -ForegroundColor Cyan
Write-Host 'Live log (preferred): /tmp/ml_trainer_logs/wsl_na_spine_finish.log'
Write-Host 'Mirror: E:\ml\ml-trainer\results\wsl_na_spine_finish.log'
Write-Host 'Tail: wsl -e bash -lc "tail -f /tmp/ml_trainer_logs/wsl_na_spine_finish.log"'
Write-Host ''
wsl -e bash /mnt/e/ml/ml-trainer/scripts/tools/wsl_finish_na_spine.sh
$code = $LASTEXITCODE
Write-Host ""
Write-Host "Exit code: $code" -ForegroundColor $(if ($code -eq 0) { 'Green' } else { 'Red' })
