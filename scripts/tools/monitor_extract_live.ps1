# Live monitor for ongoing DICOM extract (does not touch the job).
$host.UI.RawUI.WindowTitle = 'ml-trainer EXTRACT LIVE LOG'
$ErrorActionPreference = 'Continue'

Write-Host ''
Write-Host '============================================================' -ForegroundColor Cyan
Write-Host '  ml-trainer EXTRACT LIVE LOG  (fallback: CSV/GPU progress)' -ForegroundColor Cyan
Write-Host '  Method A (deleted log FD) unavailable on WSL/DrvFS' -ForegroundColor DarkYellow
Write-Host '  PIDs: bash=822  tee=826  run_extract=850  extract=851' -ForegroundColor Cyan
Write-Host '  Refresh every 8s. Closing this window only stops watching.' -ForegroundColor Cyan
Write-Host '============================================================' -ForegroundColor Cyan
Write-Host ''

$snap = '/mnt/e/ml/ml-trainer/scripts/tools/monitor_extract_live_snapshot.sh'

while ($true) {
  $stamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
  Write-Host ("----- {0} -----" -f $stamp) -ForegroundColor Green
  wsl bash $snap
  Write-Host ''
  Start-Sleep -Seconds 8
}
