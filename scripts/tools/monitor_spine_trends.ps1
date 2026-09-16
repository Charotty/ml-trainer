# Live monitor for На спине finish + trends (does not touch the job).
$host.UI.RawUI.WindowTitle = 'SPINE+TRENDS'
$ErrorActionPreference = 'Continue'

$logWin = 'E:\ml\ml-trainer\results\wsl_na_spine_finish.log'
$snap = '/mnt/e/ml/ml-trainer/scripts/tools/monitor_spine_trends_snapshot.sh'

Write-Host ''
Write-Host '============================================================' -ForegroundColor Cyan
Write-Host '  SPINE+TRENDS  (tail log + CSV poll ~10s)' -ForegroundColor Cyan
Write-Host "  Log: $logWin" -ForegroundColor Cyan
Write-Host '  Closing this window only stops watching.' -ForegroundColor Cyan
Write-Host '============================================================' -ForegroundColor Cyan
Write-Host ''

while ($true) {
  $stamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
  Write-Host ("----- {0} -----" -f $stamp) -ForegroundColor Green
  wsl -e bash $snap
  Write-Host ''
  Start-Sleep -Seconds 10
}
