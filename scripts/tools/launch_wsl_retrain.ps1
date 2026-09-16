$ErrorActionPreference = 'Continue'
$host.UI.RawUI.WindowTitle = 'ml-trainer RETRAIN clinical honest'
Write-Host '=== Retrain adaptive_ensemble_clinical_honest (na_trends) ===' -ForegroundColor Cyan
Write-Host 'Log: /tmp/ml_trainer_logs/wsl_retrain_clinical_honest.log'
Write-Host 'Tail: wsl -e bash -lc "tail -f /tmp/ml_trainer_logs/wsl_retrain_clinical_honest.log"'
Write-Host ''
wsl -e bash /mnt/e/ml/ml-trainer/scripts/tools/wsl_retrain_clinical_honest.sh
$code = $LASTEXITCODE
Write-Host ""
Write-Host "Exit code: $code" -ForegroundColor $(if ($code -eq 0) { 'Green' } else { 'Red' })
