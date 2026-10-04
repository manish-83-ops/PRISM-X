Write-Host "Running PRISMX Pre-Demo Readiness Checklist..."
python scripts/predemo_check.py
if ($LASTEXITCODE -ne 0) {
    Write-Error "Pre-demo readiness checks failed!"
    exit 1
}
