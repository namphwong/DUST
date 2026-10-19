param([string]$PythonPath, [int]$Port=4186, [switch]$CheckOnly)
$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
if (-not $PythonPath) { $PythonPath = Join-Path $projectRoot '.venv-dinov3\Scripts\python.exe' }
$PythonPath = (Resolve-Path -LiteralPath $PythonPath).Path
$webRoot = Join-Path $projectRoot 'web'
if (-not (Test-Path -LiteralPath (Join-Path $webRoot 'node_modules'))) { throw 'Chay npm ci trong web truoc.' }
if (Get-NetTCPConnection -LocalPort 4197 -State Listen -ErrorAction SilentlyContinue) { throw 'Port 4197 dang duoc su dung. Dung backend cu truoc.' }
$backendOut = Join-Path ([System.IO.Path]::GetTempPath()) 'drawai-p7-backend.stdout.log'
$backendErr = Join-Path ([System.IO.Path]::GetTempPath()) 'drawai-p7-backend.stderr.log'
$backend = Start-Process -FilePath $PythonPath -ArgumentList @('-m','training.drawai.paint_app_runtime') -WorkingDirectory $projectRoot -PassThru -WindowStyle Hidden -RedirectStandardOutput $backendOut -RedirectStandardError $backendErr
try {
    $ready=$false
    for ($attempt=0; $attempt -lt 150; $attempt++) {
        if ($backend.HasExited) { throw "Backend dung. Xem $backendErr" }
        try { $health=Invoke-RestMethod -Uri 'http://127.0.0.1:4197/health' -TimeoutSec 1; $ready=$health.status -eq 'ready' } catch { }
        if ($ready) { break }
        Start-Sleep -Milliseconds 100
    }
    if (-not $ready) { throw "Backend chua san sang. Xem $backendErr" }
    if ($CheckOnly) { Write-Host 'DrawAI model/backend startup PASS; backend will stop now.'; return }
    Write-Host "DrawAI: http://127.0.0.1:$Port/ | So sanh: /drawai/compare | Ctrl+C de dung."
    Push-Location -LiteralPath $webRoot
    try { & npm.cmd run dev -- --port $Port --strictPort } finally { Pop-Location }
} finally {
    if (-not $backend.HasExited) { Stop-Process -Id $backend.Id }
}
