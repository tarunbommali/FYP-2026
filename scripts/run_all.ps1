param(
    [switch]$Live,
    [int]$Workers = 4,
    [string]$Interface = "Ethernet"
)

$Root = Split-Path -Parent $PSScriptRoot
$Python = "$Root\venv\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    Write-Host "Python virtual environment not found at: $Python" -ForegroundColor Red
    Write-Host "Please set up your venv and install dependencies first." -ForegroundColor Yellow
    exit 1
}

if ($Live) {
    $Compose = "$Root\docker-compose.live.yml"
    if (-not (Test-Path $Compose)) { $Compose = "$Root\docker-compose.yml" }
    Write-Host "Starting LIVE Monitoring Stack ($($Compose | Split-Path -Leaf))..." -ForegroundColor Cyan
} else {
    $Compose = "$Root\docker-compose.test.yml"
    if (-not (Test-Path $Compose)) { $Compose = "$Root\docker-compose.yml" }
    Write-Host "Starting TEST Monitoring Stack ($($Compose | Split-Path -Leaf))..." -ForegroundColor Cyan
}

docker compose -f $Compose up -d

if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to start Docker services. Ensure Docker Desktop is running." -ForegroundColor Red
    exit 1
}

$env:PYTHONPATH = "$Root\src;$Root"

if ($Live) {
    Write-Host "Starting IDS in LIVE mode on '$Interface' with $Workers workers..." -ForegroundColor Green
    & $Python "$Root\src\main.py" --live --interface $Interface --workers $Workers
} else {
    Write-Host "Starting IDS in TEST DATA mode (CICIDS2017 20% Test) with $Workers workers..." -ForegroundColor Green
    & $Python "$Root\src\main.py" --workers $Workers
}
