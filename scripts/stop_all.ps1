$Root = Split-Path -Parent $PSScriptRoot

Write-Host "Stopping Docker Compose monitoring services..." -ForegroundColor Cyan

if (Test-Path "$Root\docker-compose.test.yml") {
    docker compose -f "$Root\docker-compose.test.yml" down 2>$null
}
if (Test-Path "$Root\docker-compose.live.yml") {
    docker compose -f "$Root\docker-compose.live.yml" down 2>$null
}
if (Test-Path "$Root\docker-compose.yml") {
    docker compose -f "$Root\docker-compose.yml" down 2>$null
}

Write-Host "All monitoring services stopped." -ForegroundColor Green
