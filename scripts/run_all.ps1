# Run-All Orchestrator Script for IDS, Docker, Grafana, and Prometheus
param (
    [switch]$Reset,
    [int]$Workers = 4
)

# Resolve absolute paths based on script location (works from any current working directory)
$ScriptDir   = $PSScriptRoot
$ProjectRoot = Split-Path -Parent $ScriptDir
$PythonPath  = Join-Path $ProjectRoot "venv\Scripts\python.exe"
$MainPy      = Join-Path $ProjectRoot "src\main.py"
$ComposeFile = Join-Path $ProjectRoot "docker-compose.yml"
$DataDir     = Join-Path $ProjectRoot "data"
$CachePath   = Join-Path $DataDir "interface_preference.txt"

Write-Host "==========================================================" -ForegroundColor Green
Write-Host " [IDS Orchestrator Stack Launcher]" -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Green

# 1. Verify Virtual Environment
if (-not (Test-Path $PythonPath)) {
    Write-Host "[Error] Virtual environment python executable not found at: $PythonPath" -ForegroundColor Red
    Write-Host "Please set up your virtual environment first:" -ForegroundColor Yellow
    Write-Host "  python -m venv venv"
    Write-Host "  venv\Scripts\pip.exe install -r requirements.txt"
    Exit 1
}
Write-Host "[OK] Python Virtual Environment verified." -ForegroundColor Cyan

# 2. Verify and Start Docker Daemon
Write-Host "Checking Docker status..." -ForegroundColor Cyan
& docker info >$null 2>&1
if ($LastExitCode -ne 0) {
    Write-Host "[Warning] Docker Daemon is not running." -ForegroundColor Yellow
    
    $DockerPath = "C:\Program Files\Docker\Docker\Docker Desktop.exe"
    if (Test-Path $DockerPath) {
        Write-Host "Attempting to start Docker Desktop..." -ForegroundColor Cyan
        Start-Process $DockerPath
        
        # Poll docker info for up to 45 seconds (15 iterations * 3 seconds)
        $dockerReady = $false
        for ($i = 1; $i -le 15; $i++) {
            Start-Sleep -Seconds 3
            & docker info >$null 2>&1
            if ($LastExitCode -eq 0) {
                $dockerReady = $true
                break
            }
            Write-Host "   Waiting for Docker daemon ($($i * 3)s / 45s)..." -ForegroundColor Gray
        }
        
        if (-not $dockerReady) {
            Write-Host "[Error] Docker daemon failed to start within 45 seconds." -ForegroundColor Red
            Write-Host "Please open Docker Desktop manually and try again." -ForegroundColor Yellow
            Exit 1
        }
    } else {
        Write-Host "[Error] Docker Desktop executable not found at: $DockerPath" -ForegroundColor Red
        Write-Host "Please start your Docker daemon manually and try again." -ForegroundColor Yellow
        Exit 1
    }
}
Write-Host "[OK] Docker Daemon is active." -ForegroundColor Cyan

# 3. Handle Interface Cache / Reset
if ($Reset -and (Test-Path $CachePath)) {
    Remove-Item $CachePath -Force
    Write-Host "[Reset] Interface preference cache cleared." -ForegroundColor Yellow
}

$InterfaceSelected = ""
if (Test-Path $CachePath) {
    $CachedValue = (Get-Content $CachePath -Raw).Trim()
    if (-not [string]::IsNullOrWhiteSpace($CachedValue)) {
        $InterfaceSelected = $CachedValue
        Write-Host "[Preference] Using cached network interface: $InterfaceSelected" -ForegroundColor Green
    }
}

# 4. Resolve Interface (If not cached)
if ([string]::IsNullOrWhiteSpace($InterfaceSelected)) {
    Write-Host "Querying available network interfaces..." -ForegroundColor Cyan
    
    # Run main.py with --list-interfaces to find Scapy/Npcap interfaces
    $env:PYTHONPATH = Join-Path $ProjectRoot "src"
    $output = & $PythonPath $MainPy --list-interfaces 2>$null
    
    $interfaces = @()
    foreach ($line in $output) {
        $lineTrimmed = $line.Trim()
        # Grab lines that look like actual adapter names/GUIDs
        if ($lineTrimmed -match '^(\\Device\\NPF_.+)$' -or ($lineTrimmed -and $lineTrimmed -ne "Available network interfaces:" -and -not ($lineTrimmed.StartsWith("Scapy")))) {
            $interfaces += $lineTrimmed
        }
    }

    if ($interfaces.Count -eq 0) {
        Write-Host "[Warning] No network interfaces enumerated. Defaulting to fallback interface 'Ethernet'." -ForegroundColor Yellow
        $InterfaceSelected = "Ethernet"
    } else {
        Write-Host ""
        Write-Host "Select network interface to capture on:" -ForegroundColor Green
        for ($i = 0; $i -lt $interfaces.Count; $i++) {
            Write-Host "  [$($i + 1)] $($interfaces[$i])"
        }
        Write-Host "  [A] Auto-Detect / Default Interface Fallback"
        Write-Host ""

        $selection = Read-Host "Enter selection [1-$($interfaces.Count) or A (default)]"
        $selection = $selection.Trim()
        
        if ($selection -eq 'A' -or $selection -eq 'a' -or [string]::IsNullOrWhiteSpace($selection)) {
            Write-Host "Auto-detect fallback selected." -ForegroundColor Yellow
            $InterfaceSelected = "Ethernet"
        } else {
            if ($selection -match '^\d+$') {
                $idx = [int]$selection - 1
                if ($idx -ge 0 -and $idx -lt $interfaces.Count) {
                    $InterfaceSelected = $interfaces[$idx]
                }
            }
            
            if ([string]::IsNullOrWhiteSpace($InterfaceSelected)) {
                Write-Host "[Warning] Invalid selection. Defaulting to Ethernet fallback." -ForegroundColor Red
                $InterfaceSelected = "Ethernet"
            }
        }

        # Cache the selection if it is a specific adapter and not the auto-detect placeholder
        if ($InterfaceSelected -ne "Ethernet") {
            if (-not (Test-Path $DataDir)) {
                New-Item -ItemType Directory -Path $DataDir -Force | Out-Null
            }
            New-Item -ItemType File -Path $CachePath -Force -Value $InterfaceSelected | Out-Null
            Write-Host "Saved selection to $CachePath" -ForegroundColor Gray
        }
    }
}

# 5. Core Execution Loop with Ctrl+C Cleanup Guard
try {
    Write-Host "Starting Docker Compose stack (Prometheus, Alertmanager & Grafana)..." -ForegroundColor Cyan
    docker compose -f $ComposeFile up -d
    if ($LastExitCode -ne 0) {
        Write-Host "[Error] Failed to spin up Docker Compose containers." -ForegroundColor Red
        Exit 1
    }
    
    Write-Host "Launching IDS Real-Time Application..." -ForegroundColor Green
    Write-Host "   Capture Interface: $InterfaceSelected" -ForegroundColor Gray
    Write-Host "   Worker Threads:    $Workers" -ForegroundColor Gray
    Write-Host "   FastAPI Port:      http://localhost:8000" -ForegroundColor Gray
    Write-Host "   Grafana Port:      http://localhost:3000" -ForegroundColor Gray
    Write-Host "   Prometheus Port:   http://localhost:9091" -ForegroundColor Gray
    Write-Host "   Alertmanager Port: http://localhost:9093" -ForegroundColor Gray
    Write-Host "   Press [Ctrl+C] to stop and clean up." -ForegroundColor Yellow
    Write-Host "----------------------------------------------------------" -ForegroundColor Gray

    # Execute Python IDS Process (Blocking call)
    $env:PYTHONPATH = Join-Path $ProjectRoot "src"
    & $PythonPath $MainPy --interface $InterfaceSelected --workers $Workers
}
finally {
    Write-Host "----------------------------------------------------------" -ForegroundColor Gray
    Write-Host "Interruption captured. Cleaning up stack..." -ForegroundColor Yellow
    
    Write-Host "Shutting down Docker Compose containers..." -ForegroundColor Cyan
    docker compose -f $ComposeFile down
    
    Write-Host "Shutdown sequence complete. Goodbye!" -ForegroundColor Green
    Write-Host "==========================================================" -ForegroundColor Green
}
