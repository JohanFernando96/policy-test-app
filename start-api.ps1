# Policy Document AI - API Server Startup Script
Write-Host "Starting Policy Document AI API Server..." -ForegroundColor Green

# Check if virtual environment exists
if (-not (Test-Path ".venv\Scripts\activate.ps1")) {
    Write-Host "ERROR: Virtual environment not found. Please run setup first." -ForegroundColor Red
    exit 1
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
.\.venv\Scripts\Activate.ps1

# Check if .env file exists
if (-not (Test-Path ".env")) {
    Write-Host "ERROR: .env file not found. Please create it first." -ForegroundColor Red
    exit 1
}

# Load environment variables from .env
Write-Host "Loading environment variables..." -ForegroundColor Yellow
Get-Content .env | ForEach-Object {
    if ($_ -match '^([^=]+)=(.*)$') {
        $name = $matches[1]
        $value = $matches[2]
        [System.Environment]::SetEnvironmentVariable($name, $value, [System.EnvironmentVariableTarget]::Process)
    }
}

# Start the API server
Write-Host "Starting FastAPI server on http://localhost:8000..." -ForegroundColor Green
Write-Host "API documentation will be available at http://localhost:8000/docs" -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host ""

uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload