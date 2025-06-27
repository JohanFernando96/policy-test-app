@echo off
echo 🚀 Starting Policy Document AI API Server...

REM Check virtual environment
if not exist ".venv\Scripts\activate.bat" (
    echo ❌ Virtual environment not found. Please run setup first.
    pause
    exit /b 1
)

REM Activate virtual environment
echo 📦 Activating virtual environment...
call .venv\Scripts\activate.bat

REM Check .env file
if not exist ".env" (
    echo ❌ .env file not found. Please create it first.
    pause
    exit /b 1
)

REM Load environment variables
echo ⚙️ Loading environment variables...
for /f "usebackq tokens=1,2 delims==" %%A in (".env") do (
    set "%%A=%%B"
)

REM Start API server
echo 🌐 Starting FastAPI server on http://localhost:8000...
echo 📖 API documentation: http://localhost:8000/docs
echo 🛑 Press Ctrl+C to stop
echo.

uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
pause