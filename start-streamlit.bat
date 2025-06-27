@echo off
echo 🎨 Starting Policy Document AI Frontend...

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

REM Start Streamlit
echo 🎯 Starting Streamlit on http://localhost:8501...
echo 🛑 Press Ctrl+C to stop
echo.

streamlit run streamlit_app.py --server.port 8501
pause