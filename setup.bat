@echo off
echo 🔧 Setting up Policy Document AI...

REM Create virtual environment if it doesn't exist
if not exist ".venv" (
    echo 📦 Creating virtual environment...
    python -m venv .venv
)

REM Activate virtual environment
echo 🔌 Activating virtual environment...
call .venv\Scripts\activate.bat

REM Install requirements
echo 📚 Installing requirements...
pip install --upgrade pip
pip install -r requirements.txt

REM Create directories
echo 📁 Creating directories...
mkdir data\documents 2>nul
mkdir data\vectors 2>nul
mkdir data\cache 2>nul
mkdir data\logs 2>nul

REM Check .env file
if not exist ".env" (
    echo ⚠️ .env file not found. Please create it with your API keys.
    echo 💡 See .env.example for reference.
) else (
    echo ✅ .env file found.
)

echo.
echo 🎉 Setup complete! 
echo.
echo 📝 Next steps:
echo 1. Make sure your .env file has the correct API keys
echo 2. Start the API server: start-api.bat
echo 3. Start the frontend: start-streamlit.bat
echo.
pause