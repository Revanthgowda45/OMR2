@echo off
echo ========================================
echo   OMR Evaluation System - Code4Edtech
echo   Running from OMR_Hackathon folder
echo ========================================
echo.

echo Installing dependencies...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo.
echo Starting OMR Evaluation System...
echo Choose your deployment option:
echo.
echo 1. Streamlit Cloud version (streamlit_app.py)
echo 2. Hugging Face Spaces version (app.py)
echo.

set /p choice="Enter your choice (1 or 2): "

if "%choice%"=="1" (
    echo.
    echo Starting Streamlit Cloud version...
    echo Opening http://localhost:8501
    python -m streamlit run streamlit_app.py
) else if "%choice%"=="2" (
    echo.
    echo Starting Hugging Face Spaces version...
    echo Opening http://localhost:8501
    python -m streamlit run app.py
) else (
    echo.
    echo Invalid choice. Starting default version...
    python -m streamlit run streamlit_app.py
)

echo.
echo App stopped. Press any key to exit.
pause > nul
