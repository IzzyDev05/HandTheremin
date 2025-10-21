@echo off
echo =======================================
echo   Setting up HandTheremin environment
echo =======================================
echo.

:: Optional: upgrade pip
python -m pip install --upgrade pip

:: Install all required packages from requirements.txt
echo Installing dependencies...
pip install -r requirements.txt

echo.
echo =======================================
echo   Setup complete!
echo =======================================
pause
