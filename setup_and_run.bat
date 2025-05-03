@echo off
setlocal enabledelayedexpansion

echo --- Setting up Thermal Drone Anomaly Detector ---

:: 1. Check for Python 3
python --version > nul 2>&1
if %errorlevel% neq 0 (
    echo Error: python could not be found. Please install Python 3.8+ and add it to PATH.
    goto :eof
)
echo Python found.

:: 2. Create virtual environment if it doesn't exist
set VENV_DIR=venv
if not exist "%VENV_DIR%\Scripts\activate.bat" (
    echo Creating virtual environment in '%VENV_DIR%'...
    python -m venv %VENV_DIR%
    if !errorlevel! neq 0 (
        echo Error: Failed to create virtual environment.
        goto :eof
    )
    echo Virtual environment created.
) else (
    echo Virtual environment '%VENV_DIR%' already exists.
)

:: 3. Activate virtual environment
echo Activating virtual environment...
call "%VENV_DIR%\Scripts\activate.bat"
if !errorlevel! neq 0 (
    echo Error: Failed to activate virtual environment.
    goto :eof
)

:: 4. Install base dependencies
echo Installing dependencies from requirements.txt...
pip install -r requirements.txt
if !errorlevel! neq 0 (
    echo Error: Failed to install dependencies from requirements.txt.
    goto :eof
)
echo Base dependencies installed.

:: 5. Install Ultralytics (YOLOv8)
echo Installing ultralytics for YOLOv8 support...
pip install ultralytics
if !errorlevel! neq 0 (
    echo Error: Failed to install ultralytics.
    rem Optionally, allow continuation:
    rem echo Warning: Failed to install ultralytics. YOLOv8 detection will not be available.
    goto :eof
)
echo Ultralytics installed.

:: 6. Run tests (Using run_test.py directly)
rem Note: Requires sample data in 'data/' directory.
rem If tests fail due to missing data, comment this section out or provide data.
echo Running tests via run_test.py...
rem Using a simple model load test:
python run_test.py --mode dashboard
if !errorlevel! neq 0 (
    echo Warning: Tests failed. Check run_test.py and ensure necessary data/models are present.
    rem Decide if failure should halt execution: goto :eof
)
echo Tests completed (check output for details).

:: 7. Launch the dashboard
echo --- Setup Complete ---
echo Launching the Streamlit Dashboard...
streamlit run dashboard.py

echo Dashboard closed. Script finished.

endlocal 