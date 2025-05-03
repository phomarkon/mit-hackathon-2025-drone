@echo off
echo Thermal Drone Anomaly Detection Dashboard

:: Check if virtual environment exists and activate it
if exist venv\Scripts\activate.bat (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
)

:: Run the Streamlit dashboard
echo Starting the application...
streamlit run dashboard.py

pause 