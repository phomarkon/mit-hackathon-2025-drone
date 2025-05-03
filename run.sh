#!/bin/bash

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Run Streamlit dashboard
echo "Starting Thermal Drone Anomaly Detection Dashboard..."
streamlit run dashboard.py 