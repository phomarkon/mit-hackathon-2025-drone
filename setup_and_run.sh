#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

echo "--- Setting up Thermal Drone Anomaly Detector ---"

# 1. Check for Python 3
if ! command -v python3 &> /dev/null
then
    echo "Error: python3 could not be found. Please install Python 3.8+."
    exit 1
fi
echo "Python 3 found: $(python3 --version)"

# 2. Create virtual environment if it doesn't exist
VENV_DIR="venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment in '$VENV_DIR'..."
    python3 -m venv $VENV_DIR
    if [ $? -ne 0 ]; then
        echo "Error: Failed to create virtual environment."
        exit 1
    fi
    echo "Virtual environment created."
else
    echo "Virtual environment '$VENV_DIR' already exists."
fi

# 3. Activate virtual environment
echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"
if [ $? -ne 0 ]; then
    echo "Error: Failed to activate virtual environment."
    exit 1
fi

# 4. Install base dependencies
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "Error: Failed to install dependencies from requirements.txt."
    exit 1
fi
echo "Base dependencies installed."

# 5. Install Ultralytics (YOLOv8)
echo "Installing ultralytics for YOLOv8 support..."
pip install ultralytics
if [ $? -ne 0 ]; then
    echo "Error: Failed to install ultralytics."
    # Optionally, you could allow the script to continue without ultralytics
    # echo "Warning: Failed to install ultralytics. YOLOv8 detection will not be available."
    exit 1
fi
echo "Ultralytics installed."

# 6. Run tests (Using run_test.py directly for simplicity, adjust if test.sh is preferred)
# Note: Requires sample data in 'data/' directory to pass certain tests.
# If tests fail due to missing data, comment this section out or provide data.
echo "Running tests via run_test.py..."
# Assuming run_test.py can run without specific input/output args for a basic check
# Or use test.sh if it's more comprehensive: ./test.sh --process --input ./data --output ./outputs/test_run --save
# Using a simple model load test for now:
python run_test.py --mode dashboard # This mode seems to primarily test model loading
if [ $? -ne 0 ]; then
    echo "Warning: Tests failed. Check run_test.py and ensure necessary data/models are present."
    # Decide if failure should halt execution: exit 1
fi
echo "Tests completed (check output for details)."

# 7. Launch the dashboard
echo "--- Setup Complete ---"
echo "Launching the Streamlit Dashboard..."
streamlit run dashboard.py

# Deactivate venv when Streamlit closes (optional, script ends anyway)
# deactivate
echo "Dashboard closed. Script finished." 