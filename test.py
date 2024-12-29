import os

script_path = "classification.py"

try:
    exit_code = os.system(f"python {script_path}")
    print(f"Script finished with exit code: {exit_code}")
except Exception as e:
    print(f"Error occurred: {e}")