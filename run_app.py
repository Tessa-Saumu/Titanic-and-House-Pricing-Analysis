"""
Orchestration Script.
Boots both the FastAPI Backend and Streamlit Frontend concurrently.
"""

import subprocess
import sys
import time

def main():
    print("Booting FastAPI Backend on Port 8000...")
    # Start FastAPI
    api_process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.api.main:app", "--host", "127.0.0.1", "--port", "8000"],
        stdout=sys.stdout,
        stderr=sys.stderr
    )
    
    # Wait for the API to initialize before booting frontend
    time.sleep(3)
    
    print("Booting Streamlit Frontend on Port 8501...")
    # Start Streamlit
    ui_process = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "app/ui/app.py"],
        stdout=sys.stdout,
        stderr=sys.stderr
    )
    
    try:
        # Keep main thread alive while subprocesses run
        api_process.wait()
        ui_process.wait()
    except KeyboardInterrupt:
        print("\nShutting down servers gracefully...")
        api_process.terminate()
        ui_process.terminate()
        print("Goodbye!")

if __name__ == "__main__":
    main()