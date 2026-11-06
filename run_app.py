"""
StudyAgent Web Application Launcher Script.
Run:
    python run_app.py
"""

import sys
import subprocess
import shutil
import os

def clear_pycache():
    """Purge bytecode cache directories to prevent stale module loading."""
    root_dir = os.path.dirname(os.path.abspath(__file__))
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if "__pycache__" in dirnames:
            pycache_path = os.path.join(dirpath, "__pycache__")
            try:
                shutil.rmtree(pycache_path)
            except Exception:
                pass

def main():
    clear_pycache()
    print("-" * 60)
    print("Initializing StudyAgent Web Application (Python/Streamlit)")
    print("-" * 60)
    print("Launching application server on http://localhost:8501")
    print("-" * 60)
    
    cmd = [sys.executable, "-m", "streamlit", "run", "app.py"]
    subprocess.run(cmd)

if __name__ == "__main__":
    main()
