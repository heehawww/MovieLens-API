import subprocess
import sys

api = subprocess.Popen(["uv", "run", "uvicorn", "app.main:app", "--reload"])
frontend = subprocess.Popen(["uv", "run", "python", "-m", "streamlit", "run", "app/frontend.py"])

try:
    api.wait()
    frontend.wait()
except KeyboardInterrupt:
    api.terminate()
    frontend.terminate()
    sys.exit(0)