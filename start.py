import os
import sys
import subprocess
import time
from pathlib import Path


def main():
    print("=" * 70)
    print("                     STARTING TOKENWISE PLATFORM")
    print("=" * 70)

    project_root = Path(__file__).resolve().parent
    venv_python = project_root / "venv" / "Scripts" / "python.exe"
    if not venv_python.exists():
        venv_python = sys.executable

    print("\n[1/2] Starting FastAPI Backend (uvicorn api.app:app on port 8000)...")
    backend_cmd = [
        str(venv_python),
        "-m", "uvicorn",
        "api.app:app",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--reload"
    ]
    backend_proc = subprocess.Popen(backend_cmd, cwd=str(project_root))

    time.sleep(2)

    print("[2/2] Starting React/Vite Frontend (npm run dev on port 5173)...")
    frontend_dir = project_root / "frontend"
    frontend_cmd = "npm run dev"
    frontend_proc = subprocess.Popen(frontend_cmd, cwd=str(frontend_dir), shell=True)

    print("\n" + "=" * 70)
    print("              TOKENWISE PLATFORM IS UP AND RUNNING!")
    print("=" * 70)
    print(" -> Frontend Dashboard: http://localhost:5173")
    print(" -> Backend API Docs:   http://localhost:8000/docs")
    print(" Press Ctrl+C to stop both servers.\n")

    try:
        backend_proc.wait()
        frontend_proc.wait()
    except KeyboardInterrupt:
        print("\nShutting down TokenWise services...")
        backend_proc.terminate()
        frontend_proc.terminate()
        print("Done.")


if __name__ == "__main__":
    main()
