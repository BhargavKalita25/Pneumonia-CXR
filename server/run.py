import os
import sys
import subprocess

def main():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    venv_python = (
        os.path.join(root_dir, ".venv", "Scripts", "python.exe")
        if sys.platform == "win32"
        else os.path.join(root_dir, ".venv", "bin", "python")
    )
    python_bin = venv_python if os.path.exists(venv_python) else sys.executable

    cmd = [python_bin] + sys.argv[1:]
    try:
        res = subprocess.call(cmd, cwd=root_dir)
        sys.exit(res)
    except KeyboardInterrupt:
        sys.exit(130)

if __name__ == "__main__":
    main()
