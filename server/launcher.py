import os
import sys
import re
import time
import socket
import signal
import subprocess
import atexit

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ANSI Color Codes
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

ANSI_REGEX = re.compile(r"\x1b\[[0-9;]*m")

def visible_len(s: str) -> int:
    """Calculate the visible character length of a string ignoring ANSI color codes."""
    return len(ANSI_REGEX.sub("", s))

def print_row(content: str, width: int = 72):
    """Print a box row with exact right-border alignment."""
    vlen = visible_len(content)
    pad = max(width - 2 - vlen, 0)
    print(f"{BOLD}{CYAN}│{RESET}{content}{' ' * pad}{BOLD}{CYAN}│{RESET}", flush=True)

SERVER_PROC = None
CLIENT_PROC = None

def get_network_ip() -> str:
    """Detect LAN IPv4 address."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        try:
            return socket.gethostbyname(socket.gethostname())
        except Exception:
            return ""

def cleanup():
    """Terminate child processes cleanly on exit."""
    global SERVER_PROC, CLIENT_PROC
    for proc in [SERVER_PROC, CLIENT_PROC]:
        if proc and proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=1.5)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass

atexit.register(cleanup)

def signal_handler(sig, frame):
    print(f"\n\n  {YELLOW}[STOP] Shutting down PneumoniaCXR services...{RESET}", flush=True)
    cleanup()
    print(f"  {GREEN}[OK] All services stopped cleanly.{RESET}\n", flush=True)
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def is_port_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.5):
            return True
    except (OSError, ConnectionRefusedError):
        return False

def wait_for_port(port: int, timeout: float = 25.0) -> bool:
    start = time.time()
    while time.time() - start < timeout:
        if is_port_open(port):
            return True
        time.sleep(0.4)
    return False

def main():
    global SERVER_PROC, CLIENT_PROC

    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    venv_python = (
        os.path.join(root_dir, ".venv", "Scripts", "python.exe")
        if sys.platform == "win32"
        else os.path.join(root_dir, ".venv", "bin", "python")
    )
    if os.path.exists(venv_python) and os.path.abspath(sys.executable).lower() != os.path.abspath(venv_python).lower():
        sys.exit(subprocess.call([venv_python] + sys.argv, cwd=root_dir))

    server_dir = os.path.join(root_dir, "server")
    client_app = os.path.join(root_dir, "client", "streamlit_app.py")

    banner_width = 72
    print(f"\n{BOLD}{CYAN}╔" + "═" * (banner_width - 2) + f"╗{RESET}", flush=True)
    title_text = "PneumoniaCXR Orchestrator  --  Initializing Services"
    pad_title = (banner_width - 2 - len(title_text)) // 2
    extra = (banner_width - 2 - len(title_text)) % 2
    print(f"{BOLD}{CYAN}║{RESET}{' ' * pad_title}{BOLD}{title_text}{RESET}{' ' * (pad_title + extra)}{BOLD}{CYAN}║{RESET}", flush=True)
    print(f"{BOLD}{CYAN}╚" + "═" * (banner_width - 2) + f"╝{RESET}\n", flush=True)

    # 1. Start Backend Server
    print(f"  {YELLOW}[1/2]{RESET} Starting FastAPI backend on port 8000...", end="", flush=True)

    server_cmd = [
        sys.executable, "-m", "uvicorn",
        "app.service.api:app",
        "--app-dir", server_dir,
        "--host", "0.0.0.0",
        "--port", "8000",
    ]

    SERVER_PROC = subprocess.Popen(
        server_cmd,
        cwd=root_dir,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    if wait_for_port(8000, timeout=15):
        print(f"\r  {GREEN}[OK] [1/2]{RESET} FastAPI backend running on port 8000        ", flush=True)
    else:
        print(f"\r  {YELLOW}[WARN] [1/2]{RESET} FastAPI backend starting in background...  ", flush=True)

    # 2. Start Frontend Client (headless = true so it does NOT auto-open browser)
    print(f"  {YELLOW}[2/2]{RESET} Starting Streamlit frontend on port 8501...", end="", flush=True)

    client_cmd = [
        sys.executable, "-m", "streamlit", "run",
        client_app,
        "--server.port", "8501",
        "--server.headless", "true",
    ]

    CLIENT_PROC = subprocess.Popen(
        client_cmd,
        cwd=root_dir,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    if wait_for_port(8501, timeout=20):
        print(f"\r  {GREEN}[OK] [2/2]{RESET} Streamlit frontend running on port 8501      \n", flush=True)
    else:
        print(f"\r  {YELLOW}[WARN] [2/2]{RESET} Streamlit frontend starting in background...  \n", flush=True)

    net_ip = get_network_ip()

    # Formatted Terminal Dashboard Box (width = 72)
    W = 72
    print(f"{BOLD}{CYAN}╭" + "─" * (W - 2) + f"╮{RESET}", flush=True)
    hdr = f"  {BOLD}PNEUMONIACXR SYSTEM DASHBOARD{RESET}"
    h_pad = (W - 2 - visible_len(hdr)) // 2
    h_rem = (W - 2 - visible_len(hdr)) % 2
    print(f"{BOLD}{CYAN}│{RESET}{' ' * h_pad}{hdr}{' ' * (h_pad + h_rem)}{BOLD}{CYAN}│{RESET}", flush=True)
    print(f"{BOLD}{CYAN}├" + "─" * (W - 2) + f"┤{RESET}", flush=True)
    print_row("", W)

    # Frontend section
    print_row(f"   {BOLD}{GREEN}Web Application (Frontend UI):{RESET}", W)
    print_row(f"      - Local URL:    {BOLD}http://localhost:8501{RESET}", W)
    if net_ip and net_ip != "127.0.0.1":
        print_row(f"      - Network URL:  http://{net_ip}:8501", W)
    print_row("", W)

    # Backend section
    print_row(f"   {BOLD}{BLUE}REST API (Backend Service):{RESET}", W)
    print_row(f"      - Local URL:    http://localhost:8000", W)
    if net_ip and net_ip != "127.0.0.1":
        print_row(f"      - Network URL:  http://{net_ip}:8000", W)
    print_row(f"      - API Docs:     http://localhost:8000/docs", W)
    print_row("", W)

    # Footer
    print(f"{BOLD}{CYAN}├" + "─" * (W - 2) + f"┤{RESET}", flush=True)
    print_row(f"   {DIM}Status: Active & Listening  |  Press Ctrl+C to Stop{RESET}", W)
    print(f"{BOLD}{CYAN}╰" + "─" * (W - 2) + f"╯{RESET}\n", flush=True)

    # Keep alive and monitor child processes
    try:
        while True:
            if SERVER_PROC and SERVER_PROC.poll() is not None:
                print(f"  {YELLOW}[WARN] Backend server process stopped.{RESET}", flush=True)
                break
            if CLIENT_PROC and CLIENT_PROC.poll() is not None:
                print(f"  {YELLOW}[WARN] Frontend client process stopped.{RESET}", flush=True)
                break
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        cleanup()

if __name__ == "__main__":
    main()
