import os
import sys
import re
import argparse
import subprocess

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BLUE = "\033[94m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

ANSI_REGEX = re.compile(r"\x1b\[[0-9;]*m")

def visible_len(s: str) -> int:
    return len(ANSI_REGEX.sub("", s))

def print_row(content: str, width: int = 72):
    vlen = visible_len(content)
    pad = max(width - 2 - vlen, 0)
    print(f"{BOLD}{CYAN}│{RESET}{content}{' ' * pad}{BOLD}{CYAN}│{RESET}", flush=True)

def main():
    parser = argparse.ArgumentParser(description="PneumoniaCXR Virtual Environment & Dependency Setup")
    parser.add_argument("--server", action="store_true", help="Install server dependencies only")
    parser.add_argument("--client", action="store_true", help="Install client dependencies only")
    args = parser.parse_args()

    install_server = not args.client or args.server
    install_client = not args.server or args.client
    if args.server and not args.client:
        install_client = False
    elif args.client and not args.server:
        install_server = False

    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    venv_dir = os.path.join(root_dir, ".venv")
    
    if sys.platform == "win32":
        venv_python = os.path.join(venv_dir, "Scripts", "python.exe")
    else:
        venv_python = os.path.join(venv_dir, "bin", "python")

    W = 72
    print(f"\n{BOLD}{CYAN}╔" + "═" * (W - 2) + f"╗{RESET}", flush=True)
    title_text = "PneumoniaCXR  --  Environment & Dependency Setup"
    pad_title = (W - 2 - len(title_text)) // 2
    extra = (W - 2 - len(title_text)) % 2
    print(f"{BOLD}{CYAN}║{RESET}{' ' * pad_title}{BOLD}{title_text}{RESET}{' ' * (pad_title + extra)}{BOLD}{CYAN}║{RESET}", flush=True)
    print(f"{BOLD}{CYAN}╚" + "═" * (W - 2) + f"╝{RESET}\n", flush=True)

    # 1. Virtual Environment Setup
    if not os.path.exists(venv_python):
        print(f"  {YELLOW}[1/3]{RESET} Creating virtual environment at .venv...", flush=True)
        try:
            subprocess.run([sys.executable, "-m", "venv", venv_dir], check=True, cwd=root_dir)
            print(f"  {GREEN}[OK] [1/3]{RESET} Virtual environment created successfully.\n", flush=True)
        except Exception as e:
            print(f"  {RED}[ERROR]{RESET} Failed to create virtual environment: {e}\n", flush=True)
            sys.exit(1)
    else:
        print(f"  {GREEN}[OK] [1/3]{RESET} Existing virtual environment detected at .venv\n", flush=True)

    # 2. Server Dependencies
    if install_server:
        server_req = os.path.join(root_dir, "server", "requirements.txt")
        if os.path.exists(server_req):
            print(f"  {YELLOW}[2/3]{RESET} Installing backend dependencies (server/requirements.txt)...", flush=True)
            try:
                subprocess.run([venv_python, "-m", "pip", "install", "-r", server_req], check=True, cwd=root_dir)
                print(f"  {GREEN}[OK] [2/3]{RESET} Backend dependencies installed.\n", flush=True)
            except Exception as e:
                print(f"  {RED}[ERROR]{RESET} Failed to install backend dependencies: {e}\n", flush=True)
                sys.exit(1)
    else:
        print(f"  {DIM}[SKIP] [2/3] Backend dependencies installation skipped.{RESET}\n", flush=True)

    # 3. Client Dependencies
    if install_client:
        client_req = os.path.join(root_dir, "client", "requirements.txt")
        if os.path.exists(client_req):
            print(f"  {YELLOW}[3/3]{RESET} Installing frontend dependencies (client/requirements.txt)...", flush=True)
            try:
                subprocess.run([venv_python, "-m", "pip", "install", "-r", client_req], check=True, cwd=root_dir)
                print(f"  {GREEN}[OK] [3/3]{RESET} Frontend dependencies installed.\n", flush=True)
            except Exception as e:
                print(f"  {RED}[ERROR]{RESET} Failed to install frontend dependencies: {e}\n", flush=True)
                sys.exit(1)
    else:
        print(f"  {DIM}[SKIP] [3/3] Frontend dependencies installation skipped.{RESET}\n", flush=True)

    # Summary Box
    print(f"{BOLD}{CYAN}╭" + "─" * (W - 2) + f"╮{RESET}", flush=True)
    hdr = f"  {BOLD}ENVIRONMENT SETUP COMPLETE{RESET}"
    h_pad = (W - 2 - visible_len(hdr)) // 2
    h_rem = (W - 2 - visible_len(hdr)) % 2
    print(f"{BOLD}{CYAN}│{RESET}{' ' * h_pad}{hdr}{' ' * (h_pad + h_rem)}{BOLD}{CYAN}│{RESET}", flush=True)
    print(f"{BOLD}{CYAN}├" + "─" * (W - 2) + f"┤{RESET}", flush=True)
    print_row("", W)
    print_row(f"   {GREEN}Virtual Environment:{RESET}  .venv", W)
    print_row(f"   {GREEN}Python Interpreter:{RESET}   {venv_python}", W)
    print_row("", W)
    print_row(f"   {BOLD}Next Steps:{RESET}", W)
    print_row(f"      - Start Application:   {CYAN}npm run start{RESET}", W)
    print_row(f"      - Run Unit Tests:      {CYAN}npm run test{RESET}", W)
    print_row(f"      - Run Model Training:  {CYAN}npm run train{RESET}", W)
    print_row("", W)
    print_row(f"   {DIM}Note: All npm scripts automatically use .venv directly.{RESET}", W)
    print(f"{BOLD}{CYAN}╰" + "─" * (W - 2) + f"╯{RESET}\n", flush=True)

if __name__ == "__main__":
    main()
