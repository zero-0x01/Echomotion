import subprocess
import os
import threading
import sys
import pkg_resources

def check_python_dependencies():
    """Check if Python dependencies are installed."""
    try:
        # Attempt to import a key package (e.g., numpy) from requirements.txt to check if dependencies are installed
        pkg_resources.get_distribution("numpy")  # Replace "numpy" with any package from your requirements.txt
        print("Python dependencies are already installed.")
        return True
    except pkg_resources.DistributionNotFound:
        print("Python dependencies not found. Installing...")
        return False

def install_requirements():
    """Install both Python and Node.js dependencies only if not installed already."""
    # Install Python dependencies only if they're not installed
    if not check_python_dependencies():
        print("Installing Python dependencies...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

    # Install Node.js dependencies only if not already installed
    if not os.path.exists("Frontend/node_modules"):
        print("Installing Node.js dependencies...")
        subprocess.check_call(["npm", "install"], cwd="Frontend")  # Ensure the working directory is `Frontend`
    else:
        print("Node.js dependencies already installed. Skipping installation.")

def start_backend():
    print("Starting Backend...")
    backend_path = os.path.join("./", "app.py")
    return subprocess.Popen(["python", backend_path], cwd="Backend")  # Ensure the working directory is `Backend`

def start_frontend():
    print("Starting Frontend...")
    frontend_path = os.path.join("./", "server.js")
    return subprocess.Popen(["node", frontend_path], cwd="Frontend")  # Ensure the working directory is `Frontend`

def monitor_process(name, process):
    try:
        process.wait()  # Wait for the process to complete
    except Exception as e:
        print(f"{name} encountered an error: {e}")

def main():
    try:
        # Install dependencies if not already installed
        install_requirements()

        # Start backend and frontend processes
        backend_process = start_backend()
        frontend_process = start_frontend()

        # Monitor both processes in separate threads
        backend_thread = threading.Thread(target=monitor_process, args=("Backend", backend_process))
        frontend_thread = threading.Thread(target=monitor_process, args=("Frontend", frontend_process))

        backend_thread.start()
        frontend_thread.start()

        print("Both Backend and Frontend are running. Press Ctrl+C to stop.")

        # Wait for manual interruption
        backend_thread.join()
        frontend_thread.join()

    except KeyboardInterrupt:
        print("\nStopping Backend and Frontend...")
        backend_process.terminate()
        frontend_process.terminate()
    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        backend_process.terminate()
        frontend_process.terminate()

if __name__ == "__main__":
    main()
