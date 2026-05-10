import subprocess
from pathlib import Path

def run_test(test_name: str, input_file: str):
    """Runs a full transcription and diarization cycle on a specific test file."""
    print(f"\n==============================================")
    print(f"RUNNING TEST CASE: {test_name}")
    print("==============================================")
    
    # Ensure script runs from the granite-diarization directory
    script_path = Path(__file__).parent / "server.py"
    input_path = Path(input_file)
    
    if not input_path.exists():
        print(f"SKIPPING: Input file not found at {input_path}")
        return

    # Execute the core logic using subprocess for reliable execution environment simulation.
    try:
        # We run the script from the granite-diarization folder
        result = subprocess.run(['python', str(script_path), str(input_path)], 
                                capture_output=True, text=True, check=True, 
                                cwd=str(Path(__file__).parent))
        print("--- STDOUT ---")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"TEST FAILED for {test_name}: Command exited with error code {e.returncode}.")
        print("--- STDERR ---")
        print(e.stderr)
    except FileNotFoundError:
         print("ERROR: Python interpreter or script not found.")

if __name__ == "__main__":
    # Use the provided test file from the parakeet tool folder
    TEST_FILE = r"C:\Users\gerge\source\repos\ASR_windows\parakeet\test.mkv"
    
    print("--- Starting Granite Diarization Validation Tests ---")

    # System Test: Video Input Handling (Verifying file access and pipeline initiation)
    run_test("System Test (MKV Video Input)", TEST_FILE)
    
    print("\n==============================================")
    print("All test cases executed.")