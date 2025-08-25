import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

# Set the directory paths
DATA_ORIGIN_FOLDER = "./../data/pcaps/"
DATA_OUTPUT_FOLDER = "./../data/output/"

def check_and_fix_pcap(file_path):
    """Check and fix a .pcap file if it's truncated."""
    try:
        # Construct the valid filename within the DATA_OUTPUT_FOLDER
        valid_file_path = os.path.join(DATA_OUTPUT_FOLDER, f"{os.path.basename(file_path)}")
        
        # Run tcpdump command to check and extract valid packets
        command = ['tcpdump', '-r', file_path, '-w', valid_file_path]
        
        # Run the command, redirecting stderr to suppress warning
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # If tcpdump runs without error (successful conversion)
        if result.returncode == 0:
            return

        # If there was an error, and it's related to truncated data, handle the case
        error_output = result.stderr.decode('utf-8')
        if "truncated dump file" in error_output:
            return

        # In case of other errors, print the error
        print(f"Error processing file '{file_path}': {error_output}")

    except Exception as e:
        print(f"Error processing file '{file_path}': {e}")

def process_pcap_files(directory):
    """Process all .pcap files in the specified directory using parallel execution."""
    # List of .pcap files to process
    pcap_files = [os.path.join(directory, filename) for filename in os.listdir(directory) if filename.endswith('.pcap')]

    # Use ThreadPoolExecutor to process files in parallel
    with ThreadPoolExecutor(max_workers=4) as executor:  # You can adjust max_workers for more parallelism
        # Map check_and_fix_pcap function to each file
        executor.map(check_and_fix_pcap, pcap_files)

if __name__ == "__main__":
    # Ensure the directories exist
    if not os.path.isdir(DATA_ORIGIN_FOLDER):
        print(f"Error: The directory '{DATA_ORIGIN_FOLDER}' does not exist.")
        sys.exit(1)

    if not os.path.isdir(DATA_OUTPUT_FOLDER):
        print(f"Error: The result directory '{DATA_OUTPUT_FOLDER}' does not exist. Creating it now.")
        os.makedirs(DATA_OUTPUT_FOLDER)

    # Process all .pcap files in the directory
    process_pcap_files(DATA_ORIGIN_FOLDER)



