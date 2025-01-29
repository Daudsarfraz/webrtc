import subprocess

port = 8080  # Port i am using 8080 for the server running 

# Find processes using the specified port
# command = f"sudo lsof -t -i:{port}"
command = f"lsof -t -i:{port}"

try:
    # Get the list of PIDs that are using the given port
    result = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
    pid_list = result.decode('utf-8').strip().split('\n')

    if pid_list:
        # If there are processes, kill them
        for pid in pid_list:
            # subprocess.run(f"sudo kill -9 {pid}", shell=True)
            subprocess.run(f"sudo kill -9 {pid}", shell=True)

        print(f"Killed processes running on port {port}.")
    else:
        # If no processes are found, print a message
        print(f"Nothing is running on port {port}.")
        
except subprocess.CalledProcessError:
    # If the lsof command fails (no processes found), print the error message
    print(f"Nothing is running on port {port}.")
