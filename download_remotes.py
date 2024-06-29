import os
import time

# Remote server details
remote_user = "vishravi"
remote_host = "megatron.cs.unc.edu"
remote_path = "/playpen-storage/vishravi/gan/samples/ZERO/"  # Adjust as necessary

# Local folder to save the files
local_folder = "remote_results/"
os.makedirs(local_folder, exist_ok=True)


# Function to generate the file names based on the observed pattern
def generate_file_names():
    batches = [x * 1024 for x in range(600)]
    epochs = [i for i in range(5)]  # Based on the provided file pattern
    file_names = [
        f"epoch{epoch}_batch{batch}.png" for batch in batches for epoch in epochs
    ]
    return file_names


# Generate the file names
files_to_download = generate_file_names()

# Generate and execute the scp commands
for i, file_name in enumerate(files_to_download):
    scp_command = (
        f"scp {remote_user}@{remote_host}:{remote_path}{file_name} {local_folder}"
    )
    os.system(scp_command)
    print(f"Downloaded {file_name}")


print("All files downloaded.")
