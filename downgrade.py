import subprocess

# Uninstall libraries specified in requirements.txt
with open('req.txt', 'r') as requirements_file:
    libraries_to_uninstall = [line.strip() for line in requirements_file]

uninstall_errors = []

for library in libraries_to_uninstall:
    try:
        subprocess.run(["pip", "uninstall", "-y", library], check=True)
    except subprocess.CalledProcessError:
        uninstall_errors.append(library)

# Install libraries from requirements.txt
install_errors = []

try:
    subprocess.run(["pip", "install", "-r", "req.txt"], check=True)
except subprocess.CalledProcessError:
    install_errors.append("req.txt")

# Print any errors
if uninstall_errors:
    print("Errors occurred while uninstalling the following libraries:")
    for library in uninstall_errors:
        print(library)

if install_errors:
    print("Error occurred while installing libraries from req.txt.")

