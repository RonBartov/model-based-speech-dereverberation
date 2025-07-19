import scipy.io

# Load the .mat file
mat_data = scipy.io.loadmat('doa_shoebox.mat')

# Print the variable names and their contents
for key, value in mat_data.items():
    if not key.startswith('__'):  # Ignore metadata entries like __header__
        print(f"Variable name: {key}")
        print(f"Value:\n{value}\n")
        print("-" * 40)
