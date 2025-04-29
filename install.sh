#!/bin/bash
# Install the BOMAX package in development mode

# Check if pip is available
if ! command -v pip &> /dev/null; then
    echo "Error: pip is not installed or not in PATH"
    exit 1
fi

# Install the package in development mode
echo "Installing BOMAX in development mode..."
pip install -e .

# Verify the installation
echo -e "\nVerifying installation..."
python verify_package.py

# If the verification was successful, print a success message
if [ $? -eq 0 ]; then
    echo -e "\nInstallation successful!"
    echo "You can now run the examples:"
    echo "  cd examples"
    echo "  python demo.py"
else
    echo -e "\nInstallation verification failed."
    echo "Please check the error messages above."
fi
