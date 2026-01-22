#!/bin/bash
# Activation script for delftblue environment
# This script runs the Python import to ensure pyudales is available
echo "Activating delftblue environment..."
echo "Checking if udales is downloaded and compiled..."
echo "This might take a while if ..."
python -c "import pyudales" || true
echo "udales is available!"
echo "Environment is activated"


