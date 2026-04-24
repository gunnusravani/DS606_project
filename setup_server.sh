#!/bin/bash
# Setup script for DS606 project on server

echo "DS606 Project Setup"
echo "==================="

# Change to project directory
cd /workspace/DS606_project

echo "Step 1: Checking Python version..."
python3 --version

echo ""
echo "Step 2: Installing package in development mode..."
python3 -m pip install -e .

echo ""
echo "Step 3: Testing import..."
python3 -c "from ds606.cli import main; print('✓ ds606 module imported successfully')" && echo "✓ Setup complete!" || echo "✗ Setup failed"

echo ""
echo "Step 4: Now you can run:"
echo "  python3 -m ds606.cli evaluate-with-initial --csv initial_malicious_final.csv --device-map cuda:3"
