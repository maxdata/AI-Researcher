#!/bin/bash
set -euo pipefail

# Step: Environment Setup
echo "[$(date)] Starting environment setup..."

# Validate inputs
if [ ! -f "inputs/config.json" ]; then
    echo "Error: Missing input configuration file"
    exit 1
fi

# Check Docker availability
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed or not in PATH"
    exit 1
fi

# Check if Docker daemon is running
if ! docker info &> /dev/null; then
    echo "Error: Docker daemon is not running"
    exit 1
fi

# Run environment setup
cd code
python environment_setup.py

# Validate outputs
if [ ! -f "../outputs/environment_status.json" ]; then
    echo "Error: Environment status not generated"
    exit 1
fi

if [ ! -f "../outputs/docker_config.json" ]; then
    echo "Error: Docker configuration not generated"
    exit 1
fi

# Check if setup was successful
if ! grep -q '"status": "success"' "../outputs/environment_status.json"; then
    echo "Error: Environment setup failed"
    exit 1
fi

echo "[$(date)] Environment setup completed successfully"