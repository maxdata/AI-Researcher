#!/bin/bash
set -euo pipefail

# Step: Paper Discovery
echo "[$(date)] Starting paper discovery..."

# Validate inputs
if [ ! -f "inputs/search_config.json" ]; then
    echo "Error: Missing search configuration file"
    exit 1
fi

# Check network connectivity
if ! ping -c 1 arxiv.org > /dev/null 2>&1; then
    echo "Warning: Cannot reach arxiv.org, ArXiv search may fail"
fi

if ! ping -c 1 github.com > /dev/null 2>&1; then
    echo "Warning: Cannot reach github.com, GitHub search may fail"
fi

# Run paper discovery
cd code
python paper_discovery.py

# Validate outputs
if [ ! -f "../outputs/paper_metadata.json" ]; then
    echo "Error: Paper metadata not generated"
    exit 1
fi

if [ ! -f "../outputs/github_repositories.json" ]; then
    echo "Error: GitHub repositories not generated"
    exit 1
fi

if [ ! -f "../outputs/search_summary.json" ]; then
    echo "Error: Search summary not generated"
    exit 1
fi

# Check if any papers were found
paper_count=$(python -c "import json; print(len(json.load(open('../outputs/paper_metadata.json'))))")
if [ "$paper_count" -eq 0 ]; then
    echo "Warning: No papers found, but continuing pipeline"
fi

echo "[$(date)] Paper discovery completed with $paper_count papers found"