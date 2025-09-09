#!/bin/bash
set -euo pipefail

# Step: Repository Preparation
echo "[$(date)] Starting repository preparation..."

# Validate inputs
if [ ! -f "inputs/paper_metadata.json" ]; then
    echo "Error: Missing paper metadata from discovery step"
    exit 1
fi

if [ ! -f "inputs/github_repositories.json" ]; then
    echo "Error: Missing GitHub repositories from discovery step"
    exit 1
fi

# Create default selection criteria if not provided
if [ ! -f "inputs/selection_criteria.json" ]; then
    echo "Creating default selection criteria..."
    cat > inputs/selection_criteria.json << EOF
{
  "max_repositories": 5,
  "min_stars": 10,
  "max_age_days": 1095,
  "required_language": "Python",
  "min_relevance_score": 0.3
}
EOF
fi

# Run repository preparation
cd code
python repository_preparation.py

# Validate outputs
if [ ! -f "../outputs/selected_repositories.json" ]; then
    echo "Error: Selected repositories not generated"
    exit 1
fi

if [ ! -f "../outputs/selected_papers.json" ]; then
    echo "Error: Selected papers not generated"
    exit 1
fi

echo "[$(date)] Repository preparation completed successfully"