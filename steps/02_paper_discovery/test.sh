#!/bin/bash
set -euo pipefail

echo "Testing paper discovery..."

# Create test configuration if not exists
if [ ! -f "inputs/search_config.json" ]; then
    echo "Creating test search configuration..."
    cat > inputs/search_config.json << EOF
{
  "category": "vq",
  "keywords": ["vector quantization"],
  "arxiv_categories": ["cs.LG"],
  "date_limit": "2023-01-01",
  "max_papers": 5,
  "github_search_enabled": true,
  "max_github_repos": 10
}
EOF
fi

# Run discovery
./run.sh

# Validate outputs
python -c "
import json
import sys

try:
    # Check paper metadata
    with open('outputs/paper_metadata.json') as f:
        papers = json.load(f)
    print(f'Found {len(papers)} papers')
    
    # Check GitHub repositories  
    with open('outputs/github_repositories.json') as f:
        repos = json.load(f)
    print(f'Found {len(repos)} repositories')
    
    # Check search summary
    with open('outputs/search_summary.json') as f:
        summary = json.load(f)
    
    required_fields = ['arxiv_papers_found', 'github_repos_found', 'total_papers']
    for field in required_fields:
        if field not in summary:
            print(f'Missing summary field: {field}')
            sys.exit(1)
    
    print('All tests passed')
    
except Exception as e:
    print(f'Test failed: {e}')
    sys.exit(1)
"

echo "Paper discovery tests passed"