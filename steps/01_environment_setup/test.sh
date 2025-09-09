#!/bin/bash
set -euo pipefail

echo "Testing environment setup..."

# Run with test configuration
if [ ! -f "inputs/config.json" ]; then
    echo "Creating test configuration..."
    cat > inputs/config.json << EOF
{
  "base_image": "tjbtech1/airesearcher:v1",
  "completion_model": "gpt-4o-mini",
  "cheep_model": "gpt-4o-mini",
  "workspace_name": "test_workplace",
  "container_name": "test_eval",
  "port": 12346,
  "gpu_config": "\"device=0\"",
  "category": "vq",
  "task_level": "task1",
  "max_iter_times": 0
}
EOF
fi

# Run the setup
./run.sh

# Validate output structure
python -c "
import json
import sys

try:
    with open('outputs/environment_status.json') as f:
        status = json.load(f)
    
    required_fields = ['status', 'validation_results', 'workspace_info', 'config']
    for field in required_fields:
        if field not in status:
            print(f'Missing field: {field}')
            sys.exit(1)
    
    if status['status'] != 'success':
        print('Setup status is not success')
        sys.exit(1)
        
    with open('outputs/docker_config.json') as f:
        docker_config = json.load(f)
    
    required_docker_fields = ['container_name', 'workplace_name', 'communication_port', 'local_root']
    for field in required_docker_fields:
        if field not in docker_config:
            print(f'Missing Docker config field: {field}')
            sys.exit(1)
    
    print('All tests passed')
    
except Exception as e:
    print(f'Test failed: {e}')
    sys.exit(1)
"

echo "Environment setup tests passed"