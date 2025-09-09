# Step 01: Environment Setup

## Overview

This step initializes the Docker-based research environment required for the AI-Researcher pipeline. It validates system dependencies, configures workspace directories, and prepares the containerized environment for subsequent research steps.

## Purpose

- Initialize Docker containers with the AI-Researcher base image
- Validate system requirements (Docker, GPU support)
- Create isolated research workspace with proper volume mounting
- Configure environment variables and API access
- Establish communication channels between host and container

## Input Requirements

### Configuration File (`inputs/config.json`)

```json
{
  "base_image": "tjbtech1/airesearcher:v1",
  "completion_model": "gpt-4o-2024-08-06", 
  "cheep_model": "gpt-4o-mini",
  "workspace_name": "workplace",
  "container_name": "paper_eval",
  "port": 12345,
  "gpu_config": "\"device=0\"",
  "category": "vq",
  "task_level": "task1",
  "max_iter_times": 0
}
```

### System Requirements

- Docker Engine 20.10+
- Docker Compose (optional)
- NVIDIA Docker runtime (for GPU support)
- Python 3.11+
- Minimum 8GB RAM
- 50GB available disk space

## Outputs

1. **Environment Status** (`outputs/environment_status.json`)
   - Docker validation results
   - GPU availability status  
   - Image pull status
   - Workspace creation confirmation

2. **Docker Configuration** (`outputs/docker_config.json`)
   - Container configuration details
   - Port mappings
   - Volume mount points
   - Resource allocations

3. **Workspace Directory** (`outputs/workspace/`)
   - Initialized research workspace
   - Proper permissions and structure

## Implementation Details

### Docker Environment

The step uses the official AI-Researcher Docker image `tjbtech1/airesearcher:v1` which includes:

- Ubuntu 20.04 base system
- Python 3.11 with research libraries
- PyTorch and ML frameworks
- Git and development tools
- ArXiv and research paper access tools

### GPU Configuration

GPU support is automatically detected and configured if available:

```bash
# Test GPU availability
docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi
```

### Workspace Structure

```
workspace_paper/
├── task_{instance_id}_{model}/
│   └── workplace/
│       ├── project/           # ML project development
│       ├── downloads/         # Paper and code downloads  
│       ├── cache/            # Agent execution cache
│       └── logs/             # Execution logs
```

## Error Handling

Common issues and solutions:

1. **Docker Not Available**
   - Install Docker Engine
   - Start Docker daemon
   - Add user to docker group

2. **Permission Denied**
   - Check Docker socket permissions
   - Run with appropriate privileges
   - Verify volume mount permissions

3. **Image Pull Failures**
   - Check internet connectivity
   - Verify Docker Hub access
   - Use alternative registries if needed

4. **GPU Support Issues**
   - Install NVIDIA Docker runtime
   - Verify GPU drivers
   - Check CUDA compatibility

## Validation

The setup validates:

- ✅ Docker daemon accessibility
- ✅ Base image availability
- ✅ GPU runtime support (if available)
- ✅ Workspace directory creation
- ✅ Port availability
- ✅ Volume mount permissions

## Next Steps

After successful environment setup:

1. Paper discovery process begins
2. Research workspace is ready for agent execution
3. Docker container provides isolated execution environment
4. All subsequent steps can access the configured workspace

## Dependencies

This step has no dependencies and must be executed first in the pipeline.

## Execution Time

Typical execution time: 2-5 minutes
- Docker image pull: 2-3 minutes (if not cached)
- Environment validation: 30 seconds
- Workspace setup: 30 seconds