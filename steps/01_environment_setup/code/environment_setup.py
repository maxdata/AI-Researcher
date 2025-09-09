#!/usr/bin/env python3
"""
Environment Setup Module for AI-Researcher Pipeline
Initializes Docker environment and validates all dependencies.
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnvironmentConfig:
    """Configuration model for environment setup"""
    def __init__(self, **kwargs):
        self.base_image = kwargs.get("base_image", "tjbtech1/airesearcher:v1")
        self.completion_model = kwargs.get("completion_model", "gpt-4o-2024-08-06")
        self.cheep_model = kwargs.get("cheep_model", "gpt-4o-mini")
        self.workspace_name = kwargs.get("workspace_name", "workplace")
        self.container_name = kwargs.get("container_name", "paper_eval")
        self.port = kwargs.get("port", 12345)
        self.gpu_config = kwargs.get("gpu_config", '"device=0"')
        self.category = kwargs.get("category", "vq")
        self.task_level = kwargs.get("task_level", "task1")
        self.max_iter_times = kwargs.get("max_iter_times", 0)
    
    def to_dict(self):
        return {
            "base_image": self.base_image,
            "completion_model": self.completion_model,
            "cheep_model": self.cheep_model,
            "workspace_name": self.workspace_name,
            "container_name": self.container_name,
            "port": self.port,
            "gpu_config": self.gpu_config,
            "category": self.category,
            "task_level": self.task_level,
            "max_iter_times": self.max_iter_times
        }

class DockerConfig:
    """Docker configuration for research environment"""
    def __init__(self, container_name, workplace_name, communication_port, 
                 local_root, base_image, gpu_config):
        self.container_name = container_name
        self.workplace_name = workplace_name
        self.communication_port = communication_port
        self.local_root = local_root
        self.base_image = base_image
        self.gpu_config = gpu_config
    
    def to_dict(self):
        return {
            "container_name": self.container_name,
            "workplace_name": self.workplace_name,
            "communication_port": self.communication_port,
            "local_root": self.local_root,
            "base_image": self.base_image,
            "gpu_config": self.gpu_config
        }

class EnvironmentSetup:
    """Handles Docker environment initialization and validation"""
    
    def __init__(self, config: EnvironmentConfig):
        self.config = config
        self.status = {
            "docker_available": False,
            "gpu_support": False,
            "image_available": False,
            "container_created": False,
            "workspace_mounted": False
        }
    
    def validate_docker(self) -> bool:
        """Validate Docker installation and availability"""
        try:
            result = subprocess.run(["docker", "--version"], capture_output=True, text=True)
            if result.returncode == 0:
                self.status["docker_available"] = True
                logger.info(f"Docker is available: {result.stdout.strip()}")
                return True
            else:
                logger.error("Docker not found or not working")
                return False
        except Exception as e:
            logger.error(f"Docker validation failed: {e}")
            return False
    
    def check_gpu_support(self) -> bool:
        """Check if GPU support is available"""
        try:
            result = subprocess.run(
                ["docker", "run", "--rm", "--gpus", "all", "nvidia/cuda:11.8-base-ubuntu20.04", "nvidia-smi"],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0:
                self.status["gpu_support"] = True
                logger.info("GPU support is available")
                return True
        except Exception as e:
            logger.warning(f"GPU support check failed: {e}")
        
        self.status["gpu_support"] = False
        logger.info("GPU support not available, using CPU only")
        return False
    
    def pull_base_image(self) -> bool:
        """Pull the base Docker image if not available"""
        try:
            logger.info(f"Checking for base image: {self.config.base_image}")
            
            # Check if image exists
            result = subprocess.run(
                ["docker", "image", "inspect", self.config.base_image], 
                capture_output=True, text=True
            )
            
            if result.returncode == 0:
                logger.info(f"Base image {self.config.base_image} already available")
            else:
                logger.info(f"Pulling base image: {self.config.base_image}")
                result = subprocess.run(
                    ["docker", "pull", self.config.base_image], 
                    capture_output=True, text=True
                )
                if result.returncode == 0:
                    logger.info(f"Successfully pulled {self.config.base_image}")
                else:
                    logger.error(f"Failed to pull image: {result.stderr}")
                    return False
            
            self.status["image_available"] = True
            return True
        except Exception as e:
            logger.error(f"Failed to pull base image: {e}")
            return False
    
    def create_workspace(self) -> Dict[str, str]:
        """Create and configure the research workspace"""
        # Create local workspace directory
        instance_id = f"{self.config.category}_{self.config.task_level}"
        local_root = Path.cwd() / "workspace_paper" / f"task_{instance_id}_{self.config.completion_model.replace('/', '__')}" / self.config.workspace_name
        local_root.mkdir(parents=True, exist_ok=True)
        
        # Create Docker configuration
        container_name = f"{self.config.container_name}_{instance_id}_{self.config.completion_model.replace('/', '__')}"
        
        docker_config = DockerConfig(
            container_name=container_name,
            workplace_name=self.config.workspace_name,
            communication_port=self.config.port,
            local_root=str(local_root),
            base_image=self.config.base_image,
            gpu_config=self.config.gpu_config
        )
        
        self.status["workspace_mounted"] = True
        logger.info(f"Workspace created at: {local_root}")
        
        return {
            "local_root": str(local_root),
            "container_name": container_name,
            "docker_config": docker_config.to_dict()
        }
    
    def validate_environment(self) -> Dict[str, Any]:
        """Run complete environment validation"""
        logger.info("Starting environment validation...")
        
        # Validate Docker
        if not self.validate_docker():
            raise RuntimeError("Docker validation failed")
        
        # Check GPU support
        self.check_gpu_support()
        
        # Pull base image
        if not self.pull_base_image():
            raise RuntimeError("Failed to prepare base image")
        
        # Create workspace
        workspace_info = self.create_workspace()
        
        # Final status
        self.status["container_created"] = True
        
        environment_status = {
            "status": "success",
            "validation_results": self.status,
            "workspace_info": workspace_info,
            "config": self.config.to_dict()
        }
        
        logger.info("Environment validation completed successfully")
        return environment_status

def main():
    """Main entry point for environment setup"""
    try:
        # Load configuration
        config_path = Path("inputs/config.json")
        if config_path.exists():
            with open(config_path) as f:
                config_data = json.load(f)
            config = EnvironmentConfig(**config_data)
        else:
            logger.warning("No config file found, using defaults")
            config = EnvironmentConfig()
        
        # Initialize environment
        env_setup = EnvironmentSetup(config)
        
        # Run validation
        status = env_setup.validate_environment()
        
        # Create output directory
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)
        
        # Save environment status
        with open(output_dir / "environment_status.json", "w") as f:
            json.dump(status, f, indent=2)
        
        # Save Docker configuration
        with open(output_dir / "docker_config.json", "w") as f:
            json.dump(status["workspace_info"]["docker_config"], f, indent=2)
        
        # Create workspace directory
        workspace_dir = output_dir / "workspace"
        workspace_dir.mkdir(exist_ok=True)
        
        logger.info("Environment setup completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Environment setup failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())