#!/usr/bin/env python3
"""
Repository Preparation Module for AI-Researcher Pipeline
Selects and prepares reference codebases and downloads paper sources.
"""

import json
import os
import shutil
import subprocess
import requests
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
from git import Repo
from git.exc import GitCommandError
import arxiv
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RepositoryPreparation:
    """Handles repository selection and preparation"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.selected_repos = []
        self.selected_papers = []
        self.download_stats = {
            "repos_downloaded": 0,
            "papers_downloaded": 0,
            "failed_downloads": 0
        }
    
    def load_discovery_data(self) -> tuple[List[Dict], List[Dict]]:
        """Load paper and repository data from discovery step"""
        try:
            # Load papers
            with open("inputs/paper_metadata.json") as f:
                papers = json.load(f)
            
            # Load repositories
            with open("inputs/github_repositories.json") as f:
                repositories = json.load(f)
            
            logger.info(f"Loaded {len(papers)} papers and {len(repositories)} repositories")
            return papers, repositories
            
        except Exception as e:
            logger.error(f"Failed to load discovery data: {e}")
            return [], []
    
    def select_repositories(self, repositories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Select repositories based on criteria"""
        max_repos = self.config.get("max_repositories", 5)
        min_stars = self.config.get("min_stars", 10)
        preferred_languages = self.config.get("preferred_languages", ["Python"])
        
        # Filter repositories
        filtered_repos = []
        
        for repo in repositories:
            # Check minimum stars
            if repo.get("stars", 0) < min_stars:
                continue
            
            # Check language preference
            if repo.get("language") not in preferred_languages:
                continue
            
            # Check if repository is not archived
            # (This would need additional API call in real implementation)
            
            # Check for README and documentation
            has_readme = True  # Assume most repos have README
            
            if has_readme:
                filtered_repos.append(repo)
        
        # Sort by relevance score and stars
        filtered_repos.sort(key=lambda x: (x.get("relevance_score", 0), x.get("stars", 0)), reverse=True)
        
        # Select top repositories
        selected = filtered_repos[:max_repos]
        
        logger.info(f"Selected {len(selected)} repositories from {len(repositories)} candidates")
        return selected
    
    def select_papers(self, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Select papers based on criteria"""
        max_papers = self.config.get("max_papers", 10)
        min_relevance = self.config.get("min_relevance_score", 0.1)
        
        # Filter papers by relevance
        filtered_papers = [
            paper for paper in papers 
            if paper.get("relevance_score", 0) >= min_relevance
        ]
        
        # Sort by relevance score
        filtered_papers.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        
        # Select top papers
        selected = filtered_papers[:max_papers]
        
        logger.info(f"Selected {len(selected)} papers from {len(papers)} candidates")
        return selected
    
    def download_repository(self, repo_info: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
        """Download a single repository"""
        repo_name = repo_info["full_name"].replace("/", "_")
        local_path = output_dir / repo_name
        
        try:
            logger.info(f"Downloading repository: {repo_info['full_name']}")
            
            # Clone repository
            Repo.clone_from(repo_info["clone_url"], local_path, depth=1)
            
            # Get repository info
            repo_size = self._get_directory_size(local_path)
            
            # Create repository metadata
            metadata = {
                "name": repo_info["name"],
                "full_name": repo_info["full_name"],
                "description": repo_info.get("description", ""),
                "stars": repo_info.get("stars", 0),
                "language": repo_info.get("language", ""),
                "local_path": str(local_path),
                "size_mb": round(repo_size / (1024 * 1024), 2),
                "clone_url": repo_info["clone_url"],
                "download_status": "success"
            }
            
            self.download_stats["repos_downloaded"] += 1
            logger.info(f"Successfully downloaded {repo_info['full_name']} ({metadata['size_mb']} MB)")
            
            return metadata
            
        except GitCommandError as e:
            logger.error(f"Failed to clone {repo_info['full_name']}: {e}")
            return {
                "name": repo_info["name"],
                "full_name": repo_info["full_name"],
                "download_status": "failed",
                "error": str(e)
            }
        except Exception as e:
            logger.error(f"Unexpected error downloading {repo_info['full_name']}: {e}")
            self.download_stats["failed_downloads"] += 1
            return {
                "name": repo_info["name"],
                "full_name": repo_info["full_name"],
                "download_status": "failed",
                "error": str(e)
            }
    
    def download_paper(self, paper_info: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
        """Download a single paper"""
        try:
            paper_id = paper_info.get("arxiv_id")
            if not paper_id:
                return {
                    "title": paper_info["title"],
                    "download_status": "skipped",
                    "reason": "No ArXiv ID available"
                }
            
            logger.info(f"Downloading paper: {paper_info['title'][:50]}...")
            
            # Create safe filename
            safe_title = self._sanitize_filename(paper_info["title"])
            
            # Download PDF
            pdf_path = output_dir / f"{safe_title}.pdf"
            if paper_info.get("pdf_url"):
                response = requests.get(paper_info["pdf_url"])
                if response.status_code == 200:
                    with open(pdf_path, "wb") as f:
                        f.write(response.content)
                else:
                    logger.warning(f"Failed to download PDF for {paper_info['title']}")
            
            # Try to download source if available
            source_path = None
            try:
                # This is a simplified version - in practice you'd use the arxiv API
                # or the existing arxiv_source.py functionality
                search = arxiv.Search(id_list=[paper_id])
                paper = next(search.results())
                
                if paper:
                    source_path = output_dir / f"{safe_title}_source.tex"
                    # In a real implementation, you'd download and extract the source
                    # For now, just create a placeholder
                    with open(source_path, "w") as f:
                        f.write(f"% Source for {paper.title}\n% Abstract: {paper.summary[:200]}...\n")
                
            except Exception as e:
                logger.warning(f"Could not download source for {paper_info['title']}: {e}")
            
            metadata = {
                "title": paper_info["title"],
                "authors": paper_info.get("authors", []),
                "arxiv_id": paper_id,
                "pdf_path": str(pdf_path) if pdf_path.exists() else None,
                "source_path": str(source_path) if source_path and source_path.exists() else None,
                "abstract": paper_info.get("abstract", ""),
                "published": paper_info.get("published", ""),
                "download_status": "success"
            }
            
            self.download_stats["papers_downloaded"] += 1
            logger.info(f"Successfully downloaded paper: {paper_info['title'][:50]}...")
            
            time.sleep(1)  # Rate limiting
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to download paper {paper_info['title']}: {e}")
            self.download_stats["failed_downloads"] += 1
            return {
                "title": paper_info["title"],
                "download_status": "failed",
                "error": str(e)
            }
    
    def _get_directory_size(self, path: Path) -> int:
        """Get total size of directory in bytes"""
        total_size = 0
        try:
            for dirpath, dirnames, filenames in os.walk(path):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    if os.path.exists(fp):
                        total_size += os.path.getsize(fp)
        except Exception:
            pass
        return total_size
    
    def _sanitize_filename(self, filename: str) -> str:
        """Create a safe filename from title"""
        # Remove/replace unsafe characters
        unsafe_chars = '<>:"/\\|?*'
        for char in unsafe_chars:
            filename = filename.replace(char, "_")
        
        # Limit length
        if len(filename) > 100:
            filename = filename[:100]
        
        return filename.strip()
    
    def run_preparation(self) -> Dict[str, Any]:
        """Run complete repository preparation process"""
        logger.info("Starting repository preparation...")
        
        # Load data from discovery step
        papers, repositories = self.load_discovery_data()
        
        # Select repositories and papers
        selected_repos = self.select_repositories(repositories)
        selected_papers = self.select_papers(papers)
        
        # Create output directories
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)
        
        repos_dir = output_dir / "downloaded_repos"
        repos_dir.mkdir(exist_ok=True)
        
        papers_dir = output_dir / "paper_sources"
        papers_dir.mkdir(exist_ok=True)
        
        # Download repositories
        logger.info(f"Downloading {len(selected_repos)} repositories...")
        downloaded_repos = []
        for repo in selected_repos:
            repo_metadata = self.download_repository(repo, repos_dir)
            downloaded_repos.append(repo_metadata)
        
        # Download papers
        logger.info(f"Downloading {len(selected_papers)} papers...")
        downloaded_papers = []
        for paper in selected_papers:
            paper_metadata = self.download_paper(paper, papers_dir)
            downloaded_papers.append(paper_metadata)
        
        # Compile results
        results = {
            "preparation_summary": {
                "repositories_selected": len(selected_repos),
                "papers_selected": len(selected_papers),
                "repositories_downloaded": self.download_stats["repos_downloaded"],
                "papers_downloaded": self.download_stats["papers_downloaded"],
                "failed_downloads": self.download_stats["failed_downloads"],
                "total_repo_size_mb": sum(
                    repo.get("size_mb", 0) for repo in downloaded_repos 
                    if repo.get("download_status") == "success"
                )
            },
            "selected_repositories": downloaded_repos,
            "selected_papers": downloaded_papers
        }
        
        logger.info(f"Preparation completed: {self.download_stats['repos_downloaded']} repos, "
                   f"{self.download_stats['papers_downloaded']} papers downloaded")
        
        return results

def main():
    """Main entry point for repository preparation"""
    try:
        # Load selection criteria
        config_path = Path("inputs/selection_criteria.json")
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
        else:
            # Default configuration
            config = {
                "max_repositories": 5,
                "max_papers": 10,
                "min_stars": 10,
                "min_relevance_score": 0.1,
                "preferred_languages": ["Python"]
            }
            logger.warning("No selection criteria file found, using defaults")
        
        # Initialize preparation
        prep = RepositoryPreparation(config)
        
        # Run preparation process
        results = prep.run_preparation()
        
        # Create output directory
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)
        
        # Save results
        with open(output_dir / "selected_repositories.json", "w") as f:
            json.dump(results["selected_repositories"], f, indent=2)
        
        with open(output_dir / "selected_papers.json", "w") as f:
            json.dump(results["selected_papers"], f, indent=2)
        
        with open(output_dir / "preparation_summary.json", "w") as f:
            json.dump(results["preparation_summary"], f, indent=2)
        
        logger.info("Repository preparation completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Repository preparation failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())