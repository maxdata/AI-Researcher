#!/usr/bin/env python3
"""
Paper Discovery Module for AI-Researcher Pipeline
Searches ArXiv and GitHub for relevant research papers and code repositories.
"""

import json
import os
import arxiv
import requests
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
from tqdm import tqdm
import feedparser
from github import Github
import time
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PaperDiscovery:
    """Handles paper and repository discovery from multiple sources"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.github_client = None
        self.results = {
            "papers": [],
            "repositories": [],
            "search_stats": {}
        }
        
        # Initialize GitHub client if token available
        github_token = os.getenv("GITHUB_TOKEN")
        if github_token:
            self.github_client = Github(github_token)
            logger.info("GitHub client initialized with authentication")
        else:
            self.github_client = Github()
            logger.warning("GitHub client initialized without authentication (lower rate limits)")
    
    def search_arxiv_papers(self) -> List[Dict[str, Any]]:
        """Search ArXiv for relevant papers"""
        logger.info("Starting ArXiv paper search...")
        
        category = self.config["category"]
        keywords = self.config.get("keywords", [])
        max_papers = self.config.get("max_papers", 20)
        date_limit = self.config.get("date_limit")
        
        # Map categories to ArXiv search terms
        category_mappings = {
            "vq": ["vector quantization", "VQ-VAE", "discrete representation"],
            "gnn": ["graph neural network", "GNN", "graph convolution"],
            "recommendation": ["recommendation system", "collaborative filtering", "recommender"],
            "diffu_flow": ["diffusion model", "flow matching", "normalizing flow"],
            "reasoning": ["reasoning", "logical reasoning", "causal reasoning"]
        }
        
        # Build search query
        search_terms = category_mappings.get(category, [category])
        search_terms.extend(keywords)
        query = " OR ".join([f'"{term}"' for term in search_terms])
        
        # Add category filters for ArXiv
        arxiv_cats = self.config.get("arxiv_categories", ["cs.LG", "cs.AI", "stat.ML"])
        cat_filter = " OR ".join([f"cat:{cat}" for cat in arxiv_cats])
        full_query = f"({query}) AND ({cat_filter})"
        
        logger.info(f"ArXiv search query: {full_query}")
        
        try:
            search = arxiv.Search(
                query=full_query,
                max_results=max_papers,
                sort_by=arxiv.SortCriterion.SubmittedDate
            )
            
            papers = []
            for paper in tqdm(search.results(), desc="Fetching ArXiv papers"):
                # Filter by date if specified
                if date_limit:
                    limit_date = datetime.strptime(date_limit, "%Y-%m-%d")
                    if paper.published.replace(tzinfo=None) < limit_date:
                        continue
                
                paper_data = {
                    "title": paper.title,
                    "authors": [author.name for author in paper.authors],
                    "abstract": paper.summary,
                    "published": paper.published.strftime("%Y-%m-%d"),
                    "arxiv_id": paper.entry_id.split("/")[-1],
                    "arxiv_url": paper.entry_id,
                    "pdf_url": paper.pdf_url,
                    "categories": paper.categories,
                    "source": "arxiv",
                    "relevance_score": self._calculate_relevance(paper.title + " " + paper.summary, search_terms)
                }
                papers.append(paper_data)
            
            # Sort by relevance score
            papers.sort(key=lambda x: x["relevance_score"], reverse=True)
            
            logger.info(f"Found {len(papers)} ArXiv papers")
            return papers
            
        except Exception as e:
            logger.error(f"ArXiv search failed: {e}")
            return []
    
    def search_github_repositories(self, paper_titles: List[str]) -> List[Dict[str, Any]]:
        """Search GitHub for repositories related to papers"""
        if not self.config.get("github_search_enabled", True):
            return []
        
        logger.info("Starting GitHub repository search...")
        
        repositories = []
        max_repos = self.config.get("max_github_repos", 50)
        
        try:
            # Search for repositories using paper keywords
            search_terms = []
            category = self.config["category"]
            
            # Add category-specific terms
            category_terms = {
                "vq": ["vector-quantization", "vqvae", "discrete-representation"],
                "gnn": ["graph-neural-network", "gnn", "graph-convolution"],
                "recommendation": ["recommendation-system", "collaborative-filtering", "recommender"],
                "diffu_flow": ["diffusion-model", "flow-matching", "normalizing-flow"],
                "reasoning": ["reasoning", "logical-reasoning", "causal-reasoning"]
            }
            
            search_terms.extend(category_terms.get(category, [category]))
            
            # Add terms from paper titles
            for title in paper_titles[:5]:  # Use top 5 paper titles
                # Extract meaningful terms from titles
                words = re.findall(r'\b[a-zA-Z]{4,}\b', title.lower())
                search_terms.extend(words[:3])  # Add top 3 words from each title
            
            # Remove duplicates and create search queries
            unique_terms = list(set(search_terms))
            
            repo_count = 0
            for term in unique_terms[:10]:  # Limit search terms to avoid rate limiting
                if repo_count >= max_repos:
                    break
                
                try:
                    # Search repositories
                    repos = self.github_client.search_repositories(
                        query=f"{term} language:python",
                        sort="stars",
                        order="desc"
                    )
                    
                    for repo in repos[:5]:  # Top 5 repos per term
                        if repo_count >= max_repos:
                            break
                        
                        # Filter out irrelevant repositories
                        if self._is_relevant_repository(repo, category):
                            repo_data = {
                                "name": repo.name,
                                "full_name": repo.full_name,
                                "description": repo.description or "",
                                "url": repo.html_url,
                                "clone_url": repo.clone_url,
                                "stars": repo.stargazers_count,
                                "forks": repo.forks_count,
                                "language": repo.language,
                                "created_at": repo.created_at.strftime("%Y-%m-%d") if repo.created_at else None,
                                "updated_at": repo.updated_at.strftime("%Y-%m-%d") if repo.updated_at else None,
                                "topics": repo.get_topics(),
                                "has_issues": repo.has_issues,
                                "has_wiki": repo.has_wiki,
                                "search_term": term,
                                "relevance_score": self._calculate_repo_relevance(repo, category)
                            }
                            repositories.append(repo_data)
                            repo_count += 1
                    
                    # Rate limiting
                    time.sleep(1)
                    
                except Exception as e:
                    logger.warning(f"GitHub search failed for term '{term}': {e}")
                    continue
            
            # Remove duplicates and sort by relevance
            unique_repos = {}
            for repo in repositories:
                if repo["full_name"] not in unique_repos:
                    unique_repos[repo["full_name"]] = repo
            
            repositories = list(unique_repos.values())
            repositories.sort(key=lambda x: x["relevance_score"], reverse=True)
            
            logger.info(f"Found {len(repositories)} GitHub repositories")
            return repositories
            
        except Exception as e:
            logger.error(f"GitHub search failed: {e}")
            return []
    
    def load_benchmark_instance(self, instance_path: str) -> Optional[Dict[str, Any]]:
        """Load predefined papers from benchmark instance"""
        try:
            with open(instance_path) as f:
                instance = json.load(f)
            
            papers = []
            source_papers = instance.get("source_papers", [])
            
            for paper_info in source_papers:
                paper_data = {
                    "title": paper_info.get("reference", ""),
                    "usage": paper_info.get("usage", ""),
                    "source": "benchmark",
                    "relevance_score": 1.0  # Benchmark papers are highly relevant
                }
                papers.append(paper_data)
            
            logger.info(f"Loaded {len(papers)} papers from benchmark instance")
            return {
                "papers": papers,
                "instance_metadata": {
                    "instance_id": instance.get("instance_id"),
                    "url": instance.get("url"),
                    "task_instructions": instance.get("task1", instance.get("task2", ""))
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to load benchmark instance: {e}")
            return None
    
    def _calculate_relevance(self, text: str, keywords: List[str]) -> float:
        """Calculate relevance score based on keyword matches"""
        text_lower = text.lower()
        score = 0.0
        
        for keyword in keywords:
            keyword_lower = keyword.lower()
            if keyword_lower in text_lower:
                # Higher score for exact matches
                exact_matches = text_lower.count(keyword_lower)
                score += exact_matches * 0.1
        
        return min(score, 1.0)  # Cap at 1.0
    
    def _is_relevant_repository(self, repo, category: str) -> bool:
        """Check if repository is relevant to the research category"""
        # Skip if no description
        if not repo.description:
            return False
        
        # Must be Python (primary requirement)
        if repo.language != "Python":
            return False
        
        # Must have reasonable activity
        if repo.stargazers_count < 5:
            return False
        
        # Check for research-related indicators
        description_lower = repo.description.lower()
        research_indicators = ["neural", "learning", "model", "algorithm", "deep", "machine", "ai", "research"]
        
        if not any(indicator in description_lower for indicator in research_indicators):
            return False
        
        return True
    
    def _calculate_repo_relevance(self, repo, category: str) -> float:
        """Calculate repository relevance score"""
        score = 0.0
        
        # Base score from stars (logarithmic scale)
        if repo.stargazers_count > 0:
            score += min(0.3, 0.1 * (repo.stargazers_count ** 0.5) / 100)
        
        # Recency bonus (repositories updated in last year)
        if repo.updated_at:
            days_since_update = (datetime.now() - repo.updated_at.replace(tzinfo=None)).days
            if days_since_update < 365:
                score += 0.2
        
        # Description relevance
        if repo.description:
            category_keywords = {
                "vq": ["vector", "quantization", "discrete", "vae"],
                "gnn": ["graph", "neural", "network", "gnn"],
                "recommendation": ["recommendation", "collaborative", "filtering"],
                "diffu_flow": ["diffusion", "flow", "matching", "normalizing"],
                "reasoning": ["reasoning", "logical", "causal"]
            }
            
            keywords = category_keywords.get(category, [])
            desc_lower = repo.description.lower()
            
            for keyword in keywords:
                if keyword in desc_lower:
                    score += 0.1
        
        # Topic relevance
        topics = repo.get_topics()
        relevant_topics = ["machine-learning", "deep-learning", "pytorch", "tensorflow", "research"]
        for topic in topics:
            if topic in relevant_topics:
                score += 0.05
        
        return min(score, 1.0)
    
    def run_discovery(self) -> Dict[str, Any]:
        """Run complete paper and repository discovery"""
        logger.info("Starting paper discovery process...")
        
        # Check for benchmark instance
        benchmark_data = None
        instance_path = Path("inputs/benchmark_instance.json")
        if instance_path.exists():
            benchmark_data = self.load_benchmark_instance(str(instance_path))
        
        # Search ArXiv papers
        arxiv_papers = self.search_arxiv_papers()
        
        # Combine with benchmark papers if available
        all_papers = arxiv_papers
        if benchmark_data:
            all_papers.extend(benchmark_data["papers"])
        
        # Search GitHub repositories
        paper_titles = [paper["title"] for paper in all_papers[:10]]  # Use top 10 titles
        repositories = self.search_github_repositories(paper_titles)
        
        # Compile results
        results = {
            "papers": all_papers,
            "repositories": repositories,
            "search_stats": {
                "arxiv_papers_found": len(arxiv_papers),
                "github_repos_found": len(repositories),
                "total_papers": len(all_papers),
                "search_timestamp": datetime.now().isoformat(),
                "category": self.config["category"],
                "benchmark_papers": len(benchmark_data["papers"]) if benchmark_data else 0
            }
        }
        
        if benchmark_data:
            results["benchmark_metadata"] = benchmark_data["instance_metadata"]
        
        logger.info(f"Discovery completed: {len(all_papers)} papers, {len(repositories)} repositories")
        return results

def main():
    """Main entry point for paper discovery"""
    try:
        # Load search configuration
        config_path = Path("inputs/search_config.json")
        if not config_path.exists():
            raise FileNotFoundError("Search configuration file not found")
        
        with open(config_path) as f:
            config = json.load(f)
        
        # Initialize discovery
        discovery = PaperDiscovery(config)
        
        # Run discovery process
        results = discovery.run_discovery()
        
        # Create output directory
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)
        
        # Save results
        with open(output_dir / "paper_metadata.json", "w") as f:
            json.dump(results["papers"], f, indent=2)
        
        with open(output_dir / "github_repositories.json", "w") as f:
            json.dump(results["repositories"], f, indent=2)
        
        with open(output_dir / "search_summary.json", "w") as f:
            json.dump(results["search_stats"], f, indent=2)
        
        # Create paper cache directory
        cache_dir = output_dir / "paper_cache"
        cache_dir.mkdir(exist_ok=True)
        
        logger.info("Paper discovery completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Paper discovery failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())