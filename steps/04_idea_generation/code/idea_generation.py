#!/usr/bin/env python3
"""
Idea Generation Module for AI-Researcher Pipeline
Generates research ideas through survey analysis or implements provided ideas.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IdeaGenerator:
    """Handles research idea generation and analysis"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.mode = config.get("mode", "survey_based")  # "survey_based" or "user_provided"
        self.category = config.get("category", "vq")
        
    def load_research_context(self) -> Dict[str, Any]:
        """Load papers and repositories for context"""
        context = {
            "papers": [],
            "repositories": [],
            "domain_knowledge": {}
        }
        
        try:
            # Load selected papers
            papers_path = Path("inputs/selected_papers.json")
            if papers_path.exists():
                with open(papers_path) as f:
                    context["papers"] = json.load(f)
            
            # Load selected repositories
            repos_path = Path("inputs/selected_repositories.json")
            if repos_path.exists():
                with open(repos_path) as f:
                    context["repositories"] = json.load(f)
            
            logger.info(f"Loaded {len(context['papers'])} papers and {len(context['repositories'])} repositories")
            
        except Exception as e:
            logger.error(f"Failed to load research context: {e}")
        
        return context
    
    def analyze_research_gaps(self, papers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze papers to identify research gaps"""
        # This is a simplified version - in practice would use LLM analysis
        gaps = {
            "technical_gaps": [],
            "methodological_gaps": [],
            "application_gaps": [],
            "evaluation_gaps": []
        }
        
        # Extract common themes and identify potential gaps
        common_themes = self._extract_themes(papers)
        
        # Generate gaps based on category
        category_gaps = {
            "vq": {
                "technical_gaps": [
                    "Efficient codebook learning for large-scale data",
                    "Better initialization strategies for vector quantization",
                    "Adaptive quantization levels based on data complexity"
                ],
                "methodological_gaps": [
                    "Novel architectures combining VQ with transformer models",
                    "Regularization techniques for stable VQ training",
                    "Multi-modal vector quantization approaches"
                ],
                "application_gaps": [
                    "VQ for real-time applications with latency constraints",
                    "Domain-specific VQ models for scientific data",
                    "VQ-based data augmentation techniques"
                ]
            },
            "gnn": {
                "technical_gaps": [
                    "Scalable graph neural networks for billion-node graphs",
                    "Dynamic graph representation learning",
                    "Heterogeneous graph neural architectures"
                ],
                "methodological_gaps": [
                    "Theoretical understanding of GNN expressivity",
                    "Robust GNN training with noisy graph structures",
                    "Few-shot learning for graph tasks"
                ]
            }
        }
        
        gaps.update(category_gaps.get(self.category, {}))
        
        return {
            "identified_gaps": gaps,
            "common_themes": common_themes,
            "analysis_summary": f"Found {len(papers)} papers in {self.category} domain"
        }
    
    def _extract_themes(self, papers: List[Dict[str, Any]]) -> List[str]:
        """Extract common themes from paper abstracts"""
        # Simplified theme extraction
        themes = []
        
        for paper in papers[:5]:  # Analyze top 5 papers
            title = paper.get("title", "").lower()
            abstract = paper.get("abstract", "").lower()
            
            # Extract key terms based on category
            if self.category == "vq":
                if any(term in title + abstract for term in ["quantization", "discrete", "codebook"]):
                    themes.append("Vector Quantization Methods")
                if any(term in title + abstract for term in ["vae", "autoencoder", "generative"]):
                    themes.append("Generative Models")
            elif self.category == "gnn":
                if any(term in title + abstract for term in ["graph", "node", "edge"]):
                    themes.append("Graph Neural Networks")
                if any(term in title + abstract for term in ["convolution", "attention", "transformer"]):
                    themes.append("Neural Architectures")
        
        return list(set(themes))
    
    def generate_survey_based_ideas(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate ideas based on survey of existing papers"""
        papers = context["papers"]
        repositories = context["repositories"]
        
        # Analyze research gaps
        gap_analysis = self.analyze_research_gaps(papers)
        
        # Generate ideas addressing identified gaps
        ideas = []
        
        # Template ideas based on category and gaps
        idea_templates = self._get_idea_templates()
        
        for i, template in enumerate(idea_templates[:3]):  # Generate top 3 ideas
            idea = {
                "idea_id": f"idea_{i+1}",
                "title": template["title"],
                "description": template["description"],
                "motivation": template["motivation"],
                "technical_approach": template["technical_approach"],
                "novelty": template["novelty"],
                "feasibility_score": template["feasibility_score"],
                "impact_score": template["impact_score"],
                "related_papers": self._find_related_papers(template, papers[:3]),
                "related_repositories": self._find_related_repositories(template, repositories[:2]),
                "research_questions": template["research_questions"],
                "expected_contributions": template["expected_contributions"]
            }
            ideas.append(idea)
        
        logger.info(f"Generated {len(ideas)} survey-based ideas")
        return ideas
    
    def load_user_provided_idea(self) -> Optional[Dict[str, Any]]:
        """Load user-provided research idea"""
        try:
            idea_path = Path("inputs/user_idea.txt")
            if not idea_path.exists():
                return None
            
            with open(idea_path) as f:
                user_idea_text = f.read().strip()
            
            if not user_idea_text:
                return None
            
            # Structure the user idea
            idea = {
                "idea_id": "user_provided",
                "title": "User-Provided Research Idea",
                "description": user_idea_text,
                "motivation": "As specified by the user",
                "technical_approach": "To be detailed in implementation planning",
                "novelty": "To be analyzed against existing literature",
                "feasibility_score": 0.8,  # Default high feasibility for user ideas
                "impact_score": 0.7,
                "source": "user_provided",
                "research_questions": ["How to implement the proposed approach?", "What are the expected improvements?"],
                "expected_contributions": ["Implementation of user-specified research idea"]
            }
            
            logger.info("Loaded user-provided research idea")
            return idea
            
        except Exception as e:
            logger.error(f"Failed to load user-provided idea: {e}")
            return None
    
    def _get_idea_templates(self) -> List[Dict[str, Any]]:
        """Get idea templates based on research category"""
        templates = {
            "vq": [
                {
                    "title": "Adaptive Vector Quantization with Dynamic Codebook Expansion",
                    "description": "A novel approach that dynamically expands the codebook size based on data complexity and reconstruction requirements, enabling efficient representation of both simple and complex regions.",
                    "motivation": "Current VQ methods use fixed codebook sizes, leading to inefficient representation of data with varying complexity.",
                    "technical_approach": "Implement a hierarchical VQ system with automatic codebook expansion using reconstruction error thresholds and complexity metrics.",
                    "novelty": "First approach to combine adaptive codebook sizing with hierarchical vector quantization.",
                    "feasibility_score": 0.75,
                    "impact_score": 0.8,
                    "research_questions": [
                        "How to determine optimal expansion criteria?",
                        "What is the computational overhead of dynamic expansion?",
                        "How does this approach scale to high-dimensional data?"
                    ],
                    "expected_contributions": [
                        "Novel adaptive VQ algorithm",
                        "Theoretical analysis of optimal codebook sizing",
                        "Empirical evaluation on multiple domains"
                    ]
                },
                {
                    "title": "Multi-Scale Vector Quantization for Hierarchical Data Representation",
                    "description": "A multi-resolution VQ approach that captures both fine-grained details and coarse-grained structures through hierarchical quantization at multiple scales.",
                    "motivation": "Single-scale VQ fails to capture the multi-scale nature of complex data like images and audio.",
                    "technical_approach": "Design a pyramid-based VQ architecture with cross-scale information sharing and joint optimization.",
                    "novelty": "Integration of multi-scale analysis with vector quantization in a unified framework.",
                    "feasibility_score": 0.7,
                    "impact_score": 0.75,
                    "research_questions": [
                        "How to optimize across multiple scales jointly?",
                        "What is the optimal number of scale levels?",
                        "How to handle computational complexity?"
                    ],
                    "expected_contributions": [
                        "Multi-scale VQ architecture",
                        "Cross-scale optimization algorithm",
                        "Applications to image and audio processing"
                    ]
                }
            ],
            "gnn": [
                {
                    "title": "Temporal Graph Neural Networks with Memory-Augmented Architecture",
                    "description": "A novel GNN architecture that incorporates external memory mechanisms to capture long-term temporal dependencies in dynamic graphs.",
                    "motivation": "Current temporal GNNs struggle with long-range temporal dependencies and suffer from vanishing gradient problems.",
                    "technical_approach": "Integrate neural memory modules with graph convolution operations, enabling selective reading and writing of temporal patterns.",
                    "novelty": "First integration of external memory mechanisms with temporal graph neural networks.",
                    "feasibility_score": 0.8,
                    "impact_score": 0.85,
                    "research_questions": [
                        "How to design memory access patterns for graphs?",
                        "What is the optimal memory size and structure?",
                        "How does memory help with long-term dependencies?"
                    ],
                    "expected_contributions": [
                        "Memory-augmented GNN architecture",
                        "Temporal pattern learning algorithm",
                        "Benchmark results on dynamic graph datasets"
                    ]
                }
            ]
        }
        
        return templates.get(self.category, [])
    
    def _find_related_papers(self, idea_template: Dict[str, Any], papers: List[Dict[str, Any]]) -> List[str]:
        """Find papers related to the generated idea"""
        related = []
        idea_keywords = idea_template["title"].lower().split()
        
        for paper in papers:
            title = paper.get("title", "").lower()
            if any(keyword in title for keyword in idea_keywords[-3:]):  # Use last 3 words as key terms
                related.append(paper.get("title", "Unknown"))
        
        return related[:3]  # Return top 3 related papers
    
    def _find_related_repositories(self, idea_template: Dict[str, Any], repositories: List[Dict[str, Any]]) -> List[str]:
        """Find repositories related to the generated idea"""
        related = []
        idea_keywords = idea_template["title"].lower().split()
        
        for repo in repositories:
            description = repo.get("description", "").lower()
            name = repo.get("name", "").lower()
            if any(keyword in description + name for keyword in idea_keywords[-3:]):
                related.append(repo.get("full_name", "Unknown"))
        
        return related[:2]  # Return top 2 related repositories
    
    def select_best_idea(self, ideas: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Select the best idea based on novelty and feasibility"""
        if not ideas:
            raise ValueError("No ideas to select from")
        
        if len(ideas) == 1:
            return ideas[0]
        
        # Score ideas based on combined novelty, feasibility, and impact
        for idea in ideas:
            novelty = idea.get("novelty_score", idea.get("feasibility_score", 0.5))
            feasibility = idea.get("feasibility_score", 0.5)
            impact = idea.get("impact_score", 0.5)
            
            # Combined score with weights
            idea["combined_score"] = 0.4 * novelty + 0.4 * feasibility + 0.2 * impact
        
        # Sort by combined score and select the best
        ideas.sort(key=lambda x: x.get("combined_score", 0), reverse=True)
        best_idea = ideas[0]
        
        logger.info(f"Selected best idea: {best_idea['title']}")
        return best_idea
    
    def run_idea_generation(self) -> Dict[str, Any]:
        """Run complete idea generation process"""
        logger.info(f"Starting idea generation in {self.mode} mode...")
        
        # Load research context
        context = self.load_research_context()
        
        ideas = []
        
        if self.mode == "user_provided":
            # Load user-provided idea
            user_idea = self.load_user_provided_idea()
            if user_idea:
                ideas = [user_idea]
                logger.info("Using user-provided research idea")
            else:
                logger.warning("No user idea found, falling back to survey-based generation")
                self.mode = "survey_based"
        
        if self.mode == "survey_based":
            # Generate ideas based on survey
            ideas = self.generate_survey_based_ideas(context)
        
        if not ideas:
            raise ValueError("Failed to generate any research ideas")
        
        # Select the best idea
        selected_idea = self.select_best_idea(ideas)
        
        # Prepare results
        results = {
            "generation_mode": self.mode,
            "category": self.category,
            "all_ideas": ideas,
            "selected_idea": selected_idea,
            "context_summary": {
                "papers_analyzed": len(context["papers"]),
                "repositories_analyzed": len(context["repositories"]),
                "total_ideas_generated": len(ideas)
            }
        }
        
        logger.info(f"Idea generation completed: selected '{selected_idea['title']}'")
        return results

def main():
    """Main entry point for idea generation"""
    try:
        # Load configuration
        config_path = Path("inputs/idea_config.json")
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
        else:
            # Default configuration
            config = {
                "mode": "survey_based",
                "category": "vq",
                "max_ideas": 3
            }
            logger.warning("No idea config file found, using defaults")
        
        # Initialize idea generator
        generator = IdeaGenerator(config)
        
        # Run idea generation
        results = generator.run_idea_generation()
        
        # Create output directory
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)
        
        # Save results
        with open(output_dir / "generated_ideas.json", "w") as f:
            json.dump(results["all_ideas"], f, indent=2)
        
        with open(output_dir / "selected_idea.json", "w") as f:
            json.dump(results["selected_idea"], f, indent=2)
        
        with open(output_dir / "idea_generation_summary.json", "w") as f:
            json.dump({
                "generation_mode": results["generation_mode"],
                "category": results["category"],
                "context_summary": results["context_summary"],
                "selected_idea_title": results["selected_idea"]["title"]
            }, f, indent=2)
        
        logger.info("Idea generation completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Idea generation failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())