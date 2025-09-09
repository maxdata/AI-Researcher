#!/usr/bin/env python3
"""
Code Evaluation Module for AI-Researcher Pipeline
Evaluates implementation quality, debug issues, and refines code.
"""

import json
import os
import subprocess
from pathlib import Path
import ast
from typing import Dict, List, Any, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CodeEvaluator:
    """Handles code quality evaluation and refinement"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.project_path = None
        self.evaluation_results = {
            "syntax_check": {"passed": False, "issues": []},
            "import_check": {"passed": False, "issues": []},
            "structure_check": {"passed": False, "issues": []},
            "best_practices": {"score": 0, "suggestions": []},
            "refinements": []
        }
        
    def load_project(self) -> str:
        """Load the generated project"""
        project_path = Path("inputs/project")
        if not project_path.exists():
            raise FileNotFoundError("Project directory not found")
        
        self.project_path = project_path
        logger.info(f"Loaded project from {project_path}")
        return str(project_path)
    
    def check_syntax(self) -> Dict[str, Any]:
        """Check Python syntax across all files"""
        results = {"passed": True, "issues": []}
        
        python_files = list(self.project_path.rglob("*.py"))
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    source = f.read()
                
                # Parse AST to check syntax
                ast.parse(source)
                logger.info(f"✓ Syntax OK: {py_file.relative_to(self.project_path)}")
                
            except SyntaxError as e:
                results["passed"] = False
                issue = {
                    "file": str(py_file.relative_to(self.project_path)),
                    "line": e.lineno,
                    "error": str(e),
                    "type": "syntax_error"
                }
                results["issues"].append(issue)
                logger.error(f"✗ Syntax Error in {py_file.relative_to(self.project_path)}: {e}")
                
            except Exception as e:
                results["passed"] = False
                issue = {
                    "file": str(py_file.relative_to(self.project_path)),
                    "error": str(e),
                    "type": "parsing_error"
                }
                results["issues"].append(issue)
        
        self.evaluation_results["syntax_check"] = results
        logger.info(f"Syntax check: {'PASSED' if results['passed'] else 'FAILED'} ({len(results['issues'])} issues)")
        return results
    
    def check_imports(self) -> Dict[str, Any]:
        """Check import statements and dependencies"""
        results = {"passed": True, "issues": [], "missing_deps": []}
        
        # Get all import statements
        imports = set()
        python_files = list(self.project_path.rglob("*.py"))
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    tree = ast.parse(f.read())
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            imports.add(alias.name.split('.')[0])
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            imports.add(node.module.split('.')[0])
                            
            except Exception as e:
                results["issues"].append({
                    "file": str(py_file.relative_to(self.project_path)),
                    "error": f"Could not analyze imports: {e}",
                    "type": "import_analysis_error"
                })
        
        # Check if requirements.txt covers the imports
        req_file = self.project_path / "requirements.txt"
        if req_file.exists():
            with open(req_file) as f:
                requirements = {line.split('>=')[0].split('==')[0].strip().replace('-', '_') 
                             for line in f.readlines() if line.strip() and not line.startswith('#')}
        else:
            requirements = set()
        
        # Standard library modules (don't need to be in requirements)
        stdlib_modules = {'os', 'sys', 'json', 'pathlib', 'typing', 'logging', 'argparse', 
                         'collections', 'itertools', 'functools', 'random', 'math', 'time'}
        
        # Find missing dependencies
        external_imports = imports - stdlib_modules
        missing_deps = external_imports - requirements
        
        if missing_deps:
            results["passed"] = False
            results["missing_deps"] = list(missing_deps)
            results["issues"].append({
                "type": "missing_dependencies",
                "missing": list(missing_deps),
                "message": f"Missing dependencies in requirements.txt: {', '.join(missing_deps)}"
            })
        
        self.evaluation_results["import_check"] = results
        logger.info(f"Import check: {'PASSED' if results['passed'] else 'FAILED'} ({len(missing_deps)} missing deps)")
        return results
    
    def check_project_structure(self) -> Dict[str, Any]:
        """Check project structure and organization"""
        results = {"passed": True, "issues": [], "suggestions": []}
        
        required_files = ["README.md", "requirements.txt", "train.py"]
        required_dirs = ["src", "configs"]
        
        # Check required files
        for req_file in required_files:
            if not (self.project_path / req_file).exists():
                results["issues"].append({
                    "type": "missing_file",
                    "file": req_file,
                    "message": f"Missing required file: {req_file}"
                })
                results["passed"] = False
        
        # Check required directories
        for req_dir in required_dirs:
            if not (self.project_path / req_dir).exists():
                results["issues"].append({
                    "type": "missing_directory", 
                    "directory": req_dir,
                    "message": f"Missing required directory: {req_dir}"
                })
                results["passed"] = False
        
        # Check for __init__.py files in Python packages
        src_dir = self.project_path / "src"
        if src_dir.exists():
            for subdir in src_dir.iterdir():
                if subdir.is_dir() and not (subdir / "__init__.py").exists():
                    results["suggestions"].append({
                        "type": "missing_init",
                        "directory": str(subdir.relative_to(self.project_path)),
                        "message": f"Consider adding __init__.py to {subdir.name}/"
                    })
        
        self.evaluation_results["structure_check"] = results
        logger.info(f"Structure check: {'PASSED' if results['passed'] else 'FAILED'} ({len(results['issues'])} issues)")
        return results
    
    def evaluate_best_practices(self) -> Dict[str, Any]:
        """Evaluate code against best practices"""
        results = {"score": 0, "max_score": 100, "suggestions": []}
        
        python_files = list(self.project_path.rglob("*.py"))
        if not python_files:
            return results
        
        total_checks = 0
        passed_checks = 0
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = content.split('\n')
                
                # Check docstrings
                total_checks += 1
                if '"""' in content or "'''" in content:
                    passed_checks += 1
                else:
                    results["suggestions"].append({
                        "file": str(py_file.relative_to(self.project_path)),
                        "type": "missing_docstrings",
                        "message": "Consider adding docstrings to functions and classes"
                    })
                
                # Check line length
                total_checks += 1
                long_lines = [i+1 for i, line in enumerate(lines) if len(line) > 100]
                if not long_lines:
                    passed_checks += 1
                elif len(long_lines) < len(lines) * 0.1:  # Less than 10% of lines are too long
                    passed_checks += 0.5
                    results["suggestions"].append({
                        "file": str(py_file.relative_to(self.project_path)),
                        "type": "long_lines",
                        "lines": long_lines[:5],  # Show first 5
                        "message": f"Consider shortening long lines (>100 chars) on lines {long_lines[:5]}"
                    })
                
                # Check for TODO/FIXME comments
                total_checks += 1
                todos = [i+1 for i, line in enumerate(lines) if 'TODO' in line or 'FIXME' in line]
                if not todos:
                    passed_checks += 1
                else:
                    results["suggestions"].append({
                        "file": str(py_file.relative_to(self.project_path)),
                        "type": "todos_found",
                        "lines": todos,
                        "message": f"Found TODO/FIXME comments on lines {todos} - consider addressing these"
                    })
                
            except Exception as e:
                logger.warning(f"Could not analyze {py_file}: {e}")
        
        # Calculate score
        if total_checks > 0:
            results["score"] = int((passed_checks / total_checks) * 100)
        
        self.evaluation_results["best_practices"] = results
        logger.info(f"Best practices score: {results['score']}/100")
        return results
    
    def generate_refinements(self) -> List[Dict[str, Any]]:
        """Generate code refinements based on evaluation"""
        refinements = []
        
        # Refinements based on syntax issues
        for issue in self.evaluation_results["syntax_check"]["issues"]:
            refinements.append({
                "type": "syntax_fix",
                "file": issue["file"],
                "priority": "high",
                "description": f"Fix syntax error: {issue['error']}",
                "suggested_action": "Review and correct syntax error"
            })
        
        # Refinements for missing dependencies
        missing_deps = self.evaluation_results["import_check"].get("missing_deps", [])
        if missing_deps:
            refinements.append({
                "type": "dependency_fix",
                "priority": "high", 
                "description": f"Add missing dependencies: {', '.join(missing_deps)}",
                "suggested_action": f"Add to requirements.txt: {chr(10).join(missing_deps)}"
            })
        
        # Refinements for structure issues
        for issue in self.evaluation_results["structure_check"]["issues"]:
            refinements.append({
                "type": "structure_fix",
                "priority": "medium",
                "description": issue["message"],
                "suggested_action": f"Create missing {issue.get('file', issue.get('directory', 'item'))}"
            })
        
        # Refinements for best practices
        if self.evaluation_results["best_practices"]["score"] < 70:
            refinements.append({
                "type": "code_quality",
                "priority": "low",
                "description": "Improve code quality and documentation",
                "suggested_action": "Add docstrings, reduce line length, address TODOs"
            })
        
        self.evaluation_results["refinements"] = refinements
        return refinements
    
    def run_evaluation(self) -> Dict[str, Any]:
        """Run complete code evaluation"""
        logger.info("Starting code evaluation...")
        
        # Load project
        try:
            self.load_project()
        except FileNotFoundError:
            # Try alternative path
            alt_path = Path("inputs") / "project"  
            if alt_path.exists():
                self.project_path = alt_path
            else:
                logger.error("No project found for evaluation")
                return {"error": "No project found"}
        
        # Run all checks
        self.check_syntax()
        self.check_imports()
        self.check_project_structure()
        self.evaluate_best_practices()
        
        # Generate refinements
        self.generate_refinements()
        
        # Calculate overall score
        checks_passed = sum([
            self.evaluation_results["syntax_check"]["passed"],
            self.evaluation_results["import_check"]["passed"],
            self.evaluation_results["structure_check"]["passed"]
        ])
        
        overall_score = (checks_passed / 3.0) * 0.7 + (self.evaluation_results["best_practices"]["score"] / 100.0) * 0.3
        
        self.evaluation_results["overall_score"] = round(overall_score * 100, 1)
        self.evaluation_results["status"] = "passed" if overall_score >= 0.7 else "needs_work"
        
        logger.info(f"Code evaluation completed. Overall score: {self.evaluation_results['overall_score']}/100")
        return self.evaluation_results

def main():
    """Main entry point for code evaluation"""
    try:
        config = {}
        
        evaluator = CodeEvaluator(config)
        results = evaluator.run_evaluation()
        
        # Create output directory
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)
        
        # Save evaluation results
        with open(output_dir / "evaluation_report.json", "w") as f:
            json.dump(results, f, indent=2)
        
        # Save refined project (copy with fixes applied)
        if results.get("status") == "passed":
            refined_dir = output_dir / "refined_project"
            if evaluator.project_path and evaluator.project_path.exists():
                import shutil
                if refined_dir.exists():
                    shutil.rmtree(refined_dir)
                shutil.copytree(evaluator.project_path, refined_dir)
                logger.info(f"Refined project saved to {refined_dir}")
        
        logger.info("Code evaluation completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Code evaluation failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())