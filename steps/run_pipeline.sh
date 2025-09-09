#!/bin/bash
set -euo pipefail

# AI-Researcher Pipeline Execution Script
# Runs the complete end-to-end research automation pipeline

PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="pipeline_execution_$(date +%Y%m%d_%H%M%S).log"

echo "Starting AI-Researcher Pipeline Execution" | tee -a "$LOG_FILE"
echo "Pipeline Directory: $PIPELINE_DIR" | tee -a "$LOG_FILE"
echo "Execution Time: $(date)" | tee -a "$LOG_FILE"
echo "=========================================" | tee -a "$LOG_FILE"

# Function to execute a step
execute_step() {
    local step_id=$1
    local step_name=$2
    
    echo "" | tee -a "$LOG_FILE"
    echo "Executing Step $step_id: $step_name" | tee -a "$LOG_FILE"
    echo "----------------------------------------" | tee -a "$LOG_FILE"
    
    cd "$PIPELINE_DIR/$step_id"
    
    # Make run script executable
    chmod +x run.sh
    
    # Execute the step
    if ./run.sh 2>&1 | tee -a "../$LOG_FILE"; then
        echo "✅ Step $step_id completed successfully" | tee -a "../$LOG_FILE"
        return 0
    else
        echo "❌ Step $step_id failed" | tee -a "../$LOG_FILE"
        return 1
    fi
}

# Function to copy outputs between steps
copy_step_outputs() {
    local from_step=$1
    local to_step=$2
    local file_mappings=$3
    
    echo "Copying outputs from $from_step to $to_step..." | tee -a "$LOG_FILE"
    
    IFS=',' read -ra MAPPINGS <<< "$file_mappings"
    for mapping in "${MAPPINGS[@]}"; do
        IFS=':' read -ra PARTS <<< "$mapping"
        local source="${PARTS[0]}"
        local dest="${PARTS[1]}"
        
        local source_path="$PIPELINE_DIR/$from_step/outputs/$source"
        local dest_path="$PIPELINE_DIR/$to_step/inputs/$dest"
        
        if [ -e "$source_path" ]; then
            cp -r "$source_path" "$dest_path"
            echo "  Copied $source -> $dest" | tee -a "$LOG_FILE"
        else
            echo "  Warning: Source $source_path not found" | tee -a "$LOG_FILE"
        fi
    done
}

# Validate prerequisites
echo "Validating prerequisites..." | tee -a "$LOG_FILE"

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed" | tee -a "$LOG_FILE"
    exit 1
fi

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed" | tee -a "$LOG_FILE"
    exit 1
fi

echo "✅ Prerequisites validated" | tee -a "$LOG_FILE"

# Execute pipeline steps in sequence
cd "$PIPELINE_DIR"

# Step 01: Environment Setup
execute_step "01_environment_setup" "Environment Setup" || exit 1

# Step 02: Paper Discovery
execute_step "02_paper_discovery" "Paper Discovery" || exit 1

# Copy outputs from step 02 to step 03
copy_step_outputs "02_paper_discovery" "03_repository_preparation" \
    "paper_metadata.json:paper_metadata.json,github_repositories.json:github_repositories.json"

# Step 03: Repository Preparation  
execute_step "03_repository_preparation" "Repository Preparation" || exit 1

# Copy outputs from step 03 to step 04
copy_step_outputs "03_repository_preparation" "04_idea_generation" \
    "selected_papers.json:selected_papers.json"

# Step 04: Idea Generation
execute_step "04_idea_generation" "Idea Generation" || exit 1

# Copy outputs from step 04 to step 05
copy_step_outputs "04_idea_generation" "05_implementation_planning" \
    "selected_idea.json:selected_idea.json"

# Step 05: Implementation Planning
execute_step "05_implementation_planning" "Implementation Planning" || exit 1

# Copy outputs from step 05 to step 06
copy_step_outputs "05_implementation_planning" "06_model_development" \
    "implementation_plan.json:implementation_plan.json,architecture_design.json:architecture_design.json"

# Copy repository sources to step 06
copy_step_outputs "03_repository_preparation" "06_model_development" \
    "downloaded_repos:downloaded_repos"

# Step 06: Model Development
execute_step "06_model_development" "Model Development" || exit 1

# Copy outputs from step 06 to step 07
copy_step_outputs "06_model_development" "07_code_evaluation" \
    "project:project,training_log.json:training_log.json"

# Step 07: Code Evaluation
execute_step "07_code_evaluation" "Code Evaluation" || exit 1

# Copy outputs from step 07 to step 08
copy_step_outputs "07_code_evaluation" "08_experimental_analysis" \
    "refined_project:refined_project,evaluation_report.json:evaluation_report.json"

# Step 08: Experimental Analysis
execute_step "08_experimental_analysis" "Experimental Analysis" || exit 1

# Pipeline completion
echo "" | tee -a "$LOG_FILE"
echo "=========================================" | tee -a "$LOG_FILE"
echo "🎉 AI-Researcher Pipeline Completed Successfully!" | tee -a "$LOG_FILE"
echo "Completion Time: $(date)" | tee -a "$LOG_FILE"
echo "Total Execution Log: $LOG_FILE" | tee -a "$LOG_FILE"

# Generate final summary
echo "" | tee -a "$LOG_FILE"
echo "📊 Pipeline Summary:" | tee -a "$LOG_FILE"
echo "  - Environment: $(ls 01_environment_setup/outputs/ 2>/dev/null | wc -l) artifacts" | tee -a "$LOG_FILE"
echo "  - Papers Found: $(python3 -c "import json; print(len(json.load(open('02_paper_discovery/outputs/paper_metadata.json', 'r'))))" 2>/dev/null || echo "N/A")" | tee -a "$LOG_FILE"
echo "  - Repositories: $(python3 -c "import json; print(len(json.load(open('02_paper_discovery/outputs/github_repositories.json', 'r'))))" 2>/dev/null || echo "N/A")" | tee -a "$LOG_FILE"
echo "  - Final Results: $(ls 08_experimental_analysis/outputs/ 2>/dev/null | wc -l) artifacts" | tee -a "$LOG_FILE"

echo ""
echo "🔍 Key Outputs:"
echo "  📄 Research Paper Draft: 08_experimental_analysis/outputs/research_paper_draft.md"
echo "  📊 Experimental Results: 08_experimental_analysis/outputs/experimental_results.json"
echo "  💻 ML Project Code: 07_code_evaluation/outputs/refined_project/"
echo "  📈 Analysis Report: 08_experimental_analysis/outputs/analysis_report.json"