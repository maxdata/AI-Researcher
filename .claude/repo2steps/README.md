# Repository-to-Pipeline Agent System

A hierarchical agent system for converting any repository into a structured, documented pipeline.

## ⚠️ IMPORTANT: Agent Hierarchy

**ALWAYS START WITH: `00-repo-to-pipeline`** (This is the ONLY entry point)

See **[AGENT-HIERARCHY.md](AGENT-HIERARCHY.md)** for complete invocation guide.

## 🤖 Agent Directory (By Hierarchy Level)

### Level 0: Entry Point
- **[00-repo-to-pipeline.md](00-repo-to-pipeline.md)** - 🚪 MAIN ENTRY - Meta orchestrator

### Level 1: Management 
- **[01-pipeline-orchestrator.md](01-pipeline-orchestrator.md)** - Workflow manager (called by 00)

### Level 2: Specialists (called by 01)
- **[02-code-analyzer.md](02-code-analyzer.md)** - Repository structure analysis
- **[03-structure-generator.md](03-structure-generator.md)** - Folder/file creation
- **[04-code-extractor.md](04-code-extractor.md)** - Real code extraction (no mocks!)
- **[05-pipeline-connector.md](05-pipeline-connector.md)** - Step linking/data flow
- **[06-documentation-generator.md](06-documentation-generator.md)** - Demonstration files
- **[07-quality-verifier.md](07-quality-verifier.md)** - Final quality validation

## 🚀 Quick Start

### Basic Usage (CORRECT)
```markdown
"use 00-repo-to-pipeline agent to convert [repository] to steps"
```

### ❌ WRONG (Don't skip levels)
```markdown
"use 01-pipeline-orchestrator to convert..."  # Wrong - not entry point
"use 04-code-extractor to extract..."         # Wrong - Level 2 agent
```

### With Specifications
```markdown
"use repo-to-pipeline agent to convert [repository] to steps
remove all [frontend/tests/docs]
create steps for [step1] [step2] [step3] [step4]
never mock never fake never cheat"
```

## 🔄 Agent Workflow

```
1. Pipeline Orchestrator receives request
   ↓
2. Code Analyzer identifies workflow steps
   ↓
3. Structure Generator creates folders
   ↓
4. Code Extractor copies real code
   ↓
5. Pipeline Connector links steps
   ↓
6. Documentation Generator creates demos
   ↓
7. Quality Verifier ensures everything works
```

## ✅ Quality Standards

The system enforces three core principles:

1. **Never Mock** - All code must be from the original repository
2. **Never Fake** - All data and examples must be real
3. **Never Cheat** - No shortcuts or oversimplifications

## 📊 Agent Capabilities

| Agent | Primary Role | Key Validation |
|-------|--------------|----------------|
| pipeline-orchestrator | Coordinates entire process | Ensures all phases complete |
| code-analyzer | Understands code structure | Identifies real workflows |
| structure-generator | Creates folder hierarchy | Standard structure compliance |
| code-extractor | Copies repository code | No mock implementations |
| pipeline-connector | Links step I/O | Data flows correctly |
| documentation-generator | Creates proof files | Real examples shown |
| quality-verifier | Final validation | Pipeline actually runs |

## 🎯 Supported Repository Types

- **Machine Learning**: data → preprocessing → training → evaluation
- **API Services**: request → auth → logic → database → response
- **Data Pipelines**: extract → transform → load
- **Web Scrapers**: discover → fetch → parse → extract → store
- **Automation Tools**: trigger → collect → process → execute → notify

## 📝 Example Invocations

### For ML Repository
```python
Task(
    description="Convert ML repo to pipeline",
    prompt="""
    Convert sklearn_project to pipeline
    Remove: notebooks, visualizations
    Steps: data_loading, preprocessing, training, evaluation
    Copy actual model.py and train.py code
    """,
    subagent_type="pipeline-orchestrator"
)
```

### For API Service
```python
Task(
    description="Convert API to pipeline",
    prompt="""
    Convert fastapi_service to pipeline
    Remove: frontend, admin
    Steps: request_validation, auth, business_logic, database, response
    Copy actual route handlers and middleware
    """,
    subagent_type="pipeline-orchestrator"
)
```

## 🔍 Verification Checklist

After conversion, the system verifies:

- [ ] No mock patterns in code (`TODO`, `NotImplemented`, `return "mock"`)
- [ ] All imports resolve correctly
- [ ] Pipeline runs without errors
- [ ] Each step produces output
- [ ] Output of step N is input to step N+1
- [ ] Documentation contains real data
- [ ] Complex algorithms properly implemented

## 🛠 Troubleshooting

### "Mock code detected"
→ Re-run code-extractor with stricter requirements

### "Pipeline doesn't connect"
→ Re-run pipeline-connector with explicit paths

### "Import errors"
→ Check code-extractor included all dependencies

### "No output generated"
→ Verify code-extractor got the actual processing logic

## 📚 Related Documentation

- [Prompt Engineering Guide](../docs/prompt-engineering-guide.md)
- [Quick Start Guide](../docs/prompt-quick-start.md)
- [Case Study](../docs/repomaster-conversion-case-study.md)

## 💡 Pro Tips

1. **Be Specific**: Name your steps explicitly
2. **Remove Unnecessary**: Eliminate frontend, tests, docs upfront
3. **Verify Early**: Check first step before proceeding
4. **Demand Real Code**: Always include "never mock never fake never cheat"
5. **Test Execution**: Run the pipeline after generation

## 🎉 Success Metrics

A successful conversion has:
- ✅ All workflow steps identified
- ✅ Real code extracted (no mocks)
- ✅ Steps properly connected
- ✅ Documentation with real examples
- ✅ Pipeline runs end-to-end
- ✅ Quality verification passes

---

*These agents implement the proven patterns from the RepoMaster pipeline conversion, ensuring high-quality, functional pipeline generation for any repository.*