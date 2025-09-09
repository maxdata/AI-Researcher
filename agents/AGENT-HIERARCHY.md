# Agent Hierarchy and Invocation Guide

## 🎯 Quick Reference: Which Agent to Call

**ALWAYS START WITH:** `00-repo-to-pipeline` (Level 0)

## 📊 Complete Agent Hierarchy

```
LEVEL 0 (Entry Point)
└── 00-repo-to-pipeline [META ORCHESTRATOR]
    │
    │ (delegates to)
    ↓
    LEVEL 1 (Management)
    └── 01-pipeline-orchestrator [WORKFLOW MANAGER]
        │
        │ (coordinates all Level 2 agents)
        ↓
        LEVEL 2 (Specialists)
        ├── 02-code-analyzer [ANALYSIS]
        ├── 03-structure-generator [CREATION]
        ├── 04-code-extractor [EXTRACTION]
        ├── 05-pipeline-connector [INTEGRATION]
        ├── 06-documentation-generator [DOCUMENTATION]
        └── 07-quality-verifier [VALIDATION]
```

## 🔢 Agent Numbering System

| Number | Level | Role | Entry Point? |
|--------|-------|------|--------------|
| **00** | 0 | Meta Orchestrator | ✅ YES - ALWAYS START HERE |
| **01** | 1 | Pipeline Manager | ❌ No - Called by 00 |
| **02** | 2 | Code Analyzer | ❌ No - Called by 01 |
| **03** | 2 | Structure Generator | ❌ No - Called by 01 |
| **04** | 2 | Code Extractor | ❌ No - Called by 01 |
| **05** | 2 | Pipeline Connector | ❌ No - Called by 01 |
| **06** | 2 | Doc Generator | ❌ No - Called by 01 |
| **07** | 2 | Quality Verifier | ❌ No - Called by 01 |

## 🚀 How to Invoke Agents

### Correct Usage (Start with Level 0)
```python
Task(
    description="Convert repository to pipeline",
    prompt="Convert [repo_name] to a structured pipeline with steps",
    subagent_type="00-repo-to-pipeline"  # ✅ CORRECT - Start here
)
```

### Internal Delegation (Automatic)
```python
# Level 0 automatically calls Level 1:
Task(
    subagent_type="01-pipeline-orchestrator"  # Called by 00
)

# Level 1 automatically calls Level 2:
Task(
    subagent_type="02-code-analyzer"  # Called by 01
)
```

### ❌ Common Mistakes to Avoid

```python
# WRONG - Don't skip Level 0
Task(
    subagent_type="01-pipeline-orchestrator"  # ❌ Not the entry point
)

# WRONG - Don't call Level 2 directly
Task(
    subagent_type="04-code-extractor"  # ❌ Should be called by 01
)

# WRONG - Using names without numbers
Task(
    subagent_type="repo-to-pipeline"  # ❌ Missing 00- prefix
)
```

## 📋 Execution Flow

### Step-by-Step Process

1. **User Request** → Invoke `00-repo-to-pipeline`
2. **00** analyzes request → Delegates to `01-pipeline-orchestrator`
3. **01** manages workflow → Calls `02-code-analyzer`
4. **02** analyzes repo → Returns findings to **01**
5. **01** → Calls `03-structure-generator`
6. **03** creates folders → Returns to **01**
7. **01** → Calls `04-code-extractor`
8. **04** extracts code → Returns to **01**
9. **01** → Calls `05-pipeline-connector`
10. **05** links steps → Returns to **01**
11. **01** → Calls `06-documentation-generator`
12. **06** creates docs → Returns to **01**
13. **01** → Calls `07-quality-verifier`
14. **07** validates → Returns to **01**
15. **01** → Reports to **00**
16. **00** → Returns to User

## 🎯 Agent Responsibilities by Level

### Level 0: Strategic (Decision Making)
- **00-repo-to-pipeline**: Understands user intent, initiates process

### Level 1: Tactical (Management)
- **01-pipeline-orchestrator**: Manages workflow, coordinates specialists

### Level 2: Operational (Execution)
- **02-code-analyzer**: Understands code structure
- **03-structure-generator**: Creates folder structure
- **04-code-extractor**: Copies real code
- **05-pipeline-connector**: Links steps together
- **06-documentation-generator**: Creates proof documents
- **07-quality-verifier**: Validates everything

## 💡 Key Principles

1. **Hierarchy Enforcement**: Always start at Level 0
2. **No Skip Levels**: Level 0 → Level 1 → Level 2 (never 0 → 2)
3. **Single Entry Point**: Only `00-repo-to-pipeline` is user-facing
4. **Delegation Chain**: Each level only talks to adjacent levels
5. **Specialist Independence**: Level 2 agents don't call each other

## 🔄 Communication Pattern

```
User ↔ Level 0 ↔ Level 1 ↔ Level 2
     (00)    (01)    (02-07)
```

- User only talks to **00**
- **00** only talks to **01**
- **01** orchestrates **02-07**
- **02-07** are leaf nodes (don't call others)

## ✅ Quick Checklist

When using this agent system:

- [ ] Always invoke `00-repo-to-pipeline` first
- [ ] Never skip levels in the hierarchy
- [ ] Use the numbered format (00-, 01-, etc.)
- [ ] Let agents delegate naturally
- [ ] Don't call Level 2 agents directly

## 📝 Example: Complete Invocation

```markdown
User: "Convert my ML repository to a pipeline"

Correct invocation:
Task(
    description="Repository to pipeline conversion",
    prompt="""
    Convert ML repository to structured pipeline
    Remove: notebooks, visualizations
    Steps: data_loading, preprocessing, training, evaluation
    Ensure: real code, no mocks
    """,
    subagent_type="00-repo-to-pipeline"  # ✅ Start here
)
```

The `00-repo-to-pipeline` agent will then automatically:
1. Understand the request
2. Invoke `01-pipeline-orchestrator`
3. Which coordinates all Level 2 agents
4. Returns the complete pipeline

## 🚨 Important Notes

- **The numbering system is CRITICAL** for proper delegation
- **00 is the ONLY user-facing agent**
- **Level 2 agents are specialists** that only Level 1 can invoke
- **Never invoke agents out of order**

---

*Remember: 00 → 01 → 02-07 is the only valid flow. Always start with 00-repo-to-pipeline!*