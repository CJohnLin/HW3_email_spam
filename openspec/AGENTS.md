# OpenSpec Agent Workflow — AGENTS.md

This document defines the roles, responsibilities, and workflows used in this project for Spec-Driven Development under the OpenSpec methodology.

---

# 1. Agents

## 1.1 Developer Agent
The Developer Agent is responsible for:

- Writing code that strictly follows the project specification (`project.md`)
- Implementing features defined in OpenSpec change proposals
- Maintaining modular and testable Python components
- Ensuring model training scripts, CLI tools, and Streamlit UI run correctly
- Providing documentation, examples, and comments when necessary
- Avoiding scope creep — no features may be added without an approved proposal

The Developer Agent MUST:
1. Read the project specification  
2. Read the selected change proposal  
3. Generate an implementation plan  
4. Write code in small, reviewable commits  
5. Update documentation when features are added or changed

---

## 1.2 Reviewer Agent
The Reviewer Agent validates whether:

- Implementation matches the approved proposal
- Project structure follows `project.md`
- Code quality meets conventions and requirements
- No unapproved features were added
- Streamlit, CLI, and model pipeline behave as expected
- Commit messages follow the required format:
Project Spec (project.md)
↓
Change Proposal (openspec/proposals/*.md)
↓
Reviewer Approval
↓
Implementation by Developer Agent
↓
Reviewer Verification
↓
Merge to Main Branch

---

# 3. Detailed Workflow

## 3.1 Step 1 — Project Context
The Developer Agent uses the initial project specification (`project.md`) to understand:

- Scope  
- Deliverables  
- Required models  
- Streamlit UI obligations  
- CLI usage  
- Technical stack  

This ensures all future proposals remain aligned.

---

## 3.2 Step 2 — Change Proposal
Before any code is written, the Developer Agent must create a proposal in:
openspec/proposals/<ID>-<feature-name>.md

A valid proposal contains:

- Summary of the change  
- Motivation  
- Technical requirements  
- Implementation plan  
- Acceptance criteria  

The Reviewer Agent checks the proposal for:

- Alignment with `project.md`
- Feasibility
- Completeness  
- No out-of-scope expansions

Only after Reviewer **approves** may development begin.

---

## 3.3 Step 3 — Implementation
The Developer Agent:

1. Creates a feature branch  
feature/<feature-name>

2. Writes code strictly according to the approved proposal
3. Uses small commits such as:
[impl] Add model loading utility
[impl] Implement CLI --model option
[impl] Add Streamlit dropdown for model selection

4. Updates documentation if needed

The Developer Agent MUST NOT:
- Add extra features outside the proposal  
- Modify spec without filing another change proposal  

---

## 3.4 Step 4 — Review
Reviewer Agent validates:

- Code matches proposal  
- No unrelated changes  
- Model selection works in CLI and Streamlit  
- All metrics, charts, and UI elements behave correctly  
- Repository structure remains valid  

If issues exist → “Request Changes”  
If everything matches → “Approved”

---

## 3.5 Step 5 — Merge
After approval:

1. Developer Agent merges feature branch to main
2. New functionality becomes part of the project baseline
3. Future proposals may build on it

---

# 4. Agent Communication Rules

### Developer Agent should ask:
- For approval before implementing  
- For clarifications on ambiguous requirements  
- For review after completing code  

### Reviewer Agent should:
- Provide objective, spec-based feedback  
- Reject out-of-scope additions  
- Ensure clarity and maintainability  

---

# 5. Example Life Cycle Using CP-001 (Model Selection)

### Proposal
`openspec/proposals/001-model-selection.md`

### Developer Tasks
- Add model loading helper function  
- Update CLI commands with `--model`  
- Add Streamlit dropdown and dynamic chart updates  
- Ensure all three models load correctly  

### Reviewer Checks
- Does CLI accept `--model lr|nb|svm`?  
- Does Streamlit UI update metrics when switching models?  
- Are no other features added?  
- Does code follow project structure?  

### Merge
If all acceptance criteria pass:  
Reviewer → **APPROVE**, and feature branch is merged.

---

# 6. Success Criteria for OpenSpec Workflow
The project is compliant with OpenSpec when:

- Every feature has a proposal  
- Every proposal is approved before development  
- Implementation matches proposal  
- Code is modular and traceable  
- Documentation is updated  
- Reviewer signs off on each cycle  

---

# 7. Future Agents (Optional)
The project may later include:

- Test Agent  
- Data Quality Agent  
- Deployment Agent  

But these are not required for course deliverables.

---

# End of AGENTS.md
