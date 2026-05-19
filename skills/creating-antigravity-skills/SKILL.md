---
name: creating-antigravity-skills
description: Generates, tests, and optimizes high-quality .agent/skills/ directories based on user requirements. Use when the user asks to build a new skill, test an existing skill's performance, or optimize a skill's description.
---

# Antigravity Skill Creator (Evaluation-Driven Workflow)

A robust workflow for creating new skills and iteratively improving them through quantitative and qualitative evaluation.

## When to use this skill
- User asks to create, build, or generate a new agent skill.
- User provides instructions or examples and requests them to be turned into a skill.
- User wants to test, benchmark, or optimize the description of an existing skill.

## Workflow

- [ ] **Capture Intent & Structure**: Understand the core logic. Generate the mandatory `.agent/skills/<skill-name>/` directory structure.
- [ ] **Write the Draft**: Write `SKILL.md` following the "Claude Way" principles (concise, progressive disclosure).
- [ ] **Generate Test Cases**: Draft 2-3 realistic test prompts and expected outputs.
- [ ] **Execute Evaluations**: Run the test cases using the skill (and optionally a baseline). Collect outputs and record timing.
- [ ] **Review & Grade**: Use the scripts in `scripts/` or `resources/eval-viewer/` to show results to the user. Grade the outputs based on predefined assertions.
- [ ] **Iterate**: Improve the `SKILL.md` based on the user's feedback or grading failures. Keep testing until the skill reliably passes all checks.
- [ ] **Optimize Description**: Generate 20 test queries and use the `run_loop.py` script to find the optimal triggering description.
- [ ] **Package**: Finalize the directory structure.

## Instructions

### 1. Core Structural Requirements
Every skill you generate MUST physically exist on the filesystem:
- `.agent/skills/<skill-name>/`
    - `SKILL.md` (Required: Main logic and instructions)
    - `scripts/` (Optional: Helper scripts)
    - `examples/` (Optional: Reference implementations)
    - `resources/` (Optional: Templates or assets)

### 2. YAML Frontmatter Standards
The `SKILL.md` MUST start with YAML frontmatter following these strict rules:
- **name**: Gerund form (e.g., `testing-code`). Max 64 chars. Lowercase, numbers, and hyphens only.
- **description**: Written in **third person**. Must include specific triggers/keywords. Max 1024 chars.

### 3. Writing Principles (The "Claude Way")
- **Conciseness**: Assume the agent is smart. Focus ONLY on the unique logic.
- **Progressive Disclosure**: Keep `SKILL.md` under 500 lines. Heavy assets go in `resources/`.
- **Explain the Why**: Use theory of mind. Instead of rigid MUSTs, explain *why* something is important. Avoid unneeded all-caps instructions.
- **Identify Repeated Work**: If test cases show the agent writing the same code (like a `build_chart.py`) over and over, package it as a helper script in `scripts/` and instruct the skill to use it.

### 4. Evaluation Loop
Do not stop at the first draft. A bulletproof skill is tested:
1. **Define Test Cases**: Ask the user for 2-3 test scenarios and save them to an `evals.json`.
2. **Run the Tests**: Execute the prompts against the drafted skill.
3. **Draft Assertions**: While runs are in progress, draft quantitative assertions (e.g. "Contains a JSON object").
4. **Aggregate & Review**: Once tests are complete, use the tools in `scripts/` (like `aggregate_benchmark.py`) to compile the benchmark. Optionally launch the eval viewer (found in `resources/eval-viewer/`) for the user.
5. **Listen to Feedback**: Ask the user to review the outputs. Adjust the `SKILL.md` logic accordingly.

### 5. Description Optimization
The description field determines if the agent will use the skill. 
1. Create 20 eval queries (10 should-trigger, 10 should-not-trigger) and save to a JSON file.
2. Run `python .agent/skills/creating-antigravity-skills/scripts/run_loop.py` with the eval set to automatically test and tune the `description` for maximum accuracy.

### 6. Tools & Scripts
You have access to several Python scripts in `scripts/` from the Anthropic repository:
- `aggregate_benchmark.py`: Aggregates test run results.
- `improve_description.py` & `run_loop.py`: Description optimization engine.
- `package_skill.py`: Packages the skill for deployment.
- `run_eval.py`: Runs evaluations.

Consult `resources/agents/` for subagent prompting templates (like graders and comparators).
