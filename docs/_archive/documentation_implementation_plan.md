# Implementation Plan: Professional Technical Documentation
### Active Episodic Memory Management via Reinforcement Learning for Robust CNN Inference on Out-of-Distribution Data

> [!NOTE]
> **Plan for LLM Agent execution in Antigravity IDE.**
> This plan must be executed phase by phase. The agent MUST NOT implement all phases at once -- it must wait for the user to request "Phase X" and deliver all files for that phase only.
> After completing each phase, the agent MUST update the status markers in this plan from `[PENDING]` to `[COMPLETED]` with a brief resolution note.

---

## CONTEXT AND OBJECTIVE

This project implements an RL-driven active memory management architecture for a semiparametric vision system (CNN + k-NN episodic memory) targeting Edge AI deployment. The system has been submitted to **EAAI (Engineering Applications of Artificial Intelligence)**, an Elsevier journal.

All 4 implementation phases of the codebase are complete. What is missing is **publication-grade technical documentation** structured as a navigable hierarchy of `README.md` files within the `docs/` directory, with a fully rewritten root `README.md`.

**Documentation Language:** English (all content).
**Formatting Constraints:** No emojis. No decorative Unicode characters. Professional, academic tone.
**License:** GPL-3.0 (must be included as `LICENSE` file and referenced in all READMEs).
**Journal:** EAAI (Elsevier) -- documentation must support reproducibility, include a citation block, and follow Elsevier supplementary material standards.

---

## CURRENT STATE OF THE REPOSITORY

```text
projeto-cnn/
├── src/
│   ├── config.py                              # Centralized configuration (61 lines)
│   ├── dashboard/                             # Flask web dashboard
│   │   ├── app.py
│   │   ├── static/
│   │   └── templates/
│   ├── data/
│   │   └── loader.py                          # MNIST binary parser + tf.data loader
│   ├── models/
│   │   ├── custom_cnn.py                      # From-scratch CNN (Conv2D, Dense via tf.Module)
│   │   ├── knn_bandit_agent.py                # Phase 1: NumPy episodic memory (capacity-bounded)
│   │   ├── mlp_bandit_agent.py                # Legacy MLP Q-Network agent
│   │   ├── reward_manager.py                  # Phase 2: Curriculum Learning reward manager
│   │   └── rl_agent.py                        # Phase 2: Double DQN + PER agent
│   └── scratch/
│       ├── activations.py                     # Raw ReLU, Softmax
│       ├── layers.py                          # Raw Conv2D, Dense, MaxPool2D, Flatten
│       ├── losses.py                          # Categorical cross-entropy
│       └── optimizers.py                      # SGD with assign_sub
├── training/
│   ├── __init__.py
│   └── train_rl_online_simulation.py          # Phase 3: Online RL simulation with Mahalanobis++
├── scripts/
│   ├── benchmark_l40s.py                      # GPU throughput profiling
│   ├── demo_inference.py                      # 6-image inference demo
│   ├── evaluate_global.py                     # Threshold sweep evaluation
│   ├── profile_latent.py                      # Mahalanobis profile computation
│   ├── run_benchmark.py                       # Full benchmark suite
│   ├── simulate_online.py                     # Online simulation runner
│   ├── train_cnn.py                           # CNN training script
│   └── train_rl.py                            # RL training script
├── tests/
│   ├── test_agent.py                          # Phase 1 unit tests
│   ├── test_full_acceptance_phase3.py         # Phase 3 acceptance (50k steps)
│   ├── test_memory_diag.py                    # Memory diagnostics
│   ├── test_phase2.py                         # Phase 2 unit tests
│   ├── test_phase3.py                         # Phase 3 unit tests
│   ├── test_phase4.py                         # Phase 4 unit tests
│   └── test_router.py                         # OOD routing tests
├── visualizations/
│   ├── make_decision_profiles.py              # CNN vs k-NN confidence profiles
│   ├── make_memory_rescue.py                  # Episodic memory rescue dashboard
│   ├── make_saliency.py                       # Gradient saliency maps
│   └── make_tsne.py                           # t-SNE latent space projection
├── evaluate_hybrid_global.py                  # Phase 4: EAAI evaluation + baselines
├── outputs/                                   # Checkpoints, logs, metrics, figures
├── assets/                                    # Documentation images
├── papers/                                    # Reference PDFs
├── docs/                                      # EXISTING docs (Portuguese, unstructured)
│   ├── Literatura/                            # Curated scientific literature (12 areas)
│   ├── Plano_de_Acao_EAAI.md                  # Implementation action plan
│   ├── arquitetura_e_fluxo_cnn.md
│   ├── caso_de_estudo_eaai.md
│   ├── evaluate_hybrid_global.md
│   ├── fluxo_dados_sistema_hibrido.md
│   ├── knn_bandit_agent_128d.md
│   ├── train_rl_128d.md
│   └── train_rl_online_simulation.md
├── requirements.txt
├── README.md                                  # Current root README (outdated, mixed language)
└── .gitignore
```

---

## TARGET DOCUMENTATION STRUCTURE AFTER COMPLETION

The documentation must follow a **top-down hierarchical navigation model**: the root `README.md` provides the system overview with links into `docs/`, and each `docs/` subdirectory contains a `README.md` that zooms into that specific concern, with further links as needed.

```text
projeto-cnn/
├── LICENSE                                     # Phase 0: GPL-3.0 full text
├── CITATION.cff                                # Phase 0: Machine-readable citation
├── README.md                                   # Phase 1: Root README (system overview)
├── docs/
│   ├── README.md                               # Phase 1: Documentation index/map
│   ├── architecture/
│   │   ├── README.md                           # Phase 2: System architecture overview
│   │   ├── cnn-feature-extractor.md            # Phase 2: CNN component deep-dive
│   │   ├── ood-detection.md                    # Phase 2: Mahalanobis++ OOD routing
│   │   ├── episodic-memory.md                  # Phase 2: k-NN memory with NumPy
│   │   ├── rl-agent.md                         # Phase 2: Double DQN + PER
│   │   ├── reward-system.md                    # Phase 2: Curriculum Learning rewards
│   │   └── data-flow.md                        # Phase 2: End-to-end data flow
│   ├── getting-started/
│   │   ├── README.md                           # Phase 3: Installation + prerequisites
│   │   ├── quickstart.md                       # Phase 3: Minimal reproduction
│   │   └── configuration.md                    # Phase 3: Config reference
│   ├── guides/
│   │   ├── README.md                           # Phase 4: Guides index
│   │   ├── training-pipeline.md                # Phase 4: Full training workflow
│   │   ├── evaluation.md                       # Phase 4: Running EAAI evaluation
│   │   └── visualization.md                    # Phase 4: XAI and visualization tools
│   ├── api/
│   │   ├── README.md                           # Phase 5: API reference index
│   │   ├── config.md                           # Phase 5: config.py reference
│   │   ├── custom-cnn.md                       # Phase 5: custom_cnn.py reference
│   │   ├── knn-bandit-agent.md                 # Phase 5: knn_bandit_agent.py reference
│   │   ├── rl-agent.md                         # Phase 5: rl_agent.py reference
│   │   ├── reward-manager.md                   # Phase 5: reward_manager.py reference
│   │   └── train-rl-online-simulation.md       # Phase 5: Simulation module reference
│   ├── results/
│   │   ├── README.md                           # Phase 6: Experimental results overview
│   │   ├── baseline-comparison.md              # Phase 6: 5-baseline analysis
│   │   └── metrics-reference.md                # Phase 6: Metric definitions
│   └── Literatura/                             # PRESERVED (not modified by this plan)
│       └── ...                                 # Existing curated literature
```

---

## INTER-PHASE DEPENDENCY MAP

```text
PHASE 0 ─────────────► PHASE 1 ─────────────► PHASE 2
(License, Citation)     (Root README,           (Architecture docs:
                         Docs index)             6 component deep-dives)
                              │                        │
                              │                        ▼
                              │                  PHASE 3
                              │                  (Getting Started:
                              │                   install, quickstart, config)
                              │                        │
                              │                        ▼
                              ├────────────────► PHASE 4
                              │                  (Guides: training,
                              │                   evaluation, visualization)
                              │                        │
                              │                        ▼
                              ├────────────────► PHASE 5
                              │                  (API Reference:
                              │                   6 module references)
                              │                        │
                              │                        ▼
                              └────────────────► PHASE 6
                                                 (Results: baselines,
                                                  metrics, analysis)
```

---

## DOCUMENTATION STANDARDS AND CONVENTIONS

Every `README.md` and markdown document MUST follow these conventions:

### Header Format
```markdown
# [Document Title]

> Part of the [Active Episodic Memory Management via Reinforcement Learning for Robust CNN Inference on Out-of-Distribution Data] documentation.
> Parent: [link to parent README] | Up: [link to docs/README.md]
```

### Navigation Block
Every document must end with a navigation footer:
```markdown
---

**Navigation:**
- Previous: [Previous Document Title](#relative-link)
- Up: [Documentation Index](../README.md)
- Next: [Next Document Title](#relative-link)
```

### Cross-Reference Convention
Internal references use relative paths. Example:
```markdown
See the [OOD Detection](../architecture/ood-detection.md) documentation for details on Mahalanobis++ thresholding.
```

### Code Block Convention
All code examples must specify the language and include the file path as a comment:
```python
# src/models/knn_bandit_agent.py
agent = KNNBanditAgent128D(capacity=5000, k=30, latent_dim=128)
```

### Table Convention
Tables must use left-aligned headers with separator lines:
```markdown
| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
```

### Mathematical Notation
Use LaTeX inline `$...$` and display `$$...$$` for all mathematical expressions. No approximations or ASCII-art formulas.

### No Emojis
Zero emoji characters in any documentation file. Use text-based status markers only: `[COMPLETED]`, `[PENDING]`, `[IN PROGRESS]`.

### License Header
Every README must include at the bottom:
```markdown
---
Licensed under the GNU General Public License v3.0. See [LICENSE](../../LICENSE) for details.
```

---

## SOURCE FILES THE AGENT MUST READ

Before writing documentation for any component, the agent MUST read the corresponding source file to extract accurate:
- Class and method signatures (including type hints)
- Docstrings and inline comments
- Default parameter values
- Data structures and their shapes/dtypes
- Import dependencies

| Documentation Target | Source File(s) to Read |
|:---------------------|:----------------------|
| CNN Feature Extractor | `src/models/custom_cnn.py`, `src/scratch/layers.py`, `src/scratch/activations.py` |
| OOD Detection | `training/train_rl_online_simulation.py` (Mahalanobis++ section) |
| Episodic Memory | `src/models/knn_bandit_agent.py` |
| RL Agent | `src/models/rl_agent.py` |
| Reward System | `src/models/reward_manager.py` |
| Configuration | `src/config.py` |
| Evaluation | `evaluate_hybrid_global.py` |
| Data Loader | `src/data/loader.py` |
| Training Pipeline | `scripts/train_cnn.py`, `scripts/train_rl.py`, `scripts/profile_latent.py` |
| Visualization | `visualizations/make_*.py` |
| Test Suite | `tests/test_*.py` |

The agent MUST also read the existing Portuguese documentation files in `docs/` to extract any architectural details, mathematical formulations, or flow descriptions that should be preserved and translated:
- `docs/arquitetura_e_fluxo_cnn.md`
- `docs/fluxo_dados_sistema_hibrido.md`
- `docs/knn_bandit_agent_128d.md`
- `docs/train_rl_online_simulation.md`
- `docs/evaluate_hybrid_global.md`
- `docs/caso_de_estudo_eaai.md`

---

# PHASE 0: License and Citation `[COMPLETED]`

> [!NOTE]
> **Status:** Completed.
> **Resolution Note:** Created `LICENSE` with complete GNU GPL-3.0 text (35,149 bytes) and `CITATION.cff` conforming to CFF 1.2.0 standard in valid YAML format.
> **Depends on:** Nothing (root phase).
> **Unblocks:** All subsequent phases.

**Objective:** Create the GPL-3.0 license file and a machine-readable CITATION.cff file.

**Files to CREATE:**
- [x] 0.1 [`LICENSE`](../../LICENSE) `[COMPLETED]`
- [x] 0.2 [`CITATION.cff`](../../CITATION.cff) `[COMPLETED]`

### 0.1 -- `LICENSE` `[COMPLETED]`
- Location: `projeto-cnn/LICENSE`
- Content: The full text of the GNU General Public License v3.0.
- Source: https://www.gnu.org/licenses/gpl-3.0.txt
- The agent MUST fetch or reproduce the exact, complete GPL-3.0 license text.

### 0.2 -- `CITATION.cff` `[COMPLETED]`
- Location: `projeto-cnn/CITATION.cff`
- Content: Machine-readable citation metadata following the [Citation File Format](https://citation-file-format.github.io/) standard.
- Fields to include:
  - `cff-version: 1.2.0`
  - `title:` "Active Episodic Memory Management via Reinforcement Learning for Robust CNN Inference on Out-of-Distribution Data"
  - `message:` "If you use this software, please cite it as below."
  - `type: software`
  - `authors:` (leave as placeholder `- name: "[Author Name]"` for the user to fill)
  - `repository-code:` (placeholder URL)
  - `license: GPL-3.0`
  - `keywords:` edge-ai, reinforcement-learning, out-of-distribution, episodic-memory, trustworthy-ai, convolutional-neural-network, double-dqn, mahalanobis-distance

**Acceptance Criteria:**
- [x] `LICENSE` file exists and contains the complete GPL-3.0 text.
- [x] `CITATION.cff` file exists and is valid YAML.
- [x] Both files are in the project root directory.

---

# PHASE 1: Root README and Documentation Index `[COMPLETED]`

> [!NOTE]
> **Status:** Completed.
> **Resolution Note:** Completely rewrote root `README.md` with all 10 required sections (including Mermaid dataflow diagram and exact `outputs/eaai_metrics.json` 5-baseline evaluation results). Created `docs/README.md` containing the complete documentation map, reading order, and literature reference. Verified zero emojis and valid formatting throughout.
> **Depends on:** Phase 0 (License and Citation must exist).
> **Unblocks:** Phases 2-6 (all documentation phases reference the root).

**Objective:** Completely rewrite the root `README.md` to reflect the current state of the system (post-Phase 4 implementation) and create the `docs/README.md` documentation index.

**Files to CREATE/OVERWRITE:**
- [x] 1.1 [`README.md`](../README.md) (OVERWRITE) `[COMPLETED]`
- [x] 1.2 [`docs/README.md`](README.md) (CREATE) `[COMPLETED]`

### 1.1 -- Root `README.md` (OVERWRITE) `[COMPLETED]`
- Location: `projeto-cnn/README.md`
- This is the primary entry point for anyone visiting the repository.
- The agent MUST read ALL source files listed in the "Source Files" table above before writing this file.

**Required Sections (in order):**

1. **Title and Badges**
   - Title: "Active Episodic Memory Management via Reinforcement Learning for Robust CNN Inference on Out-of-Distribution Data"
   - Subtitle: "Robust Out-of-Distribution Routing for Resource-Constrained Semiparametric Vision Systems"
   - Badges (shields.io, text-only): Python 3.10+, PyTorch 2.0+, TensorFlow 2.10.1, License GPL-3.0, NumPy, scikit-learn
   - No emoji in badges or title.

2. **Abstract**
   - 1 paragraph (150-200 words) summarizing the research contribution in academic style.
   - Must mention: semiparametric architecture (CNN + k-NN), Mahalanobis++ OOD detection, RL-driven active memory management (Double DQN + PER), Curriculum Learning, Edge AI constraints, and the key result (B4 outperforming blind eviction baselines by +4.5% accuracy under noise with 170x better distribution matching).

3. **Key Results**
   - The 5-baseline comparison table from Phase 4 results (extracted from `outputs/eaai_metrics.json`).
   - 3-4 bullet points highlighting the most significant findings.

4. **System Architecture**
   - High-level description of the 4-stage pipeline: (1) CNN feature extraction, (2) Mahalanobis++ OOD routing, (3) k-NN episodic memory retrieval, (4) RL active memory management.
   - Mermaid flowchart diagram showing the data flow.
   - Link to detailed architecture documentation: `docs/architecture/README.md`.

5. **Repository Structure**
   - Updated `tree` view reflecting the current state of ALL directories and key files.
   - Brief one-line description for each directory and key file.

6. **Getting Started**
   - Brief prerequisites section (Python, hardware, dependencies).
   - Quick installation commands.
   - Link to detailed guide: `docs/getting-started/README.md`.

7. **Reproduction Pipeline**
   - Numbered steps with commands:
     1. Train CNN
     2. Profile latent space
     3. Populate episodic memory
     4. Train RL agent (online simulation)
     5. Evaluate (5-baseline comparison)
   - Link to detailed guide: `docs/guides/training-pipeline.md`.

8. **Documentation**
   - Table of all documentation sections with links and brief descriptions.
   - Points to `docs/README.md` as the full documentation index.

9. **Citation**
   - BibTeX block for citing the work.
   - Reference to `CITATION.cff`.

10. **License**
    - GPL-3.0 statement with link to `LICENSE` file.

### 1.2 -- Documentation Index `docs/README.md` (OVERWRITE or CREATE) `[COMPLETED]`
- Location: `projeto-cnn/docs/README.md`
- This is the central navigation hub for all documentation.

**Required Content:**

1. **Title:** "Documentation Index"
2. **Navigation breadcrumb:** Link back to root README.
3. **Documentation Map:** Table listing every documentation section with:
   - Section name (linked)
   - Directory path
   - Brief description (1 sentence)
   - Status marker
4. **Reading Order:** Recommended traversal path for first-time readers:
   1. Architecture Overview
   2. Data Flow
   3. Getting Started
   4. Training Pipeline Guide
   5. API Reference
   6. Experimental Results
5. **Scientific Literature:** Link to `docs/Literatura/README.md` explaining that curated literature is preserved separately and not part of the technical documentation hierarchy.

**Acceptance Criteria:**
- [x] Root `README.md` contains ALL 10 sections listed above.
- [x] All internal links are valid relative paths.
- [x] `docs/README.md` contains the documentation map table.
- [x] No emojis anywhere in either file.
- [x] Language is English throughout.
- [x] Mermaid diagram renders correctly.
- [x] GPL-3.0 license reference is present in both files.
- [x] The 5-baseline results table matches `outputs/eaai_metrics.json` data.

---

# PHASE 2: Architecture Documentation `[COMPLETED]`

> [!NOTE]
> **Status:** Completed.
> **Resolution Note:** Created all 7 architecture deep-dive documents in `docs/architecture/`: System Architecture Overview (`README.md`), CNN Feature Extractor (`cnn-feature-extractor.md`), Out-of-Distribution Detection (`ood-detection.md`), Episodic Memory Buffer (`episodic-memory.md`), Double DQN Active RL Agent (`rl-agent.md`), Curriculum Learning Reward Manager (`reward-system.md`), and End-to-End Data Flow (`data-flow.md`). Verified complete LaTeX mathematical formulations, code examples with exact source paths, zero emojis, and GNU GPL-3.0 license references.
> **Depends on:** Phase 1 (root README and docs index must exist for navigation links).
> **Unblocks:** Phases 3-6 (architecture docs are referenced by all other sections).

**Objective:** Create the `docs/architecture/` directory with deep-dive documentation for every architectural component.

**Files to CREATE:**
- [x] 2.1 [`docs/architecture/README.md`](../architecture/README.md) -- System Architecture Overview `[COMPLETED]`
- [x] 2.2 [`docs/architecture/cnn-feature-extractor.md`](../architecture/cnn-feature-extractor.md) -- CNN Feature Extractor `[COMPLETED]`
- [x] 2.3 [`docs/architecture/ood-detection.md`](../architecture/ood-detection.md) -- Mahalanobis++ OOD Routing `[COMPLETED]`
- [x] 2.4 [`docs/architecture/episodic-memory.md`](../architecture/episodic-memory.md) -- Capacity-Bounded k-NN Buffer `[COMPLETED]`
- [x] 2.5 [`docs/architecture/rl-agent.md`](../architecture/rl-agent.md) -- Double DQN + PER Agent `[COMPLETED]`
- [x] 2.6 [`docs/architecture/reward-system.md`](../architecture/reward-system.md) -- Curriculum Learning Rewards `[COMPLETED]`
- [x] 2.7 [`docs/architecture/data-flow.md`](../architecture/data-flow.md) -- End-to-End Data Flow `[COMPLETED]`

### 2.1 -- Architecture Overview: `docs/architecture/README.md` `[COMPLETED]`

**Before writing, the agent MUST read:**
- `docs/arquitetura_e_fluxo_cnn.md` (existing Portuguese architecture doc -- extract all technical details)
- `docs/fluxo_dados_sistema_hibrido.md` (existing Portuguese data flow doc)
- `src/config.py`

**Required Content:**
1. Title: "System Architecture"
2. Abstract paragraph describing the semiparametric architecture philosophy.
3. High-level architecture diagram (Mermaid) showing all components and their interactions.
4. Component inventory table:

| Component | Module | Purpose | Key Innovation |
|:----------|:-------|:--------|:---------------|
| CNN Feature Extractor | `src/models/custom_cnn.py` | 128D latent representation | From-scratch TF primitives |
| OOD Detector | `training/train_rl_online_simulation.py` | Mahalanobis++ routing | L2 norm + Ledoit-Wolf |
| Episodic Memory | `src/models/knn_bandit_agent.py` | k-NN retrieval buffer | O(1) NumPy pre-allocation |
| RL Agent | `src/models/rl_agent.py` | Active memory curation | Double DQN + PER |
| Reward Manager | `src/models/reward_manager.py` | Curriculum Learning rewards | Geometric-to-accuracy transition |

5. Design decisions section: Why semiparametric? Why Mahalanobis over softmax confidence? Why RL over heuristic eviction?
6. Links to each component deep-dive document.
7. Navigation footer.

### 2.2 -- CNN Feature Extractor: `docs/architecture/cnn-feature-extractor.md` `[COMPLETED]`

**Before writing, the agent MUST read:**
- `src/models/custom_cnn.py`
- `src/scratch/layers.py`
- `src/scratch/activations.py`
- `src/scratch/losses.py`
- `src/scratch/optimizers.py`
- `docs/arquitetura_e_fluxo_cnn.md` (extract the layer-by-layer table and parameter counts)

**Required Content:**
1. Title: "CNN Feature Extractor"
2. Purpose: Dual-purpose network (classification + latent extraction).
3. Architecture table: Layer-by-layer breakdown with input/output dimensions, parameter counts, and mathematical operations (extracted from existing Portuguese doc).
4. Total parameter count.
5. Initialization strategies: He Normal (convolutions), Glorot Uniform (dense).
6. Training details: Custom SGD, learning rate, batch size, epochs.
7. The latent space: Explanation of the 128D bottleneck layer and why it is the interface between the CNN and all downstream components.
8. Code example: How to instantiate and extract latent vectors.
9. Navigation footer.

### 2.3 -- OOD Detection: `docs/architecture/ood-detection.md` `[COMPLETED]`

**Before writing, the agent MUST read:**
- `training/train_rl_online_simulation.py` (specifically the `MahalanobisPPDetector` or equivalent class/functions)
- `scripts/profile_latent.py`
- `docs/train_rl_online_simulation.md`

**Required Content:**
1. Title: "Out-of-Distribution Detection: Mahalanobis++"
2. Mathematical formulation:
   - Standard Mahalanobis distance: $D_M(x) = \sqrt{(x - \mu_c)^T \Sigma_c^{-1} (x - \mu_c)}$
   - L2 normalization: $z_{\text{norm}} = z / (\|z\|_2 + \epsilon)$
   - Ledoit-Wolf shrinkage: Why it is necessary for $N=5000, D=128$.
3. Threshold calibration: 95th percentile of in-distribution distances.
4. Routing logic: ID samples go to CNN, OOD samples go to episodic memory.
5. Profile persistence: `.npz` format for serialized statistical profiles.
6. Scientific references: Lee et al. (2018), Kamoi & Kobayashi (2020), Guo et al. (2025), Chen et al. (2010).
7. Navigation footer.

### 2.4 -- Episodic Memory: `docs/architecture/episodic-memory.md` `[COMPLETED]`

**Before writing, the agent MUST read:**
- `src/models/knn_bandit_agent.py` (entire file)
- `docs/knn_bandit_agent_128d.md`

**Required Content:**
1. Title: "Episodic Memory: Capacity-Bounded k-NN Buffer"
2. Design rationale: Why pre-allocated NumPy arrays instead of dynamic lists (OOM prevention for Edge AI).
3. Data structure specification:
   - `_states`: `np.float32`, shape `(capacity, 128)`
   - `_actions`: `np.int32`, shape `(capacity,)`
   - `_rewards`: `np.float32`, shape `(capacity,)`
   - `_insertion_ticks`: `np.int64`, shape `(capacity,)`
   - `_usage_counts`: `np.int32`, shape `(capacity,)`
4. Insertion logic: O(1) slot assignment or eviction trigger.
5. k-NN search: Vectorized Euclidean distances with `np.argpartition`.
6. Three eviction policies:
   - FIFO (`evict_oldest`): Minimum insertion tick.
   - LFU (`evict_least_frequently_used`): Minimum usage count, tie-break by oldest.
   - Redundancy (`evict_most_redundant`): Minimum geometric distance within same class.
7. Memory diagnostics: `get_memory_stats()` output specification.
8. Persistence: `.npz` save/load format with backward compatibility.
9. Performance benchmarks: 10,000 insertions in 35.54ms.
10. Scientific references: NEC (Pritzel et al., 2017), MFEC (Blundell et al., 2016), Isele & Cosgun (2018).
11. Navigation footer.

### 2.5 -- RL Agent: `docs/architecture/rl-agent.md` `[COMPLETED]`

**Before writing, the agent MUST read:**
- `src/models/rl_agent.py` (entire file)

**Required Content:**
1. Title: "RL Agent: Double DQN with Prioritized Experience Replay"
2. Agent architecture:
   - MLP: 2 hidden layers of 64 units, ReLU activations.
   - Input: 5D normalized state vector.
   - Output: 4 discrete actions.
3. State space specification:

| Dimension | Feature | Normalization | Source |
|:----------|:--------|:-------------|:-------|
| 0 | Mahalanobis distance | $\min(d/50, 1)$ | OOD detector |
| 1 | Local entropy | $H / \log(k)$ | k-NN neighborhood |
| 2 | Minimum k-NN distance | $\min(d/10, 1)$ | Memory search |
| 3 | Prediction error | Binary (0 or 1) | CNN vs ground truth |
| 4 | RAM occupancy | `size / capacity` | Memory buffer |

4. Action space:

| Action | ID | Operation | Method Called |
|:-------|:---|:----------|:-------------|
| Ignore | 0 | Do not evict | None |
| FIFO | 1 | Evict oldest | `evict_oldest()` |
| LFU | 2 | Evict least used | `evict_least_frequently_used()` |
| Redundant | 3 | Evict nearest same-class | `evict_most_redundant()` |

5. Double DQN mechanism: Policy net selects action, target net evaluates value. Formal equation.
6. PER implementation:
   - SumTree data structure (binary tree in NumPy array).
   - Priority formula: $p_i = |\delta_i| + \epsilon$
   - Importance sampling weights: $w_i = (N \cdot P(i))^{-\beta}$
   - Beta annealing: $\beta_0 = 0.4 \to \beta = 1.0$
7. Target network synchronization: Every 100 steps.
8. Persistence: `.pt` checkpoint format.
9. Performance benchmark: 1,000 inferences in 62.46ms.
10. Scientific references: Schaul et al. (2015), Staffolani et al. (2023).
11. Navigation footer.

### 2.6 -- Reward System: `docs/architecture/reward-system.md` `[COMPLETED]`

**Before writing, the agent MUST read:**
- `src/models/reward_manager.py` (entire file)

**Required Content:**
1. Title: "Reward System: Curriculum Learning Reward Manager"
2. Curriculum Learning rationale: Why transition from geometric proxy to accuracy-based reward.
3. Two reward components:
   - Geometric proxy $R_{\text{geom}}$: Local Euclidean density in memory.
   - Accuracy reward $R_{\text{acc}}$: k-NN predictions on sliding validation buffer.
4. Interpolation formula: $R = \alpha \cdot R_{\text{geom}} + (1 - \alpha) \cdot R_{\text{acc}}$
5. Alpha decay: $\alpha_{t+1} = \alpha_t \cdot 0.995$, minimum $\alpha = 0.01$.
6. Sliding validation buffer: Pre-allocated NumPy arrays, size 100, circular insertion.
7. Reward bounds: $R \in [-1.0, 1.0]$ enforced by `np.clip`.
8. Scientific references: Blundell et al. (2016), Curriculum RL literature.
9. Navigation footer.

### 2.7 -- Data Flow: `docs/architecture/data-flow.md` `[COMPLETED]`

**Before writing, the agent MUST read:**
- `docs/fluxo_dados_sistema_hibrido.md`
- `training/train_rl_online_simulation.py`
- `evaluate_hybrid_global.py`

**Required Content:**
1. Title: "End-to-End Data Flow"
2. Three operational modes:
   - Training mode (CNN training)
   - Online simulation mode (RL training under concept drift)
   - Evaluation mode (5-baseline comparison)
3. For each mode: Step-by-step data flow with Mermaid sequence diagram.
4. The prequential protocol: Test-then-train evaluation.
5. Noise injection mechanism: Additive Gaussian perturbation.
6. Concept drift simulation: How noise injection rate and level create non-stationary distributions.
7. Navigation footer.

**Acceptance Criteria:**
- [x] Directory `docs/architecture/` exists with all 7 files.
- [x] Every file contains all required sections listed above.
- [x] All mathematical formulations use LaTeX notation.
- [x] All component specifications match the actual source code (signatures, types, defaults).
- [x] Cross-references between architecture docs use valid relative links.
- [x] Navigation footers are present in all files.
- [x] No emojis.
- [x] GPL-3.0 reference in each file.

---

# PHASE 3: Getting Started Documentation `[COMPLETED]`

> [!NOTE]
> **Status:** Completed.
> **Resolution Note:** Created all 3 getting-started documents in `docs/getting-started/`: Getting Started Index (`README.md`), Installation and Quick Start (`quickstart.md`), and Configuration Reference (`configuration.md`). Included prerequisites, environment verification commands, automated MNIST raw binary setup script, copy-pasteable 5-step reproduction pipeline, exhaustive table covering all 35 parameters from `src/config.py`, Edge AI memory sizing analysis, zero emojis, and GNU GPL-3.0 license references.
> **Depends on:** Phase 1 (docs index for navigation).
> **Unblocks:** Phase 4 (guides reference getting-started for prerequisites).

**Objective:** Create the `docs/getting-started/` directory with installation, quickstart, and configuration reference.

**Files to CREATE:**
- [x] 3.1 [`docs/getting-started/README.md`](../getting-started/README.md) -- Getting Started Index `[COMPLETED]`
- [x] 3.2 [`docs/getting-started/quickstart.md`](../getting-started/quickstart.md) -- Installation and Quick Start `[COMPLETED]`
- [x] 3.3 [`docs/getting-started/configuration.md`](../getting-started/configuration.md) -- Configuration Reference `[COMPLETED]`

### 3.1 -- Getting Started Index: `docs/getting-started/README.md` `[COMPLETED]`

**Required Content:**
1. Title: "Getting Started"
2. Reading order for the 3 documents in this section.
3. Navigation footer.

### 3.2 -- Installation and Prerequisites: `docs/getting-started/quickstart.md` `[COMPLETED]`

**Before writing, the agent MUST read:**
- `requirements.txt`
- `src/config.py`
- `src/data/loader.py` (to understand data path expectations)

**Required Content:**
1. Title: "Installation and Quick Start"
2. Prerequisites:
   - Python >= 3.10
   - CUDA 12.6 (optional, for GPU acceleration)
   - MNIST dataset (raw binary format)
3. Installation steps (copy-pasteable commands):
   ```bash
   git clone <repository-url>
   cd projeto-cnn
   pip install -r requirements.txt
   ```
4. Dataset setup: Where to place MNIST raw files (`data/MNIST/raw/`).
5. Quick verification: A minimal script to verify the installation works.
6. Quick start: The minimal 5-command pipeline to reproduce results.
7. Navigation footer.

### 3.3 -- Configuration Reference: `docs/getting-started/configuration.md` `[COMPLETED]`

**Before writing, the agent MUST read:**
- `src/config.py` (entire file)

**Required Content:**
1. Title: "Configuration Reference"
2. Complete table of ALL configuration parameters in `src/config.py`:

| Parameter | Value | Type | Phase | Description |
|:----------|:------|:-----|:------|:------------|

3. Grouped by functional area:
   - General (seed, paths)
   - CNN Training
   - Mahalanobis Routing
   - Phase 1: Episodic Memory
   - Phase 2: RL Agent
   - Phase 3: Online Simulation
   - Data Paths
4. How to modify parameters.
5. Navigation footer.

**Acceptance Criteria:**
- [x] Directory `docs/getting-started/` exists with 3 files.
- [x] Installation commands are copy-pasteable and correct.
- [x] Configuration table covers ALL 30+ parameters in `src/config.py`.
- [x] No emojis.
- [x] GPL-3.0 reference.

---

# PHASE 4: Usage Guides `[COMPLETED]`

> [!NOTE]
> **Status:** Completed.
> **Resolution Note:** Created all 4 workflow guides in `docs/guides/`: Guides Index (`README.md`), Full Training Pipeline (`training-pipeline.md`), EAAI Evaluation Benchmark (`evaluation.md`), and Explainable AI / Visualization Diagnostics (`visualization.md`). Fully documented the 5-stage sequential training workflow with exact commands and duration estimates, detailed the 5-baseline evaluation methodology under prequential drift simulation with all 7 scientific metrics, and provided operational documentation for all 4 XAI diagnostic tools (t-SNE latent collapse, gradient saliency maps, decision confidence profiles, and 128D episodic memory rescue visualizer). Verified zero emojis and GNU GPL-3.0 license references throughout.
> **Depends on:** Phases 1-3 (architecture and getting-started for cross-references).
> **Unblocks:** Phases 5-6.

**Objective:** Create the `docs/guides/` directory with step-by-step workflow guides.

**Files to CREATE:**
- [x] 4.1 [`docs/guides/README.md`](../guides/README.md) -- Guides Index `[COMPLETED]`
- [x] 4.2 [`docs/guides/training-pipeline.md`](../guides/training-pipeline.md) -- Full Training Pipeline `[COMPLETED]`
- [x] 4.3 [`docs/guides/evaluation.md`](../guides/evaluation.md) -- Running the EAAI Evaluation `[COMPLETED]`
- [x] 4.4 [`docs/guides/visualization.md`](../guides/visualization.md) -- Explainable AI and Visualization Tools `[COMPLETED]`

### 4.1 -- Guides Index: `docs/guides/README.md` `[COMPLETED]`

**Required Content:**
1. Title: "Usage Guides"
2. Table of available guides with descriptions.
3. Navigation footer.

### 4.2 -- Training Pipeline: `docs/guides/training-pipeline.md` `[COMPLETED]`

**Before writing, the agent MUST read:**
- `scripts/train_cnn.py`
- `scripts/profile_latent.py`
- `scripts/train_rl.py`
- `training/train_rl_online_simulation.py`
- `scripts/simulate_online.py`

**Required Content:**
1. Title: "Training Pipeline"
2. Pipeline overview diagram (Mermaid).
3. Step 1: CNN Training
   - Command, expected output, duration estimate.
   - What is produced: CNN checkpoint.
4. Step 2: Latent Space Profiling
   - Command, expected output.
   - What is produced: Mahalanobis profiles (`.npz`).
5. Step 3: Episodic Memory Population
   - Command, expected output.
   - What is produced: Memory bank (`.npz`).
6. Step 4: RL Online Simulation
   - Command, expected output, duration estimate (50,000 steps).
   - What is produced: RL agent checkpoint, simulation log CSV.
7. Step 5: Evaluation
   - Command, expected output.
   - What is produced: Metrics CSV/JSON, dashboard PNG.
8. Expected outputs summary table.
9. Navigation footer.

### 4.3 -- Evaluation Guide: `docs/guides/evaluation.md` `[COMPLETED]`

**Before writing, the agent MUST read:**
- `evaluate_hybrid_global.py`
- `docs/evaluate_hybrid_global.md`

**Required Content:**
1. Title: "Running the EAAI Evaluation"
2. The 5 baselines explained:
   - B0: CNN Only
   - B1: Hybrid + Infinite Memory
   - B2: Hybrid + FIFO Eviction
   - B3: Hybrid + LFU Eviction
   - B4: Hybrid + RL Active Memory (proposed)
3. Command-line options.
4. Output files: CSV, JSON, dashboard PNG.
5. How to interpret results.
6. Navigation footer.

### 4.4 -- Visualization Guide: `docs/guides/visualization.md` `[COMPLETED]`

**Before writing, the agent MUST read:**
- `visualizations/make_decision_profiles.py`
- `visualizations/make_memory_rescue.py`
- `visualizations/make_saliency.py`
- `visualizations/make_tsne.py`

**Required Content:**
1. Title: "Explainable AI and Visualization Tools"
2. For each visualization script:
   - What it produces.
   - Command to run.
   - Brief description of the visualization.
   - Example output reference (link to `assets/` or `outputs/` images).
3. Navigation footer.

**Acceptance Criteria:**
- [x] Directory `docs/guides/` exists with 4 files.
- [x] Training pipeline covers all 5 steps with exact commands.
- [x] All commands are verified against actual script filenames and paths.
- [x] No emojis.
- [x] GPL-3.0 reference.

---

# PHASE 5: API Reference `[COMPLETED]`

> [!NOTE]
> **Status:** Completed.
> **Resolution Note:** Created all 7 API reference documents in `docs/api/`: API Reference Index (`README.md`), Configuration Reference (`config.md`), Custom CNN Model (`custom-cnn.md`), Episodic Memory Agent (`knn-bandit-agent.md`), Double DQN RL Agent (`rl-agent.md`), Curriculum Reward Manager (`reward-manager.md`), and Online Streaming Simulation Engine (`train-rl-online-simulation.md`). Documented every public class, method, property, function, and configuration constant with exact signatures matching the source code, comprehensive parameter tables, Edge AI computational complexity notes, syntactically valid code examples, zero emojis, and GNU GPL-3.0 license references throughout.
> **Depends on:** Phase 2 (architecture docs for cross-references).
> **Unblocks:** Phase 6.

**Objective:** Create the `docs/api/` directory with detailed API reference for every public module.

**Files to CREATE:**
- [x] 5.1 [`docs/api/README.md`](../api/README.md) -- API Reference Index `[COMPLETED]`
- [x] 5.2 [`docs/api/config.md`](../api/config.md) -- `src/config.py` Reference `[COMPLETED]`
- [x] 5.3 [`docs/api/custom-cnn.md`](../api/custom-cnn.md) -- `src/models/custom_cnn.py` Reference `[COMPLETED]`
- [x] 5.4 [`docs/api/knn-bandit-agent.md`](../api/knn-bandit-agent.md) -- `src/models/knn_bandit_agent.py` Reference `[COMPLETED]`
- [x] 5.5 [`docs/api/rl-agent.md`](../api/rl-agent.md) -- `src/models/rl_agent.py` Reference `[COMPLETED]`
- [x] 5.6 [`docs/api/reward-manager.md`](../api/reward-manager.md) -- `src/models/reward_manager.py` Reference `[COMPLETED]`
- [x] 5.7 [`docs/api/train-rl-online-simulation.md`](../api/train-rl-online-simulation.md) -- `training/train_rl_online_simulation.py` Reference `[COMPLETED]`

### 5.1 -- API Reference Index: `docs/api/README.md` `[COMPLETED]`

**Required Content:**
1. Title: "API Reference"
2. Module inventory table with links.
3. Convention: All signatures extracted directly from source code.
4. Navigation footer.

### 5.2 through 5.7 -- Module References `[COMPLETED]`

For EACH of the following modules, create a separate markdown file:

| File | Source Module |
|:-----|:-------------|
| `docs/api/config.md` | `src/config.py` |
| `docs/api/custom-cnn.md` | `src/models/custom_cnn.py` |
| `docs/api/knn-bandit-agent.md` | `src/models/knn_bandit_agent.py` |
| `docs/api/rl-agent.md` | `src/models/rl_agent.py` |
| `docs/api/reward-manager.md` | `src/models/reward_manager.py` |
| `docs/api/train-rl-online-simulation.md` | `training/train_rl_online_simulation.py` |

**For EACH module reference, the agent MUST read the entire source file and document:**

1. Title: Module name.
2. File path and import statement.
3. Module overview (1 paragraph).
4. For EACH public class:
   - Class signature with inheritance.
   - Constructor: Full signature with types and defaults.
   - All public methods: Full signature, parameter table, return type, description.
   - All public properties.
5. For EACH public function:
   - Full signature with types and defaults.
   - Parameter table.
   - Return type and description.
6. Constants and configuration parameters.
7. Usage example (minimal, working code snippet).
8. Cross-references to architecture documentation.
9. Navigation footer.

**Method documentation format:**
```markdown
#### `method_name(param1: type, param2: type = default) -> ReturnType`

Description of what the method does.

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `param1` | `type` | required | Description |
| `param2` | `type` | `default` | Description |

**Returns:** `ReturnType` -- Description of return value.
```

**Acceptance Criteria:**
- [x] Directory `docs/api/` exists with 7 files.
- [x] EVERY public class, method, function, and property is documented.
- [x] ALL signatures match the actual source code exactly (types, defaults, names).
- [x] Usage examples are syntactically correct.
- [x] No emojis.
- [x] GPL-3.0 reference.

---

# PHASE 6: Experimental Results Documentation `[COMPLETED]`

> [!NOTE]
> **Status:** Completed.
> **Resolution Note:** Created all 3 experimental results documents in `docs/results/`: Results Overview (`README.md`), Baseline Comparison Analysis (`baseline-comparison.md`), and Engineering Metrics Reference (`metrics-reference.md`). Fully documented the experimental protocol, hardware constraints, 5-baseline evaluation matrix using exact values from `outputs/eaai_metrics.json`, detailed comparative analysis across 4 scientific dimensions (severe noise robustness, cache pollution mitigation via Action 0, 177x distribution matching gain, and LMOS sustainability), and formalized all 7 engineering metrics with rigorous LaTeX formulas, physical units, literature sources, and exact codebase mappings. Verified zero emojis and GNU GPL-3.0 license references throughout.
> **Depends on:** Phases 2, 4 (architecture and evaluation guide for cross-references).
> **Unblocks:** Phase 7 (Final Verification and Cleanup).

**Objective:** Create the `docs/results/` directory with a comprehensive presentation of experimental results.

**Files to CREATE:**
- [x] 6.1 [`docs/results/README.md`](../results/README.md) -- Results Overview `[COMPLETED]`
- [x] 6.2 [`docs/results/baseline-comparison.md`](../results/baseline-comparison.md) -- Baseline Comparison Analysis `[COMPLETED]`
- [x] 6.3 [`docs/results/metrics-reference.md`](../results/metrics-reference.md) -- Engineering Metrics Reference `[COMPLETED]`

### 6.1 -- Results Overview: `docs/results/README.md` `[COMPLETED]`

**Before writing, the agent MUST read:**
- `outputs/eaai_metrics.json`
- `outputs/eaai_metrics.csv`
- `docs/Plano_de_Acao_EAAI.md` (Phase 4 report section)

**Required Content:**
1. Title: "Experimental Results"
2. Experimental setup summary:
   - Dataset: MNIST
   - Capacity: 5,000 vectors
   - Latent dimensionality: 128
   - Noise levels: [0.0, 0.2, 0.4, 0.6, 0.8]
   - Evaluation protocol: Prequential (test-then-train)
3. Link to detailed baseline comparison and metrics reference.
4. Navigation footer.

### 6.2 -- Baseline Comparison: `docs/results/baseline-comparison.md` `[COMPLETED]`

**Before writing, the agent MUST read:**
- `outputs/eaai_metrics.json` (extract exact numerical values)
- `docs/Plano_de_Acao_EAAI.md` (Phase 4 analysis section)

**Required Content:**
1. Title: "Baseline Comparison Analysis"
2. Baseline descriptions (B0-B4) with formal definitions.
3. Complete results table (from `eaai_metrics.json`) with ALL metrics:
   - Accuracy under noise (5 levels)
   - Mean accuracy
   - Latency (ms)
   - RAM peak (MB)
   - Cache hit rate (%)
   - KL divergence (nats)
   - Forgetting rate (%/transition)
   - Drift restoration time (steps)
4. Key findings analysis (from Phase 4 report, translated to English):
   - Robustness under severe noise.
   - Cache pollution prevention.
   - Distribution matching.
   - Operational sustainability (LMOS).
5. Statistical significance discussion.
6. Navigation footer.

### 6.3 -- Metrics Reference: `docs/results/metrics-reference.md` `[COMPLETED]`

**Required Content:**
1. Title: "Engineering Metrics Reference"
2. For EACH of the 7 metrics:
   - Formal name.
   - Mathematical definition (LaTeX).
   - Unit of measurement.
   - What it measures (intuitive explanation).
   - Scientific source/reference.
   - How it is computed in the codebase (file and function reference).
3. Table summary:

| Metric | Unit | Range | Source |
|:-------|:-----|:------|:-------|
| Accuracy under Noise | % | [0, 100] | Standard |
| Processing Latency | ms | [0, inf) | Standard |
| RAM Peak Usage | MB | [0, inf) | LMOS (Jain et al., 2022) |
| Forgetting Rate | %/transition | [0, 100] | Haug et al. (2022) |
| Drift Restoration Time | steps | [0, inf) | Haug et al. (2022) |
| Cache Hit Rate | % | [0, 100] | RLCache (Alabed, 2019) |
| Eviction KL Divergence | nats | [0, inf) | Isele & Cosgun (2018) |

4. Navigation footer.

**Acceptance Criteria:**
- [x] Directory `docs/results/` exists with 3 files.
- [x] All numerical values match `outputs/eaai_metrics.json` exactly.
- [x] All 7 engineering metrics are formally defined.
- [x] Analysis section is translated accurately from the Portuguese Phase 4 report.
- [x] No emojis.
- [x] GPL-3.0 reference.

---

# PHASE 7: Final Verification and Cleanup `[COMPLETED]`

> [!NOTE]
> **Status:** Completed.
> **Resolution Note:** Verified link integrity across all 27 documentation files (0 broken links). Confirmed 100% concordance of API signatures against source code via AST analysis, exact numerical parity of all benchmark values against `outputs/eaai_metrics.json`, and exact correspondence of configuration constants against `src/config.py`. Verified complete bidirectional navigation footers across all documents. Verified zero emoji characters, uniform academic English language, and ubiquitous GNU GPL-3.0 licensing notices. Relocated 7 historical Portuguese documentation files to `docs/_archive/` with a comprehensive index (`docs/_archive/README.md`), while preserving `docs/Plano_de_Acao_EAAI.md` and `docs/Literatura/`. Updated `docs/README.md` to reflect the completed documentation tree.
> **Depends on:** All previous phases (0-6).
> **Unblocks:** Nothing (terminal phase).

**Objective:** Verify the integrity of the entire documentation tree and clean up the old Portuguese documentation files.

**Tasks:**

### 7.1 -- Link Verification `[COMPLETED]`
- [x] The agent MUST traverse every documentation file and verify that all internal links (relative paths) point to existing files.
- [x] Report any broken links (0 broken links detected).

### 7.2 -- Content Verification `[COMPLETED]`
- [x] Verify that all code signatures in the API reference match the actual source code.
- [x] Verify that all numerical values in the results documentation match `outputs/eaai_metrics.json`.
- [x] Verify that all configuration parameters in the configuration reference match `src/config.py`.

### 7.3 -- Navigation Consistency `[COMPLETED]`
- [x] Verify that every document has a navigation footer.
- [x] Verify that the "Previous/Up/Next" links form a consistent traversal.

### 7.4 -- Style Verification `[COMPLETED]`
- [x] Verify zero emoji characters across all documentation files.
- [x] Verify English language throughout (no Portuguese remnants except in `docs/Literatura/` which is preserved).
- [x] Verify GPL-3.0 reference in every document.

### 7.5 -- Old Documentation Disposition `[COMPLETED]`
- [x] The existing Portuguese documentation files in `docs/` (outside `docs/Literatura/`) contain valuable technical content that has been incorporated into the new English documentation.
- [x] Move these files to `docs/_archive/` to preserve them without cluttering the documentation tree:
  - `docs/arquitetura_e_fluxo_cnn.md`
  - `docs/caso_de_estudo_eaai.md`
  - `docs/evaluate_hybrid_global.md`
  - `docs/fluxo_dados_sistema_hibrido.md`
  - `docs/knn_bandit_agent_128d.md`
  - `docs/train_rl_128d.md`
  - `docs/train_rl_online_simulation.md`
- [x] Keep `docs/Plano_de_Acao_EAAI.md` in place (it is the master implementation plan and should be preserved as a historical record).

### 7.6 -- Update `docs/README.md` `[COMPLETED]`
- [x] Update the documentation index to reflect the final state of all documentation.
- [x] Mark all phases as `[COMPLETED]` in THIS plan document.

**Acceptance Criteria:**
- [x] Zero broken links across all documentation files.
- [x] Zero emoji characters.
- [x] All signatures, values, and parameters are accurate.
- [x] Old Portuguese docs moved to `docs/_archive/`.
- [x] `docs/README.md` reflects the final documentation tree.
- [x] This plan document has all phases marked `[COMPLETED]`.

---

## EXECUTION INSTRUCTIONS FOR THE AGENT

1. **Read this plan completely** before starting any phase.
2. **Execute one phase at a time** when the user requests it.
3. **Before writing any documentation file**, read ALL source files listed in the "Source Files" table for that component. Do not write documentation from memory or assumption.
4. **After completing each phase**, update the status marker in this plan from `[PENDING]` to `[COMPLETED]` and add a brief resolution note.
5. **Verify all internal links** are valid relative paths before declaring a phase complete.
6. **Never use emojis.** Not in titles, not in lists, not in status markers. Use `[COMPLETED]`, `[PENDING]`, `[IN PROGRESS]` text markers only.
7. **All content in English.** The only exception is the preserved `docs/Literatura/` directory.
8. **Mathematical notation** uses LaTeX syntax (`$...$` for inline, `$$...$$` for display).
9. **Code examples** must be syntactically correct and include the source file path as a comment.
10. **Navigation footers** must be present in every documentation file.

---

Licensed under the GNU General Public License v3.0.
