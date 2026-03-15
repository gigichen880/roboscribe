# Research Survey: AI Agents for Converting Natural Language to Robot Actions in Simulation

**Date:** 2026-02-27
**Focus:** Language-to-action systems for robotic manipulation, with emphasis on robosuite-compatible approaches
**Framework:** Python / PyTorch / robosuite (MuJoCo)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Robosuite Landscape](#2-robosuite-landscape)
3. [LLM-Based Robot Planning Approaches](#3-llm-based-robot-planning-approaches)
4. [Language-Conditioned Robot Learning](#4-language-conditioned-robot-learning)
5. [Task and Motion Planning with Language](#5-task-and-motion-planning-with-language)
6. [Robosuite-Specific Implementations](#6-robosuite-specific-implementations)
7. [Creative / Novel Approaches](#7-creative--novel-approaches)
8. [Open-Source Implementations](#8-open-source-implementations)
9. [Feasibility Analysis for This Project](#9-feasibility-analysis-for-this-project)
10. [Recommended Architecture](#10-recommended-architecture)
11. [Sources](#11-sources)

---

## 1. Executive Summary

The field of language-grounded robotic manipulation has exploded since 2022, driven by advances in large language models and vision-language models. The approaches broadly fall into three paradigms:

| Paradigm | Core Idea | Examples | Training Required? |
|---|---|---|---|
| **LLM-as-Planner** | LLM decomposes language into action sequences or code | SayCan, Code as Policies, ProgPrompt, VoxPoser | No (zero-shot) |
| **End-to-End VLA** | Single model maps (vision, language) to actions | RT-2, OpenVLA, VIMA, CLIPort | Yes (large-scale) |
| **LLM + RL Hybrid** | LLM generates rewards/guides RL training | Eureka, Text2Reward, LARAP | Yes (RL training) |

For a robosuite-based project that must handle at least 3 distinct tasks (e.g., "stack the blocks," "sort by size"), the **LLM-as-Planner** paradigm is the most feasible starting point, while **hybrid approaches** offer the most creative angles.

---

## 2. Robosuite Landscape

### Available Environments (v1.5)

Robosuite provides nine standardized benchmark tasks built on MuJoCo:

**Single-Arm Tasks:**
- **Lift** -- Pick up a single cube from the table
- **Stack** -- Place cube A on top of cube B (two cubes, randomized positions)
- **NutAssembly** -- Insert nuts onto correct pegs (two pegs, two nuts)
- **PickPlace** -- Pick objects and place in correct receptacles (up to 4 objects)
- **Door** -- Turn handle and open a door
- **Wipe** -- Wipe markers off a surface

**Two-Arm Tasks:**
- **TwoArmHandover** -- Pass object between two arms
- **TwoArmLift** -- Cooperatively lift a large object
- **TwoArmPegInHole** -- Cooperative peg insertion

**Key robosuite features relevant to this project:**
- Modular APIs for custom environment creation
- Multiple robot models (Panda, Sawyer, IIWA, Jaco, UR5e, and more in v1.5)
- Procedural generation of object layouts
- Standardized observation spaces (joint positions, object poses, images)
- Integrated with robomimic for imitation learning

**Repository:** https://github.com/ARISE-Initiative/robosuite

---

## 3. LLM-Based Robot Planning Approaches

### 3.1 SayCan (Google, 2022)

**Core Idea:** Couples a pre-trained LLM with an affordance function. The LLM proposes candidate actions from a pre-defined skill library, and the affordance model scores which actions are physically feasible in the current state. The final action is selected by multiplying LLM probability with affordance score.

**Architecture:**
```
Language Instruction
        |
    [LLM] --> candidate actions with probabilities
        |
    [Affordance Model] --> feasibility scores
        |
    Selected Skill --> Execute
        |
    Loop until task complete
```

**Pros:**
- Grounds LLM outputs in physical reality
- Handles long-horizon tasks by iterative skill selection
- Does not require fine-tuning the LLM

**Cons:**
- Requires a pre-trained library of manipulation skills (value functions)
- Each skill needs its own affordance model
- Closed-set: cannot generalize beyond the predefined skill set
- Expensive to set up (need demonstrations for every skill)

**Feasibility for robosuite:** MODERATE. You would need to pre-train individual skills (e.g., pick, place, push) in robosuite via RL or imitation learning, then use an LLM to sequence them. The skill training is the bottleneck.

**Novelty:** Foundational work but now somewhat dated. The affordance grounding idea remains influential.

**Paper:** https://say-can.github.io/
**Code:** Not officially open-sourced in full, but the concept is reproducible.

---

### 3.2 Code as Policies (Google, Liang et al., 2023)

**Core Idea:** Instead of having the LLM output natural language plans, have it directly write Python code that calls robot primitive APIs. The code can include loops, conditionals, arithmetic (via NumPy, Shapely), and perception calls (e.g., `get_obj_pos("red_block")`).

**Architecture:**
```
Language Instruction + Few-shot Code Examples (prompt)
        |
    [LLM (Codex/GPT-4)] --> Python code
        |
    [Execute code] --> calls perception APIs + control primitives
        |
    Robot executes
```

**Pros:**
- Extremely flexible: code can express complex logic, spatial reasoning, loops
- Zero-shot generalization to novel instructions
- No training required
- Transparent and debuggable (you can read the generated code)

**Cons:**
- Relies on having well-defined primitive APIs (pick, place, move_to, etc.)
- LLM hallucinations can produce syntactically valid but semantically wrong code
- No physical grounding -- the LLM does not know what is actually feasible
- Error recovery is limited

**Feasibility for robosuite:** HIGH. This is one of the most directly implementable approaches. You define a set of primitive functions that wrap robosuite controllers (e.g., `move_to(pos)`, `grasp()`, `release()`, `get_object_position(name)`), then prompt GPT-4 to write code that composes these primitives.

**Novelty:** The code-generation paradigm is now well-established. To make it novel, you could add verification/self-correction loops.

**Paper:** https://code-as-policies.github.io/
**Code:** https://github.com/google-research/google-research/tree/master/code_as_policies

---

### 3.3 ProgPrompt (Singh et al., 2023)

**Core Idea:** Provides the LLM with a Pythonic program header (available actions, objects, preconditions) and example task programs. The LLM generates new task programs as Python functions. Crucially, it adds assertion-based precondition checking: if a precondition fails (e.g., robot not near fridge), the system triggers recovery actions.

**Architecture:**
```
Pythonic Prompt (imports, object list, example functions)
        |
    [LLM] --> def new_task(): action sequence with assertions
        |
    [Execute with precondition checks]
        |
    Failed assertion? --> recovery action --> retry
```

**Pros:**
- Situated awareness through precondition assertions
- 100% executability rate in physical robot experiments
- Recovers from failures via assertion-triggered replanning
- Clean, interpretable program structure

**Cons:**
- Requires manual specification of available actions and preconditions
- Limited to tasks expressible as linear action sequences with assertions
- Does not handle continuous control or force-sensitive tasks

**Feasibility for robosuite:** HIGH. Very natural fit. Define robosuite action primitives as Python functions with assertions about robot/object states. The assertion-based recovery is a strong differentiator.

**Novelty:** The precondition-assertion pattern is elegant and underutilized in recent work.

**Paper:** https://progprompt.github.io/
**Code:** https://github.com/tan90cot0/progprompt-vh

---

### 3.4 VoxPoser (Huang et al., 2023)

**Core Idea:** Uses LLMs to write code that interacts with a vision-language model (VLM) to generate 3D affordance maps and constraint maps in voxel space. These 3D value maps are composed and used as cost functions for motion planning. No training required.

**Architecture:**
```
Language Instruction + RGB-D Observation
        |
    [LLM] --> Code that queries VLM for spatial regions
        |
    [VLM] --> 3D affordance maps + constraint maps
        |
    [Compose into value map]
        |
    [Motion Planner (MPC)] --> Optimal trajectory
        |
    Robot executes
```

**Pros:**
- Zero-shot, no training
- Combines semantic understanding (LLM) with spatial grounding (VLM + 3D voxels)
- Handles spatial language naturally ("to the left of," "between")
- Composable: different maps for different constraints can be combined

**Cons:**
- Requires RGB-D perception pipeline
- Computationally expensive (voxelization + VLM inference)
- Motion planner may fail in tight spaces
- Complex setup with multiple model dependencies

**Feasibility for robosuite:** MODERATE-HIGH. Robosuite provides depth images, so you can reconstruct voxel grids. The main challenge is integrating a VLM for spatial grounding. Could be simplified by using ground-truth object positions from robosuite instead of VLM inference.

**Novelty:** The 3D value map composition is a genuinely creative idea. Adapting it to use robosuite's state information (instead of raw perception) could be an interesting simplification.

**Paper:** https://voxposer.github.io/
**Code:** https://github.com/huangwl18/VoxPoser

---

### 3.5 Inner Monologue (Huang et al., 2022)

**Core Idea:** An LLM generates action plans, but continuously receives textual feedback from the environment (success detection, scene descriptions, human corrections). This "inner monologue" allows the LLM to adaptively replan based on what actually happened.

**Architecture:**
```
Language Instruction
        |
    [LLM] --> proposed action
        |
    [Execute] --> observe result
        |
    [Scene Descriptor] --> text description of new state
    [Success Detector] --> "success" / "failure: block fell"
    [Human Feedback] --> optional corrections
        |
    [Feed all back to LLM] --> next action
        |
    Loop until done
```

**Pros:**
- Closed-loop: adapts to execution failures
- Incorporates multiple feedback modalities as text
- Robust to unexpected outcomes
- Natural framework for human-in-the-loop correction

**Cons:**
- Requires good scene description and success detection modules
- Many LLM API calls (one per action step)
- Latency from repeated LLM inference
- No learned low-level control

**Feasibility for robosuite:** HIGH. Robosuite provides ground-truth state information that can be trivially converted to textual scene descriptions. Success detection is built into robosuite's reward functions. This is very implementable.

**Novelty:** The feedback loop pattern is now standard, but combining it with robosuite's rich state information in creative ways (e.g., using force/torque feedback as text) could be novel.

**Paper:** https://inner-monologue.github.io/

---

## 4. Language-Conditioned Robot Learning

### 4.1 CLIPort (Shridhar et al., 2022)

**Core Idea:** Combines CLIP's semantic understanding with TransporterNet's spatial precision for language-conditioned tabletop manipulation. Uses a two-stream architecture: one stream identifies "what" to manipulate (semantic), the other identifies "where" to place it (spatial).

**Architecture:**
```
Language Goal + RGB Image
        |
    [CLIP Encoder] --> semantic features ("what")
    [TransporterNet] --> spatial features ("where")
        |
    [Fuse] --> pick location + place location
        |
    Execute pick-and-place
```

**Pros:**
- Strong spatial precision from TransporterNet
- Semantic generalization from CLIP
- Works well for tabletop pick-and-place tasks
- Open-source with good documentation

**Cons:**
- Limited to pick-and-place actions (no pushing, rotating, tool use)
- Requires demonstration data for training
- 2D action space only (top-down pick and place)
- Does not handle 6-DOF manipulation

**Feasibility for robosuite:** LOW-MODERATE. CLIPort is designed for its own Ravens simulation environment, not robosuite. Porting would require significant effort. However, the architecture concepts could inspire a robosuite-native solution.

**Novelty:** The CLIP + spatial precision combination is clever but well-known.

**Paper:** https://cliport.github.io/
**Code:** https://github.com/cliport/cliport

---

### 4.2 VIMA (Jiang et al., ICML 2023)

**Core Idea:** A multimodal prompt-based approach where diverse manipulation tasks are expressed as interleaved sequences of text and images (multimodal prompts). A single transformer model processes these prompts and outputs robot actions.

**Architecture:**
```
Multimodal Prompt: "Put the [IMAGE: red block] into the [IMAGE: blue bowl]"
        |
    [T5 Encoder] --> prompt embeddings (text + image tokens)
        |
    [Cross-Attention Decoder] --> conditioned on current observation
        |
    Action output (delta position, rotation, gripper)
```

**Pros:**
- Unified interface for many task types (visual goal, text goal, one-shot imitation, etc.)
- 17 benchmark tasks in VIMA-Bench
- Multiple model sizes available (from 2M to 200M parameters)
- Fully open-source with pre-trained checkpoints

**Cons:**
- Trained in its own VIMA-Bench simulator (not robosuite)
- Requires a large dataset (650K trajectories)
- The multimodal prompt format is specific to VIMA's design
- Does not use real physics simulation (simplified)

**Feasibility for robosuite:** MODERATE. VIMA's simulator is simpler than robosuite. You could adopt the multimodal prompt architecture but would need to collect training data in robosuite. The training data requirement is substantial.

**Novelty:** The multimodal prompt paradigm (mixing text and image tokens) is creative. Adapting this concept to robosuite with a smaller model could be interesting.

**Paper:** https://vimalabs.github.io/
**Code:** https://github.com/vimalabs/VIMA
**Benchmark:** https://github.com/vimalabs/VIMABench

---

### 4.3 PerAct (Shridhar et al., 2023)

**Core Idea:** An end-to-end behavior cloning agent that uses a Perceiver-Transformer to process 3D voxelized observations and language goals. Predicts next-best-action as a voxel location + rotation + gripper state.

**Architecture:**
```
Language Goal + Multi-view RGB-D
        |
    [Voxelize scene into 3D grid]
        |
    [Split into 3D patches] + [Language features from CLIP]
        |
    [PerceiverIO Transformer] --> per-voxel action predictions
        |
    Select voxel with highest action value --> execute
```

**Pros:**
- Full 6-DOF action prediction
- Works with few demonstrations per task (5-10)
- Outperforms 2D approaches significantly (34x over image-to-action baselines)
- Language-conditioned multi-task learning

**Cons:**
- Requires multi-view RGB-D setup
- Computationally expensive (voxelization + transformer)
- Trained in RLBench, not robosuite
- Relatively complex to set up

**Feasibility for robosuite:** MODERATE. Would require adapting the perception pipeline to robosuite's camera setup. The core transformer architecture is transferable.

**Paper:** https://peract.github.io/
**Code:** https://github.com/peract/peract

---

### 4.4 OpenVLA (Kim et al., 2024)

**Core Idea:** An open-source 7B-parameter Vision-Language-Action model that directly outputs robot actions from images and language instructions. Built on Llama 2 with DINOv2 + SigLIP visual encoders. Trained on 970K trajectories from Open X-Embodiment.

**Architecture:**
```
Camera Image + Language Instruction
        |
    [DINOv2 + SigLIP] --> visual tokens
    [Llama 2] --> language + action tokens
        |
    Output: discretized action tokens
```

**Pros:**
- Open-source (MIT license), pre-trained checkpoints available
- Outperforms RT-2-X (55B) with 7x fewer parameters
- Supports LoRA fine-tuning for new embodiments
- FAST tokenizer (2025) enables 15x faster inference

**Cons:**
- 7B parameters is still very large for a student project
- Requires significant compute for fine-tuning
- Trained on real-robot data, may not transfer well to sim
- Action discretization can lose precision

**Feasibility for robosuite:** LOW-MODERATE. Fine-tuning a 7B model on robosuite data is expensive. However, using a pre-trained OpenVLA as a component in a larger system (e.g., for high-level reasoning) is more feasible.

**Paper:** https://openvla.github.io/
**Code:** https://github.com/openvla/openvla
**Model:** https://huggingface.co/openvla/openvla-7b

---

## 5. Task and Motion Planning with Language

### 5.1 SayPlan (Rana et al., 2023)

**Core Idea:** Uses 3D scene graphs to ground LLM-based planning in large-scale environments. The LLM conducts semantic search over a hierarchical scene graph representation, then generates plans that are verified by a scene graph simulator.

**Architecture:**
```
Language Instruction + 3D Scene Graph
        |
    [Collapse graph to high-level representation]
        |
    [LLM] --> semantic search for relevant subgraph
        |
    [LLM] --> step-by-step plan on relevant subgraph
        |
    [Scene Graph Simulator] --> verify plan feasibility
        |
    If invalid --> replan with feedback
        |
    [Classical Path Planner] --> motion execution
```

**Pros:**
- Scales to large environments (multi-floor, multi-room)
- Iterative verification loop catches LLM errors
- Hierarchical graph structure reduces LLM context length
- Demonstrated on real mobile manipulator

**Cons:**
- Requires a 3D scene graph (not trivial to build)
- Focused on navigation + high-level manipulation, not fine-grained control
- Complex multi-component pipeline

**Feasibility for robosuite:** LOW. SayPlan is designed for large-scale navigation environments, not tabletop manipulation. However, the idea of using a scene graph representation for robosuite's objects could be adapted at a smaller scale.

**Paper:** https://sayplan.github.io/

---

### 5.2 LLM-GROP (Zhang et al., 2025)

**Core Idea:** Combines LLM common-sense reasoning with classical TAMP solvers. The LLM generates object goal configurations (where things should go), and a TAMP solver computes physically feasible trajectories to achieve those configurations.

**Architecture:**
```
Language Instruction + Scene Observation
        |
    [LLM] --> goal object configuration (semantic)
        |
    [Visual Grounding] --> map to physical coordinates
        |
    [TAMP Solver] --> feasible motion plan
        |
    Execute
```

**Pros:**
- Combines LLM semantic understanding with physical feasibility guarantees
- TAMP solver ensures collision-free paths
- Strong on spatial rearrangement tasks

**Cons:**
- TAMP solvers can be slow
- Requires geometric models of all objects
- Complex integration between LLM and TAMP components

**Feasibility for robosuite:** MODERATE. Robosuite provides full geometric information. You could use an LLM to generate target configurations and a motion planner to execute. PyBullet or OMPL could serve as the motion planner.

**Paper:** https://arxiv.org/abs/2511.07727

---

## 6. Robosuite-Specific Implementations

### 6.1 robomimic -- Language-Conditioned Policies (ARISE Initiative)

**Core Idea:** robomimic v0.5 directly supports training language-conditioned manipulation policies in robosuite. It uses CLIP embeddings to encode language instructions and conditions diffusion policies or transformer policies on these embeddings.

**Architecture:**
```
Language Instruction --> [CLIP] --> language embedding
Robot Observation (joint state, images)
        |
    [Diffusion Policy / Transformer Policy]
        |
    conditioned on language embedding
        |
    Output: action sequence
```

**Pros:**
- Native robosuite integration (no porting required)
- Multiple policy architectures supported (BC-RNN, Diffusion Policy, Transformer)
- Well-documented tutorials
- Active community and maintenance
- Pre-trained visual representations available

**Cons:**
- Requires demonstration data collection for each task
- Language conditioning is via CLIP embeddings (not full language understanding)
- Does not do planning -- purely reactive policy

**Feasibility for robosuite:** VERY HIGH. This is the most directly applicable tool. It is literally designed for robosuite + language-conditioned policies.

**Paper:** https://robomimic.github.io/
**Code:** https://github.com/ARISE-Initiative/robomimic
**Tutorial:** https://robomimic.github.io/docs/tutorials/language_conditioned.html

---

### 6.2 Task Decomposition with RL in Robosuite (2024)

**Core Idea:** High-level tasks in robosuite (door opening, block stacking, nut assembly) are manually decomposed into subtasks, and SAC (Soft Actor-Critic) is trained separately for each subtask with task-specific reward shaping.

**Architecture:**
```
High-level task (e.g., "stack blocks")
        |
    [Manual Decomposition] --> subtask 1: reach block A
                              subtask 2: grasp block A
                              subtask 3: move above block B
                              subtask 4: place on block B
        |
    [SAC] trained per subtask with shaped rewards
        |
    Sequence subtask policies
```

**Pros:**
- Uses standard robosuite environments directly
- Well-understood RL approach
- Clear subtask structure

**Cons:**
- Manual decomposition (no language understanding)
- Requires RL training for each subtask
- No generalization to new tasks

**Feasibility for robosuite:** HIGH (but not language-grounded without additional work).

**Paper:** https://pmc.ncbi.nlm.nih.gov/articles/PMC11047822/

---

### 6.3 GenManip: LLM-Driven Simulation (CVPR 2025)

**Core Idea:** An LLM-driven simulation platform for evaluating instruction-following manipulation policies. Uses LLMs to automatically generate task-oriented scene graphs from natural language, which are then instantiated as simulation environments with 10K+ annotated 3D objects.

**Architecture:**
```
Natural Language Task Description
        |
    [LLM] --> Task-Oriented Scene Graph
        |
    [Scene Instantiation] --> 3D simulation environment
        |
    [Policy Evaluation] --> modular or end-to-end
```

**Pros:**
- Automatic task and scene generation from language
- 10,000+ object assets
- Covers long-horizon planning, spatial reasoning, commonsense, and appearance tasks
- Full open-source CVPR 2025 paper

**Cons:**
- Uses its own simulation setup (not directly robosuite)
- Focus is on benchmarking, not on the agent itself
- Complex infrastructure

**Feasibility for robosuite:** MODERATE. The task generation pipeline could be adapted to create diverse robosuite environments from language descriptions. The LLM-driven scene generation concept is very applicable.

**Paper:** https://arxiv.org/abs/2506.10966
**Code:** https://github.com/InternRobotics/GenManip

---

## 7. Creative / Novel Approaches

### 7.1 Eureka: LLM-Generated Reward Functions (ICLR 2024)

**Core Idea:** Instead of having humans write reward functions for RL, use an LLM (GPT-4) to generate reward function code. The LLM receives the environment source code as context, generates candidate reward functions, evaluates them via RL training, and iteratively improves them through evolutionary search and self-reflection.

**Architecture:**
```
Environment Source Code + Task Description
        |
    [GPT-4] --> N candidate reward functions (Python code)
        |
    [RL Training (GPU-accelerated)] --> evaluate each reward
        |
    [Reward Reflection] --> summarize what worked/failed
        |
    [GPT-4] --> improved reward functions (next generation)
        |
    Iterate until convergence
```

**Pros:**
- Fully automated reward design
- Outperforms human experts on 83% of tasks
- Works across diverse robot morphologies (29 environments)
- No demonstrations needed
- Open-source

**Cons:**
- Requires significant compute for RL training loop
- Multiple LLM calls per iteration
- Works best with IsaacGym (GPU-accelerated RL), not natively with robosuite
- The evolutionary search can be slow

**Feasibility for robosuite:** MODERATE-HIGH. The concept directly applies: give GPT-4 the robosuite environment code, have it generate reward functions, train RL agents. The main challenge is RL training speed in robosuite (slower than IsaacGym).

**Novelty:** VERY HIGH. This is one of the most creative approaches. Adapting it to robosuite would be genuinely interesting.

**Paper:** https://eureka-research.github.io/
**Code:** https://github.com/eureka-research/Eureka

---

### 7.2 Text2Reward: Language-Based Reward Shaping (ICLR 2024 Spotlight)

**Core Idea:** Given a natural language goal, an LLM generates dense reward function code that can be used to train RL policies. Unlike Eureka's evolutionary approach, Text2Reward focuses on generating interpretable, shaped dense rewards with support for iterative human feedback refinement.

**Architecture:**
```
Natural Language Goal + Environment Description
        |
    [LLM] --> Dense reward function (Python code)
        |
    [RL Training] --> policy
        |
    [Optional: Human Feedback] --> refine reward
        |
    Iterate (< 3 rounds to go from 0% to ~100% success)
```

**Pros:**
- Interpretable reward code (not a black box)
- Works on MuJoCo-based environments
- Iterative refinement with human feedback
- 13/17 manipulation tasks match or beat expert rewards
- Open-source

**Cons:**
- Still requires RL training time
- Human feedback loop needed for difficult tasks
- Generated rewards may not capture all task nuances

**Feasibility for robosuite:** HIGH. Text2Reward is already designed for MuJoCo-based environments. Adapting to robosuite is straightforward.

**Novelty:** HIGH. The language-to-reward pipeline is a compelling alternative to language-to-plan.

**Paper:** https://text-to-reward.github.io/
**Code:** https://github.com/xlang-ai/text2reward

---

### 7.3 MALMM: Multi-Agent LLM for Zero-Shot Manipulation (IROS 2025)

**Core Idea:** Three specialized LLM agents (Planner, Coder, Supervisor) collaborate to solve manipulation tasks zero-shot. The Planner generates high-level plans, the Coder converts plans to executable code, and the Supervisor coordinates and triggers replanning on failure.

**Architecture:**
```
Language Instruction
        |
    [Supervisor Agent] --> delegates to Planner
        |
    [Planner Agent] --> high-level step-by-step plan
        |
    [Coder Agent] --> executable code for each step
        |
    [Code Executor] --> run in simulation
        |
    Success? --> done
    Failure? --> [Supervisor] triggers replan
```

**Pros:**
- Zero-shot (no training or demonstrations)
- Handles long-horizon tasks through decomposition
- Adaptive replanning on failure
- Separation of concerns (planning vs. coding vs. coordination)

**Cons:**
- Multiple LLM API calls (expensive)
- Evaluated on RLBench, not robosuite
- Complex multi-agent orchestration
- Performance depends heavily on LLM quality

**Feasibility for robosuite:** HIGH. The multi-agent architecture is simulator-agnostic. Adapting to robosuite requires defining the code execution primitives for robosuite. This would be a strong project architecture.

**Novelty:** HIGH. The multi-agent separation is elegant and relatively new.

**Paper:** https://arxiv.org/abs/2411.17636
**Project:** https://malmm1.github.io/

---

### 7.4 LARAP: LLM-Augmented Hierarchical RL (2025)

**Core Idea:** Combines LLM planning with hierarchical RL and action primitives. The LLM proposes likely skill sequences to guide exploration, while RL learns to actually execute the skills. The LLM serves as a "warm start" for the RL agent rather than the sole decision-maker.

**Architecture:**
```
Task Description + Current State
        |
    [LLM] --> suggested skill sequence (warm start)
        |
    [High-Level RL Policy] --> selects from action primitives
        |   (guided by LLM suggestions early in training)
        |
    [Low-Level Skill Execution] --> primitive controllers
        |
    Environment feedback --> update RL policy
```

**Pros:**
- Combines LLM reasoning with learned control
- Addresses LLM grounding problem through RL refinement
- More sample-efficient than pure RL (LLM-guided exploration)
- Handles long-horizon tasks

**Cons:**
- Still requires RL training
- Need to define action primitives
- LLM suggestions may mislead early training

**Feasibility for robosuite:** MODERATE-HIGH. Robosuite + RL is well-studied. Adding LLM guidance on top is a natural extension.

**Novelty:** HIGH. The LLM-as-exploration-guide concept is creative.

**Paper:** https://www.nature.com/articles/s41598-025-20653-y

---

### 7.5 FAEA: Frontier Agent as Embodied Agent (2025)

**Core Idea:** Apply an unmodified LLM agent framework (like Claude Agent SDK) directly to embodied manipulation. The agent iteratively reasons through manipulation strategies, executes actions, observes results, and refines its approach -- all through natural language reasoning in context, without any training or fine-tuning.

**Architecture:**
```
Task Instruction + Environment API
        |
    [LLM Agent] --> reason about strategy
        |
    Execute action --> observe result (state, error, success)
        |
    [LLM Agent] --> reason about outcome, plan next action
        |
    Iterate until success or max steps
```

**Pros:**
- Truly zero-shot, demonstration-free
- 84.9% - 96% success rates on benchmarks
- No training, no fine-tuning, no demonstrations
- Can generate training data for other models
- Leverages frontier LLM reasoning capabilities directly

**Cons:**
- Requires privileged environment state access for best results
- Very high API cost (many LLM calls per episode)
- Slow inference (not real-time)
- Evaluated in ManiSkill3/MetaWorld/LIBERO, not robosuite

**Feasibility for robosuite:** HIGH. Robosuite provides state access. You can implement this directly: give an LLM agent access to robosuite's action space and observation space, and let it iteratively solve tasks. This is perhaps the simplest approach to implement.

**Novelty:** VERY HIGH. This is a cutting-edge 2025 approach that challenges the assumption that embodied agents need specialized training.

**Paper:** https://arxiv.org/abs/2601.20334

---

### 7.6 Language to Rewards (Google, 2023)

**Core Idea:** LLMs translate user instructions into reward parameters for MuJoCo MPC (Model Predictive Control). A "Reward Translator" interprets language into motion descriptions, then into reward code. MuJoCo's built-in MPC optimizer then finds optimal actions.

**Architecture:**
```
User Instruction: "do a backflip"
        |
    [LLM: Motion Description] --> "rotate body 360 degrees backward while maintaining height"
        |
    [LLM: Reward Coding] --> reward_function(state): maximize angular velocity, penalize ground contact
        |
    [MuJoCo MPC] --> optimal action sequence
        |
    Execute
```

**Pros:**
- Leverages MuJoCo's MPC for optimal control
- No RL training needed (online optimization)
- Works for locomotion and manipulation
- Can express complex behaviors through reward composition

**Cons:**
- MPC is computationally expensive
- Requires access to full state for MPC
- Limited to tasks expressible as reward functions
- MuJoCo MPC may not scale to very complex manipulation

**Feasibility for robosuite:** HIGH. Robosuite is MuJoCo-based. The reward coding approach is directly applicable. You would use robosuite's state information for MPC optimization.

**Novelty:** HIGH. Combining language-to-reward with MPC is elegant.

**Paper:** https://language-to-reward.github.io/

---

### 7.7 GenCHiP: Code Generation for Contact-Rich Manipulation (2024)

**Core Idea:** LLMs generate policy code specifically for high-precision, contact-rich tasks by reparameterizing the action space to include compliance parameters (stiffnesses, force limits). This allows the generated code to handle delicate operations like insertion and assembly.

**Architecture:**
```
Task Description + Compliance-Aware API
        |
    [LLM] --> policy code with force/stiffness parameters
        |
    [Impedance Controller] --> execute with compliance
        |
    Handle contact forces gracefully
```

**Pros:**
- Handles tasks that pure position control cannot (insertion, assembly)
- 3-4x improvement over position-only code generation
- Addresses a key limitation of Code as Policies

**Cons:**
- Requires impedance/compliance control framework
- Specific to contact-rich tasks
- Limited evaluation scope

**Feasibility for robosuite:** MODERATE. Robosuite supports OSC (Operational Space Control) which provides some compliance. Extending this with explicit force control would require controller modifications.

**Paper:** https://arxiv.org/abs/2404.06645

---

### 7.8 VLM-Based Reward Shaping (2025)

**Core Idea:** Use vision-language models to compute reward signals directly from visual observations and language task descriptions. The VLM scores how well the current scene matches the language goal, providing a dense reward signal for RL without manual reward engineering.

**Approaches include:**
- **ReWiND:** Language-conditioned reward learning without per-task demonstrations
- **SARM:** Stage-aware reward modeling for long-horizon tasks
- **VLAC:** Vision-Language-Action-Critic that generates both rewards and actions

**Feasibility for robosuite:** MODERATE-HIGH. Robosuite can render camera images. A VLM can score these images against language goals. This creates a self-supervised training loop.

**Novelty:** HIGH. Using VLMs as reward models is a frontier research direction.

---

## 8. Open-Source Implementations

### Directly Usable Repositories

| Repository | Description | License | Stars | Robosuite Compatible? |
|---|---|---|---|---|
| [robomimic](https://github.com/ARISE-Initiative/robomimic) | Framework for robot learning from demonstration with language conditioning | MIT | 500+ | YES (native) |
| [robosuite](https://github.com/ARISE-Initiative/robosuite) | Simulation framework with 9 benchmark tasks | MIT | 1.4K+ | YES (it IS robosuite) |
| [VIMA](https://github.com/vimalabs/VIMA) | Multimodal prompt robot manipulation | MIT | 600+ | No (own sim) |
| [VoxPoser](https://github.com/huangwl18/VoxPoser) | 3D value maps for manipulation | Apache 2.0 | 400+ | Adaptable |
| [PerAct](https://github.com/peract/peract) | 3D voxel transformer for manipulation | Apache 2.0 | 400+ | Adaptable |
| [CLIPort](https://github.com/cliport/cliport) | CLIP + TransporterNet | Apache 2.0 | 600+ | No (Ravens sim) |
| [OpenVLA](https://github.com/openvla/openvla) | 7B VLA model | MIT | 2K+ | Adaptable |
| [Eureka](https://github.com/eureka-research/Eureka) | LLM reward design | MIT | 2K+ | Adaptable (IsaacGym) |
| [Text2Reward](https://github.com/xlang-ai/text2reward) | LLM reward shaping | -- | 300+ | Adaptable (MuJoCo) |
| [GenManip](https://github.com/InternRobotics/GenManip) | LLM-driven simulation for manipulation | -- | New | Conceptually applicable |
| [ProgPrompt](https://github.com/tan90cot0/progprompt-vh) | Programmatic LLM prompting | -- | 100+ | Adaptable |
| [Awesome-LLM-Robotics](https://github.com/GT-RIPL/Awesome-LLM-Robotics) | Curated paper list | MIT | 4K+ | Reference |

### Key Tutorials and Documentation

- **robomimic Language-Conditioned Policies:** https://robomimic.github.io/docs/tutorials/language_conditioned.html
- **robosuite Getting Started:** https://robosuite.ai/docs/
- **PerAct Colab Tutorial:** Available at https://github.com/peract/peract
- **Code as Policies Interactive Demo:** https://code-as-policies.github.io/

---

## 9. Feasibility Analysis for This Project

### Project Requirements Recap

1. Parse at least 3 distinct task instructions ("stack the blocks," "sort by size," etc.)
2. Generate a sequence of low-level actions or subgoals
3. Successfully execute in robosuite simulation
4. Clear mapping from language -> plan -> control

### Feasibility Ranking of Approaches

| Approach | Implementation Effort | Compute Cost | Novelty | Task Diversity | Recommended? |
|---|---|---|---|---|---|
| **Code as Policies + robosuite** | LOW | LOW (API calls only) | MEDIUM | HIGH | YES |
| **Multi-Agent LLM (MALMM-style)** | MEDIUM | MEDIUM | HIGH | HIGH | YES |
| **ProgPrompt + robosuite** | LOW | LOW | MEDIUM | HIGH | YES |
| **Inner Monologue + robosuite** | LOW-MEDIUM | MEDIUM | MEDIUM | HIGH | YES |
| **LLM Reward Gen + RL (Eureka-style)** | HIGH | HIGH | VERY HIGH | MEDIUM | CREATIVE OPTION |
| **FAEA-style Direct Agent** | LOW | HIGH (many API calls) | VERY HIGH | HIGH | CREATIVE OPTION |
| **robomimic Language-Conditioned** | MEDIUM | MEDIUM | LOW | MEDIUM | BASELINE OPTION |
| **VoxPoser adaptation** | HIGH | HIGH | HIGH | HIGH | AMBITIOUS OPTION |
| **OpenVLA fine-tuning** | VERY HIGH | VERY HIGH | MEDIUM | HIGH | NOT RECOMMENDED |

---

## 10. Recommended Architecture

Based on this survey, here is a recommended architecture that balances feasibility, novelty, and the project requirements:

### Proposed: "Hierarchical LLM Agent with Self-Verifying Execution in Robosuite"

This combines the best ideas from multiple approaches:

```
                    Natural Language Instruction
                              |
                    [Task Parser / Planner Agent]
                    (LLM: GPT-4 / Claude / LLaMA)
                              |
                    Structured Plan (subtask sequence)
                              |
                    [Code Generator Agent]
                    (LLM generates Python code per subtask)
                              |
                    Executable code calling robosuite primitives
                              |
                    [Execution Engine]
                    robosuite environment + primitive API
                              |
                    [State Verifier]
                    Check postconditions after each subtask
                              |
                    Success? --> next subtask
                    Failure? --> feedback to Planner --> replan
```

**Key Components:**

1. **Primitive Action Library** (wrap robosuite controllers):
   - `move_to(position)` -- move end-effector to target
   - `grasp()` / `release()` -- gripper control
   - `get_object_pose(name)` -- query object positions
   - `get_gripper_state()` -- check if holding something

2. **Task Planner** (LLM-based):
   - Takes natural language + scene state as text
   - Outputs ordered list of subtasks with preconditions/postconditions
   - Supports at least: Stack, PickPlace, NutAssembly (3+ tasks)

3. **Code Generator** (LLM-based):
   - Converts each subtask into executable Python using the primitive library
   - Includes assertion-based verification (ProgPrompt-inspired)

4. **Closed-Loop Executor:**
   - Runs generated code in robosuite
   - Monitors state after each action
   - Reports failures back to planner for replanning (Inner Monologue-inspired)

**What makes this creative:**
- Combines the multi-agent separation of MALMM with the assertion-based recovery of ProgPrompt
- Adds Inner Monologue-style feedback for adaptive replanning
- Can be extended with Eureka-style reward generation for RL-based skill improvement
- The self-verifying execution loop is not commonly seen in robosuite implementations

**Minimum viable tasks:**
1. **Stack** -- "Stack the red block on the green block" (uses robosuite Stack env)
2. **PickPlace** -- "Put the milk in the bin" (uses robosuite PickPlace env)
3. **NutAssembly** -- "Put the round nut on the round peg" (uses robosuite NutAssembly env)

Optional extensions for creativity:
- "Sort the objects by color" (compound PickPlace with reasoning)
- "Clear the table" (multi-object sequential manipulation)
- LLM-generated reward functions for learning new skills

---

## 11. Sources

### Papers
- [SayCan: Grounding Language in Robotic Affordances](https://say-can.github.io/)
- [Code as Policies: Language Model Programs for Embodied Control](https://code-as-policies.github.io/)
- [ProgPrompt: Generating Situated Robot Task Plans using Large Language Models](https://progprompt.github.io/)
- [VoxPoser: Composable 3D Value Maps for Robotic Manipulation](https://voxposer.github.io/)
- [Inner Monologue: Embodied Reasoning through Planning with Language Models](https://inner-monologue.github.io/)
- [CLIPort: What and Where Pathways for Robotic Manipulation](https://cliport.github.io/)
- [VIMA: General Robot Manipulation with Multimodal Prompts](https://vimalabs.github.io/)
- [PerAct: Perceiver-Actor for Multi-Task Robotic Manipulation](https://peract.github.io/)
- [OpenVLA: An Open-Source Vision-Language-Action Model](https://openvla.github.io/)
- [SayPlan: Grounding LLMs using 3D Scene Graphs](https://sayplan.github.io/)
- [LLM-GROP: Visually Grounded Robot Task and Motion Planning](https://arxiv.org/abs/2511.07727)
- [Eureka: Human-Level Reward Design via Coding LLMs](https://eureka-research.github.io/)
- [Text2Reward: Reward Shaping with Language Models for RL](https://text-to-reward.github.io/)
- [Language to Rewards for Robotic Skill Synthesis](https://language-to-reward.github.io/)
- [MALMM: Multi-Agent LLMs for Zero-Shot Robotic Manipulation](https://malmm1.github.io/)
- [GenManip: LLM-driven Simulation for Generalizable Manipulation (CVPR 2025)](https://arxiv.org/abs/2506.10966)
- [GenCHiP: Generating Robot Policy Code for Contact-Rich Tasks](https://arxiv.org/abs/2404.06645)
- [FAEA: Demonstration-Free Robotic Control via LLM Agents](https://arxiv.org/abs/2601.20334)
- [LARAP: LLMs Augmented Hierarchical RL with Action Primitives](https://www.nature.com/articles/s41598-025-20653-y)
- [Large Language Models for Robotics: Opportunities, Challenges, and Perspectives](https://www.sciencedirect.com/science/article/pii/S2949855424000613)

### Code Repositories
- [robosuite](https://github.com/ARISE-Initiative/robosuite)
- [robomimic](https://github.com/ARISE-Initiative/robomimic)
- [VIMA](https://github.com/vimalabs/VIMA)
- [VIMABench](https://github.com/vimalabs/VIMABench)
- [VoxPoser](https://github.com/huangwl18/VoxPoser)
- [PerAct](https://github.com/peract/peract)
- [CLIPort](https://github.com/cliport/cliport)
- [OpenVLA](https://github.com/openvla/openvla)
- [Eureka](https://github.com/eureka-research/Eureka)
- [Text2Reward](https://github.com/xlang-ai/text2reward)
- [GenManip](https://github.com/InternRobotics/GenManip)
- [ProgPrompt (VirtualHome)](https://github.com/tan90cot0/progprompt-vh)
- [Awesome-LLM-Robotics](https://github.com/GT-RIPL/Awesome-LLM-Robotics)
- [RT-2 (Community)](https://github.com/kyegomez/RT-2)
