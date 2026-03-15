# AI Agentic Robotic Control — Proposal Summary

**RoboScribe**: Self-Debugging Code Generation for Automated Scripted Policy Synthesis in Robosuite

---

## 1. Problem Statement

### Why It Matters

Scripted policies (hand-coded robot controllers) are **indispensable** in robotics research:
- **Data collection** — imitation learning pipelines (robomimic, MimicGen) need thousands of demos; scripted policies generate these at scale
- **Baselines** — every new learned policy is compared against a deterministic scripted baseline
- **Environment validation** — sanity-check that a new sim environment is physically solvable
- **Rapid prototyping** — verify task design before investing GPU-hours in training

**The pain**: Writing one scripted policy (e.g., NutAssembly) takes hours of manual trial-and-error — tuning waypoints, grasp timings, approach angles, force thresholds. This multiplies across dozens of task-robot combos.

**The gap**: robosuite is a top-5 manipulation benchmark (1,400+ GitHub stars, 9 tasks, 5+ robots), yet has **zero** LLM-based tooling. Eureka → IsaacGym, MALMM → RLBench, FAEA → LIBERO/ManiSkill, Text2Reward → MetaWorld. Nobody serves robosuite.

### Our Contribution

**RoboScribe** = first AI agent that automates scripted policy generation for robosuite:
1. First LLM agent framework for robosuite
2. Simulation-based self-debugging loop (generate → simulate → diagnose → revise)
3. Human-supervised autonomy (agent debugs, human approves)
4. Open-source pip-installable tool

---

## 2. Proposed Method

### Pipeline (4-stage closed loop)

```
Natural Language Task Description
        ↓
┌─── Stage 1: Prompt Builder ───┐
│  API ref + env desc + examples │
│  + (prev code + diagnosis)     │
└───────────┬───────────────────┘
            ↓
┌─── Stage 2: LLM Code Gen ────┐
│  Generates get_action(obs)→a  │
│  State machine + P-control    │
└───────────┬───────────────────┘
            ↓
┌─── Stage 3: Simulation Run ──┐
│  Sandboxed subprocess         │
│  K episodes, collect (s,a,r,f)│
└───────────┬───────────────────┘
            ↓
        Success ≥ 80%?
       /           \
     Yes            No
      ↓              ↓
  Output .py    Stage 4: Failure Diagnosis
  policy file     ↓ categorize failure type
                  ↓ (CODE_ERROR, TIMEOUT,
                  ↓  MISSED_GRASP, etc.)
                  ↓ feed back to Stage 1
                  └──→ loop (up to N iters)
```

**Human supervisor** can inspect/redirect at any stage.

### Why This Architecture

| Decision | Rationale |
|----------|-----------|
| **Code gen, not reward gen** | Eureka/Text2Reward produce rewards → need GPU-hours of RL training. We produce immediately-runnable .py files. |
| **Self-debugging, not single-pass** | Code as Policies fails on contact-rich tasks. Simulation feedback reveals *where* and *why* — enabling targeted fixes. |
| **Subprocess isolation** | LLM code often has syntax errors, infinite loops, segfaults. Subprocess prevents agent crash. |
| **Human supervision** | Full autonomy = expensive + unreliable. Full manual = slow. Middle ground: agent debugs, human guides. |

### Key Equations

Action vector: `a_t = [Δx, Δy, Δz, Δαx, Δαy, Δαz, g] ∈ ℝ⁷`

Proportional control: `a_pos = Kp · (x_target - x_eef)`

Revision loop: `π^(n+1) = LLM(ℓ, E, π^(n), Diag(ξ^(n)))`

---

## 3. Literature Review

### Has Anyone Done This Before?

**No.** The comparison table:

| System | Output | Sim Debug | Force/Torque | Robosuite | Training-free |
|--------|--------|-----------|--------------|-----------|---------------|
| Code as Policies | Policy code | — | — | — | ✓ |
| ProgPrompt | Task plans | — | — | — | ✓ |
| Inner Monologue | Skill sequences | ✓ | — | — | ✓ |
| MALMM | Policy code | ✓ | — | — | ✓ |
| Besimulator | Policy code | text only | — | — | ✓ |
| Eureka | Reward code | ✓ | — | — | — |
| Text2Reward | Reward code | ✓ | — | — | — |
| FAEA | Actions | ✓ | — | — | ✓ |
| Plan-Seq-Learn | Learned policy | ✓ | — | ✓ | — |
| **RoboScribe (Ours)** | **Policy code** | **✓** | **✓** | **✓** | **✓** |

**Closest systems and why they don't solve our problem:**
- **Code as Policies** — no sim feedback, no self-debugging
- **Besimulator** — uses *mental* (text-based) simulation, not real physics
- **MALMM** — targets RLBench, no force-torque
- **Eureka** — generates rewards (not policies), requires RL training, targets IsaacGym
- **Plan-Seq-Learn** — only robosuite system, but needs RL training, not code gen

### Codebases of Related Work

- Code as Policies: `github.com/google-research/google-research/tree/master/code_as_policies`
- ProgPrompt: `github.com/tan90cot0/progprompt-vh`
- Eureka: `github.com/eureka-research/Eureka`
- Text2Reward: `github.com/xlang-ai/text2reward`
- OpenVLA: `github.com/openvla/openvla`
- MALMM: `malmm1.github.io`
- robomimic: `github.com/ARISE-Initiative/robomimic`

### What's Creative About Ours

1. **Simulation as unit test** — use robosuite as a test suite for generated code (TDD for robot policies), not as a training environment
2. **Multi-modal failure diagnosis** — combine 4 feedback channels (task reward, code errors, force-torque anomalies, state-machine phase analysis). Force-torque + LLM code revision = zero prior work in any simulator
3. **Right abstraction level** — scripted policies are the "Goldilocks" target: structurally predictable, inherently verifiable, immediately useful, human-readable. No prior work identified this.
4. **Filling a verified gap** — systematic analysis (18+ approaches, 15+ surveys, 14 integration gaps) confirms robosuite has zero LLM tooling. Even two-arm tasks have zero language baselines.

---

## 4. Evaluation Plan

**Tasks** (increasing complexity):
1. **Lift** — pick up a cube (sanity check)
2. **Door** — turn handle, open door (precise positioning)
3. **Stack** — place cube A on cube B (two-object coordination)
4. **PickPlace** — pick objects, place in receptacles (multi-object reasoning)
5. **NutAssembly** — insert nuts onto pegs (contact-rich, force-sensitive)

**Metrics:**
- Task success rate (50 randomized seeds per task)
- Iterations to convergence (how many debug rounds to reach ≥80%)
- Wall-clock time (vs. estimated manual dev time)
- Ablation: single-pass vs. self-debugging, structured vs. raw feedback
