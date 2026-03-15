# RoboScribe

AI agent that generates robosuite scripted policies from natural language descriptions.

Tell the robot what to do in plain English, and RoboScribe writes the control code, tests it in simulation, diagnoses what went wrong, and revises — until it works.

```
"pick up the cube"  →  generate code  →  simulate  →  diagnose  →  revise  →  working policy
```

## Getting Started

### 1. Install

```bash
# From the repo root:
pip install -e "roboscribe/[sim]"         # includes robosuite for simulation

# Or from inside the package directory:
cd roboscribe && pip install -e ".[sim]"
```

### 2. Set your LLM API key

You only need **one** provider. Pick whichever you have access to:

```bash
export DASHSCOPE_API_KEY=sk-...      # Qwen (free tier available — good default)
# OR
export OPENAI_API_KEY=sk-...         # OpenAI
# OR
export ANTHROPIC_API_KEY=sk-ant-...  # Anthropic
# OR
export DEEPSEEK_API_KEY=sk-...       # DeepSeek
```

Verify it's detected:

```bash
roboscribe backends
```

**Tip — save your keys so you don't have to set them every time:**

You can create a `.env` file in the `roboscribe/` directory:

```bash
# roboscribe/.env
DASHSCOPE_API_KEY=sk-...
OPENAI_API_KEY=sk-...
```

RoboScribe loads this file automatically on startup. You can also save keys directly from the web UI sidebar (see below). The `.env` file is in `.gitignore` so your keys are never committed.

### 3. Run it

You can use RoboScribe two ways: through the **web UI** (recommended for first-time users) or the **command line**.

---

## Option A: Web UI (Streamlit)

The web UI walks you through a simple three-step flow: **Describe → Generate → Results**.

### Launch the app

```bash
pip install streamlit              # one-time setup
cd roboscribe/src/roboscribe
streamlit run ui/app.py
```

This opens a browser tab at `http://localhost:8501`.

### First time? Save your API keys

Click the **sidebar arrow** (top-left) to open the settings panel. Enter your API key(s) and click **Save**. They're stored in a local `roboscribe/.env` file and loaded automatically every time you launch the app — no need to re-enter them or set environment variables.

### Step 1 — Describe

You'll see a form with:

- **Task description** — type what the robot should do in plain English
  (e.g., "Pick up the cube and lift it above the table")
- **Environment** — select a robosuite environment (Lift, Stack, Door, etc.).
  The UI shows the goal and available objects for each one.
- **Robot** — currently Panda (more coming)
- **LLM Backend** — choose your provider (qwen, openai, anthropic, deepseek)
- **Review each attempt** — optional. When checked, the agent pauses after each failed attempt so you can describe what you saw. Your feedback guides the next revision. You can also set how many seconds to wait before auto-continuing.

Click **Generate Policy** to start.

### Step 2 — Generate

The agent works in a loop:

1. Writes a policy using the LLM
2. Runs it in simulation (you'll see a live video feed)
3. Diagnoses what went wrong (e.g., "Robot missed the grasp")
4. Revises and tries again

You'll see:
- A **progress bar** showing attempt N of M
- A **live simulation view** with an overlay (attempt, episode, step, best rate)
- **Previous attempts** with color-coded results (green = good, yellow = partial, red = failed) and plain-English descriptions of what happened and what the agent is doing next
- A **Stop button** if you want to end early

If you enabled human review, the agent pauses after each failed attempt and shows:
- The last simulation frame
- The diagnosis in plain English
- A text box for your observations (e.g., "The gripper approaches from the wrong angle")
- **Send feedback** and **Skip** buttons, plus an auto-continue countdown

### Step 3 — Results

When the agent finishes (or you stop it), you'll see:

- A **success/warning banner** with the final success rate
- **Metrics**: best success rate, number of attempts, output file path
- A **progress chart** showing success rate across attempts
- The **generated policy code** displayed inline
- A **Download** button to save the policy as a `.py` file
- **Attempt history** (collapsed) with detailed breakdown of each attempt — diagnosis, trajectory log, human feedback if any, and the policy code for that attempt
- A **Start over** button to go back to Step 1

---

## Option B: Command Line

### Generate a policy

```bash
roboscribe generate "pick up the cube" --env Lift --backend qwen
```

The agent runs in your terminal — you'll see each attempt, its success rate, and the diagnosis. When it finishes, the policy is saved to a file.

Full options:

```bash
roboscribe generate "pick up the cube" \
    --env Lift \
    --robot Panda \
    --backend qwen \
    --model qwen-plus \
    --max-attempts 5 \
    --episodes 10 \
    --interactive \
    -v
```

| Flag | What it does |
|------|-------------|
| `--env, -e` | Robosuite environment (Lift, Stack, PickPlaceCan, NutAssemblySquare, Door, Wipe) |
| `--robot, -r` | Robot model (default: Panda) |
| `--backend, -b` | LLM provider — `openai`, `anthropic`, `qwen`, or `deepseek` |
| `--model, -m` | Specific model name (auto-selected per backend if omitted) |
| `--base-url` | Custom API base URL for OpenAI-compatible providers |
| `--api-key` | API key (or use environment variables above) |
| `--max-attempts` | Max self-debugging iterations (default: 5) |
| `--episodes` | Episodes per evaluation (default: 10) |
| `--interactive, -i` | Pause between iterations so you can review and give feedback |
| `--verbose, -v` | Show detailed output |

### Test a saved policy

```bash
roboscribe test lift_policy.py --episodes 20
```

### Record a video

```bash
roboscribe record lift_policy.py --env Lift -o demo.mp4

# Different camera angles
roboscribe record lift_policy.py --env Lift --camera frontview -o front.mp4
roboscribe record lift_policy.py --env Lift --camera sideview -o side.mp4

# Multiple episodes
roboscribe record lift_policy.py --env Lift -n 3 -o multi_episode.mp4
```

Requires: `pip install imageio imageio-ffmpeg`

### Other commands

```bash
roboscribe envs        # list all supported environments
roboscribe backends    # check which LLM providers are configured
```

---

## Supported Environments

| Environment | Task | Objects |
|---|---|---|
| Lift | Pick up a cube | cube |
| Stack | Stack red cube on green cube | cubeA, cubeB |
| PickPlaceCan | Pick can, place in bin | can, bin |
| NutAssemblySquare | Place nut on peg | square_nut, peg |
| Door | Open door via handle | door, handle |
| Wipe | Wipe markers off table | markers |

## Supported LLM Backends

| Backend | Flag | Default Model | API Key Env Var |
|---------|------|---------------|-----------------|
| Qwen | `--backend qwen` | qwen-plus | `DASHSCOPE_API_KEY` |
| OpenAI | `--backend openai` | gpt-4o | `OPENAI_API_KEY` |
| Anthropic | `--backend anthropic` | claude-sonnet-4-20250514 | `ANTHROPIC_API_KEY` |
| DeepSeek | `--backend deepseek` | deepseek-chat | `DEEPSEEK_API_KEY` |

Any OpenAI-compatible API can also be used via `--base-url`.

## How It Works

1. **Generate** — the LLM writes a `get_action(obs)` function using few-shot examples and environment context
2. **Simulate** — the policy runs in robosuite via subprocess (isolates crashes and hangs)
3. **Diagnose** — failures are categorized (code error, timeout, missed grasp, no movement, etc.)
4. **Revise** — the LLM receives the diagnosis + trajectory data and writes a corrected policy
5. **Repeat** — the loop continues until success rate exceeds 80% or max attempts are reached

## Generated Policy Format

Policies are standalone Python files that define a `get_action(obs)` function:

```python
import numpy as np

APPROACH, LOWER, GRASP, LIFT = 0, 1, 2, 3
state = APPROACH

def get_action(obs):
    global state
    eef_pos = obs["robot0_eef_pos"]
    cube_pos = obs["cube_pos"]
    action = np.zeros(7)  # OSC_POSE: [dx, dy, dz, dax, day, daz, gripper]
    # ... state machine logic ...
    return action

def reset():
    global state
    state = APPROACH
```

You can use the generated file directly with `roboscribe test` and `roboscribe record`, or integrate it into your own robosuite pipeline.

## Architecture

```
roboscribe/
├── cli.py              # Click CLI (generate, test, record, envs, backends)
├── config.py           # Configuration + provider registry
├── agent/
│   ├── loop.py         # Generate → simulate → diagnose → revise loop
│   ├── prompts.py      # LLM prompt templates
│   ├── few_shot.py     # Hand-written example policies
│   └── interactive.py  # Human-in-the-loop review (CLI)
├── llm/
│   ├── base.py         # Abstract backend + code extraction
│   ├── openai_backend.py   # OpenAI + compatible providers (Qwen, DeepSeek)
│   ├── anthropic_backend.py
│   └── factory.py
├── sim/
│   ├── runner.py       # Subprocess simulation execution
│   ├── recorder.py     # Video recording (MP4)
│   ├── env_registry.py # Environment descriptions
│   ├── trajectory.py   # Result dataclass
│   └── diagnostics.py  # Failure categorization
├── ui/
│   └── app.py          # Streamlit web interface
└── output/
    └── writer.py       # Standalone policy file writer
```

## Requirements

- Python 3.10+
- An LLM API key (OpenAI, Anthropic, Qwen, or DeepSeek)
- robosuite >= 1.4 (install with `pip install -e ".[sim]"`)
- MuJoCo (required by robosuite)
- Streamlit (for web UI: `pip install streamlit`)
- imageio + imageio-ffmpeg (for video recording)
