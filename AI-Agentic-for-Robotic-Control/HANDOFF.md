# UI-Fix Branch Handoff for Bree

## TL;DR

I rewrote the UI to support the new `ToolAgentLoop` (two-phase agent with tool use). The agent logic works, but the UI code is rough — I got it functional with Claude's help but it needs a proper frontend engineer to clean up. This doc explains what changed, what's hacky, and what needs work.

**Branch:** `ui-fix`
**Main file you'll work in:** `roboscribe/src/roboscribe/ui/app.py`
**Agent file (backend, mostly stable):** `roboscribe/src/roboscribe/agent/tool_loop.py`

---

## What Changed vs Main

### The Agent (backend) — new, works well

Main branch had `AgentLoop` — a rigid loop: generate code → simulate → diagnose → revise, repeat.

The new `ToolAgentLoop` is split into two steps the UI can control:

```
Step 1: run_phase_design(task)
   → auto-detect environment
   → introspect obs shapes / reward / success condition
   → LLM designs a phase plan (JSON: name/goal/control/exit_condition per phase)
   → returns PhaseDesignResult for human review

Step 2: run_with_phases(task, env, phases, introspection)
   → LLM generates initial policy code from approved phases
   → tests in simulation (robosuite subprocess, frames via shared memory)
   → if <80% success: diagnoses, then enters tool-use loop
   → LLM can call: test_policy, read_robosuite_source, submit_policy
   → human feedback injected between turns (45s timeout, optional)
   → returns ToolAgentResult
```

The key design choice: **the UI controls the transition between Step 1 and Step 2**. This lets the user review/edit the phase plan before code generation starts.

### The UI — functional but rough

Rewrote from a single-page layout to a **4-step state machine**:

```
Describe → Phase Review → Generating → Results
```

Each step is a branch in an `if/elif` chain (line ~302). Only one step renders per Streamlit rerun cycle.

---

## Architecture: How the UI Talks to the Agent

The agent runs in a **background thread** (Streamlit is single-threaded, so we can't block). Communication:

```
UI thread (Streamlit reruns every 120ms)
    │
    │  ←── _event_queue (Queue) ──── Agent thread
    │       typed tuples: ("frame", data), ("tool", data),
    │       ("status", data), ("code", data), ("video", data),
    │       ("diag", data), ("turn", N), ("awaiting_feedback", bool),
    │       ("error", traceback), ("done", result)
    │
    │  ──── feedback_queue (Queue) ──→ Agent thread
    │       user typed feedback string, or None
    │
    │  ──── _cancel_event (threading.Event) ──→ Agent thread
    │       set by Stop button, checked in on_tool_call / get_feedback
```

The `_drain_events()` function (line ~155) runs each rerun, reads all events from the queue, and updates `st.session_state`. Only the latest frame is kept (others discarded) to avoid backlog.

### Why not just use callbacks directly?

`st.session_state` is NOT accessible from background threads. Any callback that needs to update the UI must go through a queue. All session state values needed by the thread (`task`, `env_name`, `phase_plan`, `introspection_str`) are captured as local variables before `threading.Thread` starts (line ~589).

### Why a single event queue instead of separate queues?

The original code had 8 separate queues (frame, tool_log, code, video, diag, feedback, log, status). Each was created, stored, and drained independently — easy to lose messages if any drain was missed. One queue with typed events is simpler and nothing gets lost.

### Thread cancellation

The Stop button sets `_cancel_event` and pushes `None` into `feedback_queue` (to unblock `get_feedback` if it's waiting). The thread checks the event in two safe places:

- `on_tool_call` — after each simulation finishes (shared memory already cleaned up)
- `get_feedback` — when paused waiting for user input

**Never raise from `on_frame`** — the `SimulationRunner.run_policy()` frame callback runs inside a shared-memory loop (runner.py:252-258). Raising there skips cleanup at line 266-270, leaking shared memory.

---

## Known Issues / Hacks

### 1. Streamlit icon font bug
Streamlit's Material Icons font sometimes fails to load, causing icon names like `arrow_down`, `keyboard_double_arrow_right` to render as literal text. **Workaround:** I removed all `st.expander` usage and replaced with `st.checkbox` (code viewer) or `st.caption` (phase plan). If you bring back expanders, this bug will return.

### 2. Ghost widgets between steps
Even with `if/elif`, users sometimes see widgets from a previous step persisting on screen (e.g., the generating page's feedback input appearing on the phase review page). This seems to be a Streamlit DOM caching issue. A hard refresh (Cmd+Shift+R) clears it, but it shouldn't happen in the first place. May need to investigate Streamlit's widget lifecycle or add explicit `st.empty()` containers.

### 3. Phase review → Generating transition
The "Send" button does double duty: empty text = approve & start, text = revise phases. This is functional but potentially confusing UX. A dedicated "Start" button might be clearer.

### 4. Init/redesign still use old queue pattern
The "initializing" (phase design) and "redesign phases" steps still use the old `log_queue` + `status_queue` + `_holder` dict pattern, while "generating" uses the new unified event queue. Ideally these should be unified, but they're simple one-shot operations so it's low priority.

### 5. Stop doesn't kill the simulation subprocess instantly
The cancellation waits for the current simulation to finish (a few seconds per episode). The robosuite subprocess runs via `multiprocessing.Process` with a `stop_event`, but we don't have access to it from outside `run_policy()`. For instant cancellation, the runner would need to expose the stop event.

### 6. No back navigation from Generating
Once generation starts, the user can only Stop → Results → Start Over. There's no "Back to Phase Review" button. Adding one would require cancelling the thread first.

### 7. The code is in one big file
`app.py` is ~900 lines. The step rendering, thread management, event handling, and helpers are all mixed together. Splitting into separate modules (e.g., `steps/describe.py`, `steps/generating.py`, `components/video_player.py`) would be much cleaner.

---

## File Map

```
roboscribe/src/roboscribe/
├── ui/
│   └── app.py                    ← THE FILE (Streamlit UI, ~900 lines)
├── agent/
│   ├── tool_loop.py              ← ToolAgentLoop (two-phase agent, stable)
│   ├── tools.py                  ← Tool definitions (test_policy, read_source, submit)
│   ├── prompts.py                ← All LLM prompts (system, phase design, generation)
│   ├── loop.py                   ← Old AgentLoop (main branch, not used anymore)
│   └── few_shot.py               ← Reference policies for few-shot examples
├── sim/
│   ├── runner.py                 ← SimulationRunner (robosuite subprocess + shared memory)
│   ├── introspect.py             ← Environment introspection (obs shapes, reward)
│   ├── diagnostics.py            ← Failure diagnosis
│   └── env_registry.py           ← Environment definitions (Lift, Door, NutAssembly...)
├── llm/
│   ├── base.py                   ← LLM response types (LLMToolResponse)
│   ├── openai_backend.py         ← OpenAI/Qwen backend (with tool use)
│   └── anthropic_backend.py      ← Anthropic backend (with tool use)
├── pid.py                        ← PID + RotationPID controllers
├── config.py                     ← Config + API key management
├── .streamlit/config.toml        ← Dark theme (MUST be at CWD, not project root)
└── output/
    └── writer.py                 ← Save policy files
```

---

## How to Run

```bash
cd roboscribe/src/roboscribe
streamlit run ui/app.py
```

The `.streamlit/config.toml` must be in the CWD where Streamlit starts (`roboscribe/src/roboscribe/`), NOT the project root.

