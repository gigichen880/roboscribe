"""
RoboScribe Streamlit UI — 4-step flow with phase-design-first architecture.

Steps:  [Describe] → [Phase Review] → [Generating] → [Results]
"""

import multiprocessing
multiprocessing.set_start_method("spawn", force=True)

import streamlit as st
import numpy as np
import cv2
import threading
import queue
import time
import traceback
import os
import re
import tempfile

from roboscribe.config import Config, save_env_vars, get_env_file_path
from roboscribe.agent.tool_loop import ToolAgentLoop, PhaseDesignResult
from roboscribe.sim.env_registry import ENV_REGISTRY

st.set_page_config(page_title="RoboScribe", layout="wide", initial_sidebar_state="collapsed")

# ── Global style ──────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');
  html, body, [class*="st-"] { font-family: 'IBM Plex Sans', sans-serif; }
  .block-container { padding-top: 1.2rem !important; padding-bottom: 1rem !important; max-width: 1100px; }
  #MainMenu, footer { visibility: hidden; }
  [data-testid="metric-container"] {
    background: #080c10; border: 1px solid #1c2a3a; border-radius: 6px; padding: 10px 14px;
  }
  textarea { font-family: 'IBM Plex Mono', monospace !important; font-size: 0.82rem !important; }
  video { border-radius: 8px; }
  hr { opacity: 0.1 !important; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar: API keys ────────────────────────────────────
_KEY_VARS = [
    ("DASHSCOPE_API_KEY", "Qwen (DashScope)"),
    ("OPENAI_API_KEY", "OpenAI"),
    ("ANTHROPIC_API_KEY", "Anthropic"),
    ("DEEPSEEK_API_KEY", "DeepSeek"),
]
with st.sidebar:
    st.markdown("### API Keys")
    st.caption("Keys are saved to a local `.env` file.")
    env_path = get_env_file_path()
    if env_path.exists():
        st.caption(f"Loaded from `{env_path}`")
    _new_vals: dict[str, str] = {}
    for var, label in _KEY_VARS:
        current = os.environ.get(var, "")
        val = st.text_input(label, value=current, type="password",
                            key=f"_setting_{var}", placeholder="sk-...")
        _new_vals[var] = val.strip()
    if st.button("Save", use_container_width=True):
        to_save = {k: v for k, v in _new_vals.items() if v != os.environ.get(k, "")}
        if to_save:
            path = save_env_vars(to_save)
            st.success(f"Saved to `{path.name}`")
        else:
            st.info("No changes to save.")


# ─────────────────────────────────────────
# CANCELLATION
# ─────────────────────────────────────────

class _CancelledError(Exception):
    """Raised inside the generation thread when the user clicks Stop."""
    pass


# ─────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────

DEFAULTS = {
    # Step machine
    "ui_step":          "describe",   # describe | initializing | phase_review | generating | results
    # Config
    "task":             "",
    "robot":            "Panda",
    "backend":          "qwen",
    # Phase design
    "env_name":         "",
    "env_desc":         "",
    "phase_plan":       [],
    "introspection_str": "",
    # Generation
    "result":           None,
    "gen_error":        "",
    "gen_done":         False,
    "tool_log":         [],
    "current_code":     "",
    "current_status":   "",
    "log":              [],
    "current_turn":     0,
    "awaiting_feedback": False,
    # Sim display
    "latest_frame":     None,
    "latest_meta":      (1, 0, 0),
    "latest_best":      None,
    "frame_count":      0,
    # Video
    "attempt_videos":   {},
    "latest_video":     "",
    # Feedback
    "feedback_queue":   None,
    # Diagnosis
    "diag_problem":     "",
    "diag_suggestion":  "",
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# Simple queues for init/redesign steps (not generation)
if "log_queue" not in st.session_state:
    st.session_state.log_queue = queue.Queue()
if "status_queue" not in st.session_state:
    st.session_state.status_queue = queue.Queue()


# ─────────────────────────────────────────
# THREAD-SAFE LOGGING (init/redesign steps)
# ─────────────────────────────────────────

def add_log(msg: str, _lq=None):
    ts = time.strftime("%H:%M:%S")
    q = _lq if _lq is not None else st.session_state.log_queue
    q.put_nowait(f"[{ts}] {msg}")

def set_status(msg: str, _lq=None, _sq=None):
    sq = _sq if _sq is not None else st.session_state.status_queue
    sq.put_nowait(msg)
    add_log(msg, _lq=_lq)

def _drain_queues():
    """Drain init/redesign log and status queues into session state."""
    lq = st.session_state.log_queue
    while not lq.empty():
        try: st.session_state.log.append(lq.get_nowait())
        except queue.Empty: break
    sq = st.session_state.status_queue
    latest = None
    while not sq.empty():
        try: latest = sq.get_nowait()
        except queue.Empty: break
    if latest is not None:
        st.session_state.current_status = latest


# ─────────────────────────────────────────
# UNIFIED EVENT QUEUE (generation step)
# ─────────────────────────────────────────
# Replaces 6 separate queues with one typed stream.
# Events: ("frame", data) | ("tool", data) | ("code", data)
#         ("video", data) | ("diag", data) | ("status", data)
#         ("log", data)   | ("turn", data) | ("awaiting_feedback", data)
#         ("error", data) | ("done", data)

def _drain_events():
    """Drain the unified event queue into session state."""
    eq = st.session_state.get("_event_queue")
    if not eq:
        return
    latest_frame = None
    while not eq.empty():
        try:
            kind, data = eq.get_nowait()
        except queue.Empty:
            break

        if kind == "frame":
            latest_frame = data          # keep only the most recent frame
            st.session_state.frame_count += 1
        elif kind == "tool":
            st.session_state.tool_log.append(data)
        elif kind == "code":
            st.session_state.current_code = data
        elif kind == "video":
            label, vpath = data
            st.session_state.attempt_videos[label] = vpath
            st.session_state.latest_video = vpath
        elif kind == "diag":
            problem, suggestion = data
            st.session_state.diag_problem = problem
            st.session_state.diag_suggestion = suggestion
        elif kind == "status":
            st.session_state.current_status = data
        elif kind == "log":
            ts = time.strftime("%H:%M:%S")
            st.session_state.log.append(f"[{ts}] {data}")
        elif kind == "turn":
            st.session_state.current_turn = data
        elif kind == "awaiting_feedback":
            st.session_state.awaiting_feedback = data
        elif kind == "error":
            st.session_state.gen_error = data
        elif kind == "done":
            st.session_state.result = data
            st.session_state.gen_done = True

    if latest_frame is not None:
        raw, att, ep, step, best = latest_frame
        st.session_state.latest_frame = raw
        st.session_state.latest_meta = (att, ep, step)
        st.session_state.latest_best = best


# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────

def draw_overlay(frame, attempt, episode, step, best_rate=None):
    frame = np.flipud(frame).copy()
    h, w = frame.shape[:2]
    ov = frame.copy()
    cv2.rectangle(ov, (0, 0), (w, 38), (6, 10, 16), -1)
    cv2.addWeighted(ov, 0.75, frame, 0.25, 0, frame)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, f"Attempt {attempt} | Ep {episode+1} | Step {step}",
                (8, 25), font, 0.5, (160, 175, 200), 1, cv2.LINE_AA)
    if best_rate is not None:
        rtxt = f"{best_rate:.0%}"
        tw = cv2.getTextSize(rtxt, font, 0.5, 1)[0][0]
        col = (80, 200, 120) if best_rate >= 0.8 else (220, 170, 60) if best_rate > 0 else (200, 80, 80)
        cv2.putText(frame, rtxt, (w - tw - 8, 25), font, 0.5, col, 1, cv2.LINE_AA)
    return frame


def extract_phases(code: str) -> list[str]:
    phases = []
    for match in re.finditer(
        r'(?:phase|state|stage|mode)\s*[=]=?\s*["\']([A-Z][A-Z_0-9]+)["\']', code,
    ):
        name = match.group(1)
        if name not in phases:
            phases.append(name)
    return phases


def save_frames_as_video(frames: list[np.ndarray], fps: int = 20) -> str:
    if not frames:
        return ""
    path = os.path.join(tempfile.gettempdir(), f"roboscribe_{time.time():.0f}.mp4")
    try:
        import imageio
        writer = imageio.get_writer(path, fps=fps, codec="libx264",
                                    quality=8, pixelformat="yuv420p")
        for frame in frames:
            writer.append_data(np.flipud(frame))
        writer.close()
    except ImportError:
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
        for frame in frames:
            writer.write(cv2.cvtColor(np.flipud(frame), cv2.COLOR_RGB2BGR))
        writer.release()
    return path


# ─────────────────────────────────────────
# STEP INDICATOR
# ─────────────────────────────────────────

_STEPS = ["Describe", "Phase Review", "Generating", "Results"]
_STEP_MAP = {"describe": 0, "initializing": 0, "phase_review": 1, "generating": 2, "results": 3}

def render_steps():
    current = _STEP_MAP.get(st.session_state.ui_step, 0)
    parts = []
    for i, name in enumerate(_STEPS):
        if i < current:
            parts.append(f"<span style='color:#22c55e'>&#10003; {name}</span>")
        elif i == current:
            parts.append(f"<span style='color:#3b82f6;font-weight:600'>&#9679; {name}</span>")
        else:
            parts.append(f"<span style='color:#334155'>{name}</span>")
    st.markdown(
        "<div style='display:flex;gap:24px;font-family:\"IBM Plex Mono\",monospace;"
        "font-size:0.75rem;margin-bottom:12px'>"
        + "".join(f"<div>{p}</div>" for p in parts) + "</div>",
        unsafe_allow_html=True)

render_steps()


# ═════════════════════════════════════════
# STEP 1: DESCRIBE
# ═════════════════════════════════════════

if st.session_state.ui_step == "describe":  # noqa: chain start

    task = st.text_area(
        "What should the robot do?", height=85,
        placeholder="e.g. lift the cube, open the door, stack cube A on cube B..."
    )

    c1, c2 = st.columns([1.2, 1.2])
    with c1: robot   = st.selectbox("Robot",   ["Panda"], label_visibility="collapsed")
    with c2: backend = st.selectbox("Backend", ["qwen","openai","anthropic","deepseek"], label_visibility="collapsed")

    st.markdown(
        "<div style='display:flex;gap:0;font-size:0.67rem;color:#475569;margin-top:-8px;"
        "font-family:\"IBM Plex Mono\",monospace'>"
        "<div style='flex:1.2'>robot</div>"
        "<div style='flex:1.2'>backend</div></div>",
        unsafe_allow_html=True)

    if st.button("Generate Policy", type="primary", use_container_width=True):
        if not task.strip():
            st.warning("Please describe the task first.")
            st.stop()

        st.session_state.task = task
        st.session_state.robot = robot
        st.session_state.backend = backend
        st.session_state.ui_step = "initializing"
        st.rerun()

    st.stop()


# ═════════════════════════════════════════
# STEP 1.5: INITIALIZING (env detect + introspect + phase design)
# ═════════════════════════════════════════

elif st.session_state.ui_step == "initializing":

    # Only run the background thread once (check for existing holder)
    if "_init_holder" not in st.session_state or st.session_state._init_holder is None:
        try:
            cfg = Config.from_env(
                robot=st.session_state.robot, num_episodes=5, verbose=True,
            )
            cfg.llm_backend = st.session_state.backend
            cfg.validate()
        except Exception as e:
            st.error(f"Config error: {e}")
            st.session_state.ui_step = "describe"
            st.rerun()

        _holder = {"result": None, "done": False, "error": None}
        _log_q = st.session_state.log_queue
        _status_q = st.session_state.status_queue
        _task = st.session_state.task

        loop = ToolAgentLoop(cfg)
        st.session_state._loop = loop
        st.session_state._cfg = cfg

        def _run_phase_design():
            try:
                result = loop.run_phase_design(
                    _task,
                    on_status=lambda msg: set_status(msg, _lq=_log_q, _sq=_status_q),
                )
                _holder["result"] = result
            except Exception as e:
                _holder["error"] = str(e)
                traceback.print_exc()
            _holder["done"] = True

        t = threading.Thread(target=_run_phase_design, daemon=True)
        t.start()
        st.session_state._init_holder = _holder

    _drain_queues()

    # Show loading state
    st.markdown(f"**Task:** {st.session_state.task}")
    if st.session_state.current_status:
        st.info(st.session_state.current_status)
    else:
        st.info("Starting...")

    with st.spinner("Detecting environment, inspecting observations, designing phases..."):
        holder = st.session_state._init_holder
        if holder["done"]:
            if holder["error"]:
                st.error(f"Error: {holder['error']}")
                if st.button("Back to Describe"):
                    st.session_state._init_holder = None
                    st.session_state.ui_step = "describe"
                    st.rerun()
                st.stop()

            result: PhaseDesignResult = holder["result"]
            st.session_state.env_name = result.env_name
            env_info = ENV_REGISTRY.get(result.env_name)
            st.session_state.env_desc = env_info.description if env_info else ""
            st.session_state.phase_plan = result.phase_plan
            st.session_state.introspection_str = result.introspection_str
            st.session_state._init_holder = None
            st.session_state.ui_step = "phase_review"
            st.rerun()
        else:
            time.sleep(0.3)
            st.rerun()

    st.stop()


# ═════════════════════════════════════════
# STEP 2: PHASE REVIEW
# ═════════════════════════════════════════

elif st.session_state.ui_step == "phase_review":

    _drain_queues()

    # Check for redesign result
    rh = st.session_state.get("_redesign_holder")
    if rh is not None and rh["done"]:
        if rh.get("result"):
            st.session_state.phase_plan = rh["result"].phase_plan
        st.session_state._redesign_holder = None
        st.rerun()

    # Show redesign loading
    if rh is not None and not rh["done"]:
        st.info("Redesigning phases with your feedback...")
        _drain_queues()
        time.sleep(0.3)
        st.rerun()

    # Header
    st.markdown(f"**Environment:** {st.session_state.env_name} — {st.session_state.env_desc}")
    st.markdown(f"**Task:** {st.session_state.task}")

    st.markdown("---")
    st.markdown("### Phase Plan")
    st.caption("Review the proposed phases. Edit via the chat below, or approve to start generating code.")

    plan = st.session_state.phase_plan

    if not plan:
        st.warning("No phases were generated. Try regenerating with more details.")
    else:
        for i, p in enumerate(plan):
            with st.container():
                st.markdown(
                    f"**{i}. {p['name']}** — {p['goal']}"
                )
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"Control: `{p.get('control', '-')}`")
                with c2:
                    st.markdown(f"Exit: `{p.get('exit_condition', '-')}`")
                if p.get("notes"):
                    st.caption(f"Notes: {p['notes']}")

        # Flow visualization
        phase_flow = " \u2192 ".join(f"`{p['name']}`" for p in plan)
        st.markdown(f"**Flow:** {phase_flow}")

    st.markdown("---")

    # Single row: [feedback] [Send] [Back]
    # Send with text = revise phases; Send empty = approve & start generation
    if "phase_fb_key" not in st.session_state:
        st.session_state.phase_fb_key = 0

    fb_col, send_col, back_col = st.columns([7, 1.5, 1])
    with fb_col:
        phase_feedback = st.text_input(
            "feedback", label_visibility="collapsed",
            placeholder="Type feedback to revise, or press Send to approve & start...",
            key=f"phase_fb_{st.session_state.phase_fb_key}",
        )
    with send_col:
        send_fb = st.button("Send", type="primary", use_container_width=True, key="send_phase_fb",
                            disabled=len(plan) == 0)
    with back_col:
        back = st.button("Back", use_container_width=True)

    if back:
        st.session_state.ui_step = "describe"
        st.session_state._init_holder = None
        st.rerun()

    if send_fb:
        if not phase_feedback.strip():
            # Empty send = approve & start generation
            st.session_state.ui_step = "generating"
            st.rerun()

        # Has feedback = revise phases
        feedback_text = phase_feedback
        st.session_state.phase_fb_key += 1

        loop = st.session_state.get("_loop")
        if loop is None:
            cfg = Config.from_env(
                robot=st.session_state.robot, num_episodes=5, verbose=True,
            )
            cfg.llm_backend = st.session_state.backend
            cfg.validate()
            loop = ToolAgentLoop(cfg)
            st.session_state._loop = loop
            st.session_state._cfg = cfg

        _holder = {"result": None, "done": False}
        _log_q = st.session_state.log_queue
        _status_q = st.session_state.status_queue

        def _run_redesign():
            try:
                result = loop.redesign_phases(
                    st.session_state.task,
                    st.session_state.env_name,
                    st.session_state.introspection_str,
                    feedback_text,
                    previous_plan=st.session_state.phase_plan,
                    on_status=lambda msg: set_status(msg, _lq=_log_q, _sq=_status_q),
                )
                _holder["result"] = result
            except Exception:
                traceback.print_exc()
            _holder["done"] = True

        t = threading.Thread(target=_run_redesign, daemon=True)
        t.start()
        st.session_state._redesign_holder = _holder
        add_log(f"Redesigning phases: {feedback_text[:60]}")
        st.rerun()

    st.stop()


# ═════════════════════════════════════════
# STEP 3: GENERATING
# ═════════════════════════════════════════

elif st.session_state.ui_step == "generating":

    # ── Launch generation thread (once) ──────────────────────
    if not st.session_state.get("_gen_started"):

        # Reset generation state
        st.session_state.log = []
        st.session_state.tool_log = []
        st.session_state.current_code = ""
        st.session_state.current_status = ""
        st.session_state.latest_frame = None
        st.session_state.latest_video = ""
        st.session_state.attempt_videos = {}
        st.session_state.frame_count = 0
        st.session_state.diag_problem = ""
        st.session_state.diag_suggestion = ""
        st.session_state.current_turn = 0
        st.session_state.awaiting_feedback = False
        st.session_state.gen_error = ""
        st.session_state.gen_done = False
        st.session_state.result = None
        st.session_state.feedback_queue = queue.Queue()

        _event_queue = queue.Queue()
        _cancel_event = threading.Event()
        _feedback_queue = st.session_state.feedback_queue

        st.session_state._event_queue = _event_queue
        st.session_state._cancel_event = _cancel_event

        loop = st.session_state.get("_loop")
        cfg = st.session_state.get("_cfg")

        if loop is None or cfg is None:
            cfg = Config.from_env(
                robot=st.session_state.robot, num_episodes=5, verbose=True,
            )
            cfg.llm_backend = st.session_state.backend
            cfg.validate()
            loop = ToolAgentLoop(cfg)

        # Capture session state values for thread access
        # (st.session_state is NOT accessible from background threads)
        _task = st.session_state.task
        _env_name = st.session_state.env_name
        _phase_plan = st.session_state.phase_plan
        _introspection_str = st.session_state.introspection_str

        _video_frames: list[np.ndarray] = []
        _cb = {"attempt": 0, "best_rate": None, "last_push": 0.0}

        def _emit(kind: str, data=None):
            _event_queue.put_nowait((kind, data))

        def _save_and_push_video(label: str):
            if _video_frames:
                try:
                    vpath = save_frames_as_video(list(_video_frames))
                    if vpath:
                        _emit("video", (label, vpath))
                except Exception:
                    pass
                _video_frames.clear()

        def on_frame(frame, episode, step):
            # NOTE: Never raise from on_frame — it would leak shared memory
            # in SimulationRunner. Just skip frames if cancelled.
            if _cancel_event.is_set():
                return
            _video_frames.append(frame.copy())
            now = time.time()
            if now - _cb["last_push"] < 0.05:
                return
            _cb["last_push"] = now
            _emit("frame", (frame.copy(), _cb["attempt"], episode, step, _cb["best_rate"]))

        def on_tool_call(tool_name, args, result_text):
            # Check cancellation — safe to raise here (sim already finished)
            if _cancel_event.is_set():
                raise _CancelledError()

            if tool_name == "test_policy":
                _cb["attempt"] += 1
                _emit("turn", _cb["attempt"])
                summary = "Testing..."
                if "Success rate:" in result_text:
                    for line in result_text.split("\n"):
                        if "Success rate:" in line:
                            summary = line.strip()
                            break
                elif "ERROR" in result_text:
                    summary = f"Error: {result_text.split(chr(10))[0][:100]}"
                _save_and_push_video(f"attempt_{_cb['attempt']}")
                code = args.get("code", "")
                if code:
                    _emit("code", code)
            elif tool_name == "diagnose_failure":
                lines = result_text.split("\n")
                problem = lines[0] if lines else "Unknown"
                suggestion = ""
                for line in lines:
                    if "Suggest" in line:
                        suggestion = line.split(":", 1)[-1].strip() if ":" in line else line
                        break
                _emit("diag", (problem, suggestion))
                summary = f"Diagnosis: {problem[:100]}"
            elif tool_name == "generate_policy":
                summary = "Generated initial policy"
            elif tool_name == "submit_policy":
                summary = "Submitting final policy"
            elif tool_name == "read_robosuite_source":
                summary = f"Reading: {args.get('module_path', '?')}"
            else:
                summary = f"{tool_name}"

            _emit("tool", {"tool": tool_name, "summary": summary})
            _emit("status", summary)

        def on_submit(code, rate):
            _cb["best_rate"] = rate

        def get_feedback():
            if _cancel_event.is_set():
                raise _CancelledError()
            _emit("status", "Waiting for your feedback (auto-continues in 45s)...")
            _emit("awaiting_feedback", True)
            try:
                fb = _feedback_queue.get(timeout=45)
            except queue.Empty:
                fb = None
            _emit("awaiting_feedback", False)
            if _cancel_event.is_set():
                raise _CancelledError()
            if fb:
                _emit("log", f"Feedback: {fb[:80]}")
            else:
                _emit("log", "No feedback -- auto-continuing")
            return fb

        def _run_generation():
            try:
                _emit("turn", 1)
                result = loop.run_with_phases(
                    _task,
                    _env_name,
                    _phase_plan,
                    _introspection_str,
                    on_tool_call=on_tool_call,
                    on_frame=on_frame,
                    on_status=lambda msg: _emit("status", msg),
                    on_submit=on_submit,
                    get_human_feedback=get_feedback,
                )
            except _CancelledError:
                _emit("log", "Cancelled by user")
                result = None
            except Exception:
                _emit("error", traceback.format_exc())
                _emit("log", "Generation failed with error")
                result = None
            _emit("done", result)

        t = threading.Thread(target=_run_generation, daemon=True)
        t.start()
        st.session_state._gen_started = True
        st.rerun()

    # ── Drain events ──────────────────────────────────────────
    _drain_events()

    # ── Transition to results when done ───────────────────────
    if st.session_state.gen_done:
        st.session_state._gen_started = False
        st.session_state._event_queue = None
        st.session_state._cancel_event = None
        st.session_state.ui_step = "results"
        st.rerun()

    # ── Top bar ───────────────────────────────────────────────
    top_l, top_r = st.columns([8, 1])
    with top_l:
        turn_label = f"  |  Attempt {st.session_state.current_turn}" if st.session_state.current_turn > 0 else ""
        st.markdown(f"**{st.session_state.env_name}** — {st.session_state.task[:80]}{turn_label}")
    with top_r:
        if st.session_state.get("_stopping"):
            st.button("Stopping...", disabled=True, use_container_width=True)
        elif st.button("Stop", use_container_width=True):
            st.session_state._stopping = True
            cancel = st.session_state.get("_cancel_event")
            if cancel:
                cancel.set()
            # Unblock feedback queue if thread is waiting there
            fq = st.session_state.feedback_queue
            if fq:
                try:
                    fq.put_nowait(None)
                except Exception:
                    pass
            st.rerun()

    # Status line
    if st.session_state.get("_stopping"):
        st.caption("Stopping after current operation finishes...")
    elif st.session_state.current_status:
        st.caption(st.session_state.current_status)

    # ── Two-column layout ─────────────────────────────────────
    vid_col, info_col = st.columns([5, 5], gap="medium")

    with vid_col:
        if st.session_state.latest_frame is not None:
            att, ep, step = st.session_state.latest_meta
            best = st.session_state.latest_best
            annotated = draw_overlay(st.session_state.latest_frame, att, ep, step, best)
            st.image(annotated, channels="RGB", use_container_width=True)
        else:
            st.markdown(
                '<div style="width:100%;aspect-ratio:4/3;max-height:320px;'
                'background:rgba(0,0,0,0.3);border:1px solid rgba(255,255,255,0.06);'
                'border-radius:8px;display:flex;align-items:center;justify-content:center;'
                'color:rgba(255,255,255,0.15);font-size:0.9rem;">Waiting for simulation...</div>',
                unsafe_allow_html=True)

        # Video replay of latest attempt
        if st.session_state.latest_video and os.path.exists(st.session_state.latest_video):
            st.video(st.session_state.latest_video)

    with info_col:
        # Phase plan reference (compact)
        if st.session_state.phase_plan:
            flow = " -> ".join(p["name"] for p in st.session_state.phase_plan)
            st.caption(f"Phases: {flow}")

        # Agent process log
        st.markdown("**Agent Process**")
        for entry in st.session_state.tool_log:
            if isinstance(entry, dict):
                summary = entry.get("summary", "")
                tool = entry.get("tool", "")
                if "error" in summary.lower() or "Error" in summary:
                    bg = "#3b1c1c"
                elif "Success rate:" in summary:
                    bg = "#1c2b1c" if "100%" in summary else "#2a2a1c"
                elif tool in ("generate_policy", "submit_policy"):
                    bg = "#1c1c2b"
                else:
                    bg = "#0f1318"
                st.markdown(
                    f'<div style="background:{bg};border-radius:6px;padding:6px 12px;'
                    f'margin-bottom:4px;font-size:0.82rem;font-family:IBM Plex Mono,monospace">'
                    f'{summary}</div>',
                    unsafe_allow_html=True)

        # Diagnosis
        if st.session_state.diag_problem:
            st.markdown(f"**Problem:** {st.session_state.diag_problem}")
        if st.session_state.diag_suggestion:
            st.markdown(f"**Fix:** {st.session_state.diag_suggestion}")

    # ── Code (full width, below columns) ─────────────────────
    if st.session_state.current_code:
        show_code = st.checkbox("Show code", value=False, key="show_code_toggle")
        if show_code:
            st.code(st.session_state.current_code, language="python")

    # ── Error display ─────────────────────────────────────────
    if st.session_state.gen_error:
        st.error(st.session_state.gen_error[:500])

    # ── Feedback input ────────────────────────────────────────
    st.divider()
    if st.session_state.awaiting_feedback:
        st.markdown(
            '<div style="background:#1a2744;border:1px solid #3b82f6;border-radius:6px;'
            'padding:8px 14px;margin-bottom:8px;font-size:0.85rem">'
            'Agent is waiting for your feedback. Type below or wait to auto-continue.</div>',
            unsafe_allow_html=True)

    if "gen_fb_key" not in st.session_state:
        st.session_state.gen_fb_key = 0

    fb_col, send_col = st.columns([8, 1])
    with fb_col:
        if st.session_state.awaiting_feedback:
            placeholder = "Agent is waiting -- type your feedback here..."
        else:
            placeholder = "Agent is working... feedback will be delivered at the next pause"
        user_input = st.text_input(
            "feedback", label_visibility="collapsed",
            placeholder=placeholder,
            key=f"gen_fb_{st.session_state.gen_fb_key}",
        )
    with send_col:
        send = st.button("Send", type="primary", use_container_width=True, key="send_gen_fb")
    if send and user_input:
        fq = st.session_state.feedback_queue
        if fq is not None:
            fq.put_nowait(user_input)
            st.session_state.gen_fb_key += 1  # clear input on next render
            st.toast("Feedback sent!")

    # Auto-refresh while running
    time.sleep(0.12)
    st.rerun()


# ═════════════════════════════════════════
# STEP 4: RESULTS
# ═════════════════════════════════════════

elif st.session_state.ui_step == "results":

    # Clean up generation flags
    st.session_state._gen_started = False
    st.session_state._stopping = False

    result = st.session_state.result

    if result is None:
        # No result — either cancelled or errored
        if st.session_state.gen_error:
            st.error("Generation failed with an error.")
            with st.expander("Error details"):
                st.code(st.session_state.gen_error)
        else:
            st.warning("Generation was stopped before producing results.")

        # Still show any partial work (videos, code) that was collected
        if st.session_state.current_code:
            st.markdown("#### Last Generated Code")
            with st.expander("View code", expanded=True):
                st.code(st.session_state.current_code, language="python")
            st.download_button(
                "Download last code", data=st.session_state.current_code,
                file_name="policy_partial.py", mime="text/x-python",
            )

        if st.session_state.attempt_videos:
            st.markdown("#### Collected Videos")
            for label, vpath in sorted(st.session_state.attempt_videos.items()):
                if os.path.exists(vpath):
                    st.video(vpath)

        if st.button("Start Over"):
            for k, v in DEFAULTS.items():
                st.session_state[k] = v
            for k in list(st.session_state.keys()):
                if k.startswith("_"):
                    del st.session_state[k]
            st.rerun()
        st.stop()

    _success = getattr(result, "success", False)
    _rate = getattr(result, "success_rate", 0.0)
    _turns = getattr(result, "total_turns", 0)
    _code = getattr(result, "policy_code", "")

    if _success:
        st.success("Policy generation succeeded!")
    else:
        st.warning(f"Best success rate: {_rate:.0%} — target not reached")

    sa, sb, sc = st.columns(3)
    sa.metric("Best Success Rate", f"{_rate:.0%}")
    sb.metric("Turns", str(_turns))
    if getattr(result, "total_tokens", 0):
        sc.metric("Tokens Used", str(result.total_tokens))

    # Phase plan used
    plan = getattr(result, "phase_plan", []) or st.session_state.phase_plan
    if plan:
        phase_flow = " \u2192 ".join(f"`{p['name']}`" for p in plan)
        st.markdown(f"**Phase Plan:** {phase_flow}")

    # Video replays
    if st.session_state.attempt_videos:
        st.markdown("#### Simulation Replays")
        video_items = sorted(st.session_state.attempt_videos.items())
        if len(video_items) == 1:
            vlabel, vpath = video_items[0]
            if os.path.exists(vpath):
                st.video(vpath)
                with open(vpath, "rb") as vf:
                    st.download_button(
                        f"Download video ({vlabel})", data=vf.read(),
                        file_name=f"{vlabel}.mp4", mime="video/mp4",
                        key=f"dl_result_{vlabel}",
                    )
        else:
            vid_tabs = st.tabs([l.replace("_", " ").title() for l, _ in video_items])
            for tab, (vlabel, vpath) in zip(vid_tabs, video_items):
                with tab:
                    if os.path.exists(vpath):
                        st.video(vpath)
                        with open(vpath, "rb") as vf:
                            st.download_button(
                                "Download video", data=vf.read(),
                                file_name=f"{vlabel}.mp4", mime="video/mp4",
                                key=f"dl_result_{vlabel}",
                            )

    # Tool call history
    tool_history = getattr(result, "tool_history", [])
    if tool_history:
        with st.expander(f"Agent tool calls ({len(tool_history)})"):
            for entry in tool_history:
                icon = "\u2717" if entry.get("is_error") else "\u25b6"
                st.markdown(
                    f"{icon} **Turn {entry.get('turn', '?')}** — `{entry['tool']}` "
                    f"({entry.get('args_summary', '')})",
                )

    # Best policy code
    st.markdown("#### Best Policy")
    if _code:
        phases = extract_phases(_code)
        if phases:
            st.markdown("**Strategy:** " + " \u2192 ".join(f"`{p}`" for p in phases))
        with st.expander("View full code", expanded=True):
            st.code(_code, language="python")
        st.download_button(
            "Download policy file", data=_code,
            file_name="policy.py", mime="text/x-python",
            use_container_width=True,
        )

    st.divider()
    if st.button("Start Over", use_container_width=False):
        for k, v in DEFAULTS.items():
            st.session_state[k] = v
        # Clean up internal state
        for k in list(st.session_state.keys()):
            if k.startswith("_"):
                del st.session_state[k]
        st.rerun()

    st.stop()
