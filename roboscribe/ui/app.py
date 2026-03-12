"""
RoboScribe Streamlit UI — redesigned layout
- Compact live view: video (fixed width) + status/log side by side
- Refined dark-terminal aesthetic with monospace log
- All logic preserved exactly
"""

import multiprocessing
multiprocessing.set_start_method("spawn", force=True)

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import cv2
import threading
import queue
import time
import traceback
import tempfile
import os

from roboscribe.config import Config
from roboscribe.agent.loop import AgentLoop
from roboscribe.sim.env_registry import ENV_REGISTRY

st.set_page_config(page_title="RoboScribe", layout="wide", initial_sidebar_state="collapsed")

# ── Global style injection ────────────────────────────────────
st.markdown("""
<style>
  /* tighter page padding */
  .block-container { padding-top: 1.5rem !important; padding-bottom: 1rem !important; max-width: 1100px; }
  /* hide streamlit chrome */
  #MainMenu, footer { visibility: hidden; }
  /* metric cards */
  [data-testid="metric-container"] {
    background: #0e1117;
    border: 1px solid #2a2a3a;
    border-radius: 8px;
    padding: 10px 14px;
  }
  /* tabs */
  [data-testid="stTabs"] button { font-size: 0.82rem; }
  /* info box */
  [data-testid="stAlert"] { padding: 8px 12px; }
  /* captions smaller */
  [data-testid="stImage"] p { font-size: 0.7rem; color: #555; margin-top: 2px; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────
st.markdown("""
<div style="display:flex; align-items:baseline; gap:10px; margin-bottom:0.5rem;">
  <span style="font-size:1.6rem; font-weight:700; letter-spacing:-0.5px; color:#fff;">RoboScribe</span>
  <span style="font-size:0.85rem; color:#555; font-family:monospace;">agentic robot policy generation</span>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────

DEFAULTS = {
    "running":        False,
    "done":           False,
    "result":         None,
    "attempt_history":[],
    "success_rates":  [],
    "frame_queue":    None,
    "attempt_queue":  None,
    "latest_frame":   None,
    "latest_meta":    (1, 0, 0),
    "latest_best":    None,
    "log":            [],
    "frame_count":    0,
    "current_status": "Waiting to start…",
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

if "log_queue" not in st.session_state:
    st.session_state.log_queue    = queue.Queue()
if "status_queue" not in st.session_state:
    st.session_state.status_queue = queue.Queue()


# ─────────────────────────────────────────
# THREAD-SAFE LOGGING
# ─────────────────────────────────────────

def add_log(msg: str, _lq=None):
    ts = time.strftime("%H:%M:%S")
    entry = f"<span style='color:#4a9eff;font-size:0.7rem;user-select:none'>{ts}</span><span style='color:#555'> › </span>{msg}"
    q = _lq if _lq is not None else st.session_state.log_queue
    q.put_nowait(entry)
    print(f"[LOG] {ts} {msg}")

def set_status(msg: str, _lq=None, _sq=None):
    sq = _sq if _sq is not None else st.session_state.status_queue
    sq.put_nowait(msg)
    add_log(msg, _lq=_lq)

def _drain_queues():
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
# FRAME OVERLAY
# ─────────────────────────────────────────

def draw_overlay(frame, attempt, episode, step, best_rate=None):
    frame = np.flipud(frame).copy()
    h, w  = frame.shape[:2]
    ov = frame.copy()
    cv2.rectangle(ov, (0, 0), (w, 38), (8, 8, 12), -1)
    cv2.addWeighted(ov, 0.7, frame, 0.3, 0, frame)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, f"T{attempt}  EP{episode+1}  S{step}",
                (8, 25), font, 0.52, (180, 180, 200), 1, cv2.LINE_AA)
    if best_rate is not None:
        rtxt = f"{best_rate:.0%}"
        tw = cv2.getTextSize(rtxt, font, 0.52, 1)[0][0]
        cv2.putText(frame, rtxt, (w - tw - 8, 25), font, 0.52, (80, 220, 120), 1, cv2.LINE_AA)
    return frame


# ─────────────────────────────────────────
# UI COMPONENTS
# ─────────────────────────────────────────

def render_status_banner(msg: str):
    if any(x in msg for x in ["Revis", "Writing", "script", "✍"]):
        color, dot = "#1a3a5c", "#4a9eff"
    elif any(x in msg for x in ["Simul", "simulat", "🤖"]):
        color, dot = "#0d2d1a", "#22c55e"
    elif any(x in msg for x in ["error", "Error", "❌"]):
        color, dot = "#2d0d0d", "#ef4444"
    elif any(x in msg for x in ["done", "Done", "success", "🎉", "🏁", "reached"]):
        color, dot = "#0d2d1a", "#22c55e"
    elif any(x in msg for x in ["Diagno", "📊"]):
        color, dot = "#1a1a0d", "#eab308"
    else:
        color, dot = "#12121a", "#555"

    components.html(f"""
    <div style="background:{color}; border-left:3px solid {dot};
                border-radius:0 6px 6px 0; padding:9px 14px;
                font-family:'SF Mono',monospace; font-size:0.82rem;
                color:#ddd; letter-spacing:0.01em; white-space:nowrap;
                overflow:hidden; text-overflow:ellipsis;">
      <span style="color:{dot}; margin-right:8px;">●</span>{msg}
    </div>""", height=44)


def render_log_panel(entries: list):
    html = "<br>".join(reversed(entries)) if entries else \
           "<span style='color:#333'>— no entries yet —</span>"
    components.html(f"""
    <div style="height:195px; overflow-y:auto; overflow-x:hidden;
                background:#080810; border:1px solid #1a1a2e;
                border-radius:6px; padding:10px 12px;
                font-family:'SF Mono','Fira Code',monospace;
                font-size:0.72rem; line-height:1.7; color:#aaa;">
      {html}
    </div>""", height=211, scrolling=False)


# ─────────────────────────────────────────
# USER INPUT
# ─────────────────────────────────────────

if not st.session_state.running and not st.session_state.done:
    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
    task = st.text_area("Task description", height=90,
                        placeholder="e.g. lift the cube, stack cube A on cube B…")
    c1, c2, c3, c4 = st.columns([3, 1.2, 1.2, 1.5])
    with c1: env     = st.selectbox("Environment", list(ENV_REGISTRY.keys()), label_visibility="collapsed")
    with c2: robot   = st.selectbox("Robot", ["Panda"], label_visibility="collapsed")
    with c3: backend = st.selectbox("Backend", ["qwen","openai","anthropic","deepseek"], label_visibility="collapsed")
    with c4: run_button = st.button("⚡  Generate Policy", type="primary", use_container_width=True)
    # small labels under each
    st.markdown(
        "<div style='display:flex;gap:0;font-size:0.68rem;color:#555;margin-top:-10px'>"
        "<div style='flex:3'>environment</div>"
        "<div style='flex:1.2'>robot</div>"
        "<div style='flex:1.2'>backend</div>"
        "<div style='flex:1.5'></div></div>",
        unsafe_allow_html=True)
else:
    task = env = robot = backend = None
    run_button = False


# ─────────────────────────────────────────
# LAUNCH
# ─────────────────────────────────────────

if run_button:
    for k, v in DEFAULTS.items():
        st.session_state[k] = v
    st.session_state.running       = True
    st.session_state.frame_queue   = queue.Queue(maxsize=8)
    st.session_state.attempt_queue = queue.Queue()

    add_log("Starting agent…")

    try:
        cfg = Config.from_env(robot=robot, num_episodes=5, verbose=True)
        cfg.llm_backend = backend
        cfg.validate()
        add_log(f"backend=<b style='color:#4a9eff'>{cfg.llm_backend}</b> model=<b style='color:#4a9eff'>{cfg.llm_model}</b>")
    except Exception as e:
        add_log(f"Config error: {e}")
        st.session_state.running = False
        st.error(f"Config error: {e}"); st.stop()

    loop = AgentLoop(cfg)
    add_log("AgentLoop ready")

    _task          = task
    _env           = env
    _robot         = robot
    _frame_queue   = st.session_state.frame_queue
    _attempt_queue = st.session_state.attempt_queue
    _log_q         = st.session_state.log_queue
    _status_q      = st.session_state.status_queue
    _cb            = {"attempt":1, "best_rate":None, "last_push":0.0, "frames_sent":0}
    _result_holder = {"result":None, "done":False}

    def _tlog(msg):    add_log(msg, _lq=_log_q)
    def _tstatus(msg): set_status(msg, _lq=_log_q, _sq=_status_q)

    def on_frame(frame, episode, step):
        now = time.time()
        if now - _cb["last_push"] < 0.05: return
        _cb["last_push"] = now
        _cb["frames_sent"] += 1
        item = (frame.copy(), _cb["attempt"], episode, step, _cb["best_rate"])
        if _frame_queue.full():
            try: _frame_queue.get_nowait()
            except queue.Empty: pass
        try: _frame_queue.put_nowait(item)
        except Exception: pass

    def on_attempt(record):
        rate = record.result.success_rate
        if _cb["best_rate"] is None or rate > _cb["best_rate"]:
            _cb["best_rate"] = rate
        _cb["attempt"] = record.attempt + 1
        try: _attempt_queue.put_nowait(record)
        except Exception as e: print(f"[ATTEMPT] queue failed: {e}")

    def run_agent():
        print(f"[THREAD] start task={_task!r} env={_env!r}")
        try:
            result = loop.run(_task, _env, on_attempt=on_attempt,
                              on_frame=on_frame, on_status=_tstatus)
            print(f"[THREAD] done rate={result.success_rate:.0%}")
        except Exception as e:
            print(f"[THREAD] ❌ {e}"); traceback.print_exc(); result = None
        _result_holder["result"] = result
        _result_holder["done"]   = True
        try: _attempt_queue.put_nowait("__DONE__")
        except Exception: pass

    t = threading.Thread(target=run_agent, daemon=True, name="run_agent")
    t.start()
    add_log(f"Thread started (id={t.ident})")
    st.session_state._result_holder = _result_holder
    st.rerun()


# ─────────────────────────────────────────
# RUNNING VIEW
# ─────────────────────────────────────────

if st.session_state.running or (st.session_state.done and st.session_state.latest_frame is not None):

    _drain_queues()

    aq = st.session_state.attempt_queue
    if aq is not None:
        while not aq.empty():
            item = aq.get_nowait()
            if item == "__DONE__":
                rh = st.session_state.get("_result_holder")
                if rh and rh["done"]:
                    st.session_state.result  = rh["result"]
                    st.session_state.done    = True
                    st.session_state.running = False
                    add_log("Agent finished")
                    _drain_queues()
            else:
                st.session_state.attempt_history.append(item)
                st.session_state.success_rates.append(item.result.success_rate)
                add_log(f"Attempt {item.attempt} — {item.result.success_rate:.0%} success")

    fq = st.session_state.frame_queue
    new_item = None
    if fq is not None:
        while not fq.empty():
            try:
                new_item = fq.get_nowait()
                st.session_state.frame_count += 1
            except queue.Empty: break
    if new_item is not None:
        raw, att, ep, step, best = new_item
        st.session_state.latest_frame = raw
        st.session_state.latest_meta  = (att, ep, step)
        st.session_state.latest_best  = best

    # ── Status banner (full width, compact) ──────────────────
    render_status_banner(st.session_state.current_status)
    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    # ── Main live row: video (fixed 340px) | right panel ─────
    vid_col, right_col = st.columns([4, 5], gap="medium")

    with vid_col:
        if st.session_state.latest_frame is not None:
            att, ep, step = st.session_state.latest_meta
            best          = st.session_state.latest_best
            annotated = draw_overlay(st.session_state.latest_frame, att, ep, step, best)
            st.image(annotated, channels="RGB", use_container_width=True)
            st.markdown(
                f"<div style='font-family:monospace;font-size:0.68rem;color:#444;margin-top:2px'>"
                f"attempt {att} · ep {ep+1} · step {step} · {st.session_state.frame_count} frames"
                f"</div>", unsafe_allow_html=True)
        else:
            # placeholder with same aspect ratio as 256x256 frame
            components.html("""
            <div style="width:100%;aspect-ratio:1/1;max-height:300px;
                        background:#080810;border:1px solid #1a1a2e;
                        border-radius:8px;display:flex;align-items:center;
                        justify-content:center;color:#333;font-family:monospace;font-size:0.8rem;">
              waiting for sim…
            </div>""", height=272)

    with right_col:
        # # ── Success chart (compact) ───────────────────────────
        # if st.session_state.success_rates:
        #     df = pd.DataFrame({
        #         "Attempt":      range(1, len(st.session_state.success_rates)+1),
        #         "Success Rate": st.session_state.success_rates,
        #     })
        #     st.markdown("<div style='font-size:0.72rem;color:#555;margin-bottom:-14px;font-family:monospace'>SUCCESS RATE</div>",
        #                 unsafe_allow_html=True)
        #     st.line_chart(df.set_index("Attempt"), height=100, use_container_width=True)
        # else:
        #     st.markdown("<div style='font-size:0.72rem;color:#333;font-family:monospace;padding:8px 0'>SUCCESS RATE — no attempts yet</div>",
        #                 unsafe_allow_html=True)

        # ── Log panel ─────────────────────────────────────────
        st.markdown("<div style='font-size:0.72rem;color:#555;margin-bottom:4px;font-family:monospace;margin-top:8px'>STATUS LOG</div>",
                    unsafe_allow_html=True)
        render_log_panel(st.session_state.log)

    if st.session_state.running:
        time.sleep(0.1)
        st.rerun()


# ─────────────────────────────────────────
# FINAL RESULTS
# ─────────────────────────────────────────

if st.session_state.done and st.session_state.result is not None:

    result   = st.session_state.result
    attempts = st.session_state.attempt_history

    st.divider()

    # ── Summary row ───────────────────────────────────────────
    if result.success:
        st.success("Policy generation succeeded!")
    else:
        st.warning(f"Best success rate: {result.success_rate:.0%} — target not reached")

    sa, sb, sc = st.columns(3)
    sa.metric("Best Success Rate", f"{result.success_rate:.0%}")
    sb.metric("Total Attempts",    str(result.attempts))
    if result.output_path:
        sc.metric("Saved to", os.path.basename(result.output_path))

    # ── Final chart ───────────────────────────────────────────
    if st.session_state.success_rates:
        df = pd.DataFrame({
            "Attempt":      range(1, len(st.session_state.success_rates)+1),
            "Success Rate": st.session_state.success_rates,
        })
        st.line_chart(df.set_index("Attempt"), height=140, use_container_width=True)

    # ── Per-attempt tabs ──────────────────────────────────────
    st.markdown("#### Attempt Breakdown")
    if attempts:
        best_idx = max(range(len(attempts)), key=lambda i: attempts[i].result.success_rate)
        tabs = st.tabs([
            f"{'⭐ ' if i==best_idx else ''}Attempt {r.attempt}"
            for i, r in enumerate(attempts)
        ])
        for tab, record, i in zip(tabs, attempts, range(len(attempts))):
            with tab:
                rate = record.result.success_rate
                m1, m2, m3 = st.columns(3)
                m1.metric("Success Rate", f"{rate:.0%}")
                m2.metric("Episodes", f"{record.result.successes}/{record.result.total_episodes}")
                if record.result.episode_rewards:
                    m3.metric("Avg Reward",
                              f"{sum(record.result.episode_rewards)/len(record.result.episode_rewards):.2f}")
                if i == best_idx:
                    st.success("Best attempt — this policy was saved.")
                if record.diagnosis:
                    with st.expander("Diagnosis", expanded=(i==best_idx)):
                        st.markdown(f"**Category:** `{record.diagnosis.category}`")
                        st.markdown(f"**Summary:** {record.diagnosis.summary}")
                        if record.diagnosis.suggestions:
                            st.markdown(f"**Suggestions:** {record.diagnosis.suggestions}")
                if record.result.trajectory_summary:
                    with st.expander("Trajectory Log"):
                        st.text(record.result.trajectory_summary)
                with st.expander("Policy Code", expanded=(i==best_idx)):
                    st.code(record.code, language="python")

    # ── Best policy ───────────────────────────────────────────
    st.markdown("#### Best Policy")
    with st.expander("View full code", expanded=True):
        st.code(result.policy_code, language="python")

    st.divider()
    if st.button("↩  Clear and Run Again", use_container_width=False):
        for k, v in DEFAULTS.items():
            st.session_state[k] = v
        st.rerun()