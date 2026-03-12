"""
RoboScribe Live Video Diagnostic
=================================
Run this from your repo root:
    python diagnose_video.py

It tests each layer independently and tells you exactly where it breaks.
"""

import sys
import time
import queue
import threading
import traceback

PASS = "✅ PASS"
FAIL = "❌ FAIL"
SKIP = "⚠️  SKIP"

results = []

def check(name, fn):
    try:
        msg = fn()
        tag = PASS
        results.append((tag, name, msg or ""))
        print(f"{tag}  {name}" + (f" — {msg}" if msg else ""))
        return True
    except Exception as e:
        tag = FAIL
        results.append((tag, name, str(e)))
        print(f"{tag}  {name}")
        print(f"       {e}")
        traceback.print_exc()
        return False


print("\n" + "="*60)
print(" RoboScribe Live Video Diagnostic")
print("="*60 + "\n")


# ── 1. Imports ───────────────────────────────────────────────

def test_numpy():
    import numpy as np
    a = np.zeros((64, 64, 3), dtype=np.uint8)
    return f"shape={a.shape}"

def test_cv2():
    import cv2
    import numpy as np
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    cv2.putText(frame, "test", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    return f"version={cv2.__version__}"

def test_robosuite():
    import robosuite as suite
    return f"version={suite.__version__}"

def test_roboscribe_runner():
    from roboscribe.sim.runner import SimulationRunner
    return "import ok"

check("numpy import", test_numpy)
check("opencv import", test_cv2)
check("robosuite import", test_robosuite)
check("SimulationRunner import", test_roboscribe_runner)

print()


# ── 2. Does robosuite render produce frames? ─────────────────

def test_render_one_frame():
    import robosuite as suite
    import numpy as np

    controller_config = suite.load_composite_controller_config(controller="BASIC")
    env = suite.make(
        "Lift",
        robots="Panda",
        controller_configs=controller_config,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="agentview",
        use_camera_obs=False,
        camera_names=["agentview"],
        camera_heights=128,
        camera_widths=128,
        reward_shaping=True,
        horizon=5,
    )
    obs = env.reset()
    action = np.zeros(7)
    obs, r, done, info = env.step(action)

    # Try method 1: env.sim.render
    frame = None
    method = None
    try:
        frame = env.sim.render(camera_name="agentview", width=128, height=128)
        frame = frame[:, :, ::-1]
        method = "env.sim.render()"
    except Exception as e:
        print(f"       env.sim.render() failed: {e}")

    # Try method 2: obs-based camera
    if frame is None:
        try:
            frame = obs.get("agentview_image")
            if frame is not None:
                method = "obs['agentview_image']"
        except Exception as e:
            print(f"       obs camera failed: {e}")

    # Try method 3: use_camera_obs=True
    env.close()

    if frame is None:
        raise RuntimeError("All render methods failed — no frame produced")

    if not isinstance(frame, __import__("numpy").ndarray):
        raise RuntimeError(f"Frame is not ndarray, got: {type(frame)}")

    h, w = frame.shape[:2]
    return f"method={method}, shape={frame.shape}, dtype={frame.dtype}, min={frame.min()}, max={frame.max()}"

print("── Testing robosuite rendering ──")
render_ok = check("robosuite render produces frames", test_render_one_frame)
print()


# ── 3. Does frame_callback actually get called? ──────────────

def test_callback_called():
    from roboscribe.config import Config
    from roboscribe.sim.runner import SimulationRunner

    cfg = Config.from_env()
    cfg.robot = "Panda"
    cfg.num_episodes = 1
    cfg.max_episode_steps = 10

    runner = SimulationRunner(cfg)

    call_log = []

    def on_frame(frame, episode, step):
        call_log.append((episode, step, type(frame).__name__,
                         getattr(frame, 'shape', None)))

    import numpy as np

    # Write a trivial policy to a temp file
    import tempfile, os
    policy_code = """
import numpy as np
def get_action(obs):
    return np.zeros(7)
def reset():
    pass
"""
    result = runner.run_policy(policy_code, "Lift", frame_callback=on_frame)

    if not call_log:
        raise RuntimeError(
            "frame_callback was NEVER called!\n"
            "  This means runner.run_policy() is not calling on_frame().\n"
            "  Check that runner.py passes frame_callback to the simulation loop."
        )

    first = call_log[0]
    return (f"callback called {len(call_log)} times. "
            f"First call: episode={first[0]}, step={first[1]}, "
            f"frame_type={first[2]}, shape={first[3]}")

print("── Testing frame_callback pipeline ──")
cb_ok = check("frame_callback is called during run_policy", test_callback_called)
print()


# ── 4. Queue round-trip test ─────────────────────────────────

def test_queue_roundtrip():
    import numpy as np
    q = queue.Queue(maxsize=2)

    frames_sent = []
    frames_recv = []

    def producer():
        for i in range(5):
            frame = np.full((64, 64, 3), i * 40, dtype=np.uint8)
            if q.full():
                try: q.get_nowait()
                except queue.Empty: pass
            q.put_nowait(frame)
            frames_sent.append(i)
            time.sleep(0.02)

    t = threading.Thread(target=producer, daemon=True)
    t.start()
    t.join(timeout=2)

    while not q.empty():
        frames_recv.append(q.get_nowait())

    return f"sent={len(frames_sent)}, received={len(frames_recv)} (queue drops oldest when full — expected)"

check("queue round-trip (background thread → main thread)", test_queue_roundtrip)
print()


# ── 5. Full pipeline end-to-end (no Streamlit) ──────────────

def test_full_pipeline():
    import numpy as np
    from roboscribe.config import Config
    from roboscribe.sim.runner import SimulationRunner

    cfg = Config.from_env()
    cfg.robot = "Panda"
    cfg.num_episodes = 1
    cfg.max_episode_steps = 15

    runner = SimulationRunner(cfg)

    q = queue.Queue(maxsize=4)
    errors = []

    policy_code = """
import numpy as np
def get_action(obs):
    return np.zeros(7)
def reset():
    pass
"""

    def on_frame(frame, episode, step):
        if not isinstance(frame, np.ndarray):
            errors.append(f"step {step}: frame is {type(frame)}, not ndarray")
            return
        if q.full():
            try: q.get_nowait()
            except queue.Empty: pass
        q.put_nowait(frame.copy())

    result_box = {}

    def run():
        result_box["r"] = runner.run_policy(policy_code, "Lift", frame_callback=on_frame)

    t = threading.Thread(target=run, daemon=True)
    t.start()

    # simulate what Streamlit main loop would do
    received = []
    deadline = time.time() + 15
    while t.is_alive() and time.time() < deadline:
        try:
            frame = q.get(timeout=0.1)
            received.append(frame)
        except queue.Empty:
            pass
    t.join(timeout=5)

    if errors:
        raise RuntimeError(f"Frame errors: {errors[:3]}")
    if not received:
        raise RuntimeError(
            "No frames reached the consumer loop!\n"
            "  Either callback never fires, or queue is broken."
        )

    shapes = set(f.shape for f in received)
    return f"frames received by consumer={len(received)}, shapes={shapes}"

print("── Full end-to-end pipeline (no Streamlit) ──")
e2e_ok = check("full pipeline: sim → callback → queue → consumer", test_full_pipeline)
print()


# ── Summary ──────────────────────────────────────────────────

print("="*60)
print(" Summary")
print("="*60)
for tag, name, msg in results:
    print(f"  {tag}  {name}")
    if tag == FAIL and msg:
        print(f"          → {msg}")

print()
if all(r[0] == PASS for r in results):
    print("All checks passed! The issue is likely in Streamlit's")
    print("threading model. The polling loop in app.py should work.")
    print("Make sure you replaced app.py and restarted streamlit.")
else:
    fails = [r for r in results if r[0] == FAIL]
    print(f"{len(fails)} check(s) failed. Fix those first.")
print()