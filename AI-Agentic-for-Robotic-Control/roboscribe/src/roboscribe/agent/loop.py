"""Core agent loop: generate → simulate → diagnose → revise.

Human-in-the-loop: on_attempt() may inject a `_human_feedback` attribute
onto the AttemptRecord before returning. The loop reads it and forwards it
to build_revision_prompt() so the LLM can incorporate the user's observations.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import click

from roboscribe.config import Config
from roboscribe.exceptions import RoboScribeError
from roboscribe.llm.base import LLMResponse
from roboscribe.llm.factory import create_backend
from roboscribe.sim.env_registry import ENV_REGISTRY
from roboscribe.sim.runner import SimulationRunner
from roboscribe.sim.diagnostics import diagnose_failure, Diagnosis
from roboscribe.sim.trajectory import TrajectoryResult
from roboscribe.sim.introspect import introspect_env, format_obs_report
from roboscribe.agent.prompts import (
    SYSTEM_PROMPT,
    ENV_SELECTION_SYSTEM_PROMPT,
    build_generation_prompt,
    build_revision_prompt,
    build_env_selection_prompt,
)
from roboscribe.agent.interactive import InteractiveReviewer, ReviewAction
from roboscribe.output.writer import PolicyWriter


@dataclass
class AttemptRecord:
    """Record of a single generation attempt."""
    attempt: int
    code: str
    result: TrajectoryResult
    diagnosis: Diagnosis | None = None
    llm_response: LLMResponse | None = None
    video_path: str | None = None
    # Populated by on_attempt() hook when human feedback is collected via UI
    _human_feedback: str = ""


@dataclass
class GenerationResult:
    """Final result of the generation process."""
    success: bool
    policy_code: str
    success_rate: float
    attempts: int
    output_path: str = ""
    history: list[AttemptRecord] = field(default_factory=list)


class AgentLoop:
    """Self-debugging agent loop for policy generation."""

    def __init__(self, config: Config):
        self.config = config
        self.llm = create_backend(config)
        self.runner = SimulationRunner(config)
        self.writer = PolicyWriter(config)
        self.reviewer = InteractiveReviewer() if config.interactive else None

    def run(
        self,
        task_description: str,
        env_name: str,
        on_attempt=None,
        on_frame=None,
        on_status=None,
    ) -> GenerationResult:
        """Run the full generate → simulate → diagnose → revise loop."""

        def status(msg: str):
            click.secho(msg)
            if on_status:
                on_status(msg)

        if env_name not in ENV_REGISTRY:
            raise RoboScribeError(
                f"Unknown environment: {env_name}. "
                f"Available: {', '.join(sorted(ENV_REGISTRY.keys()))}"
            )

        env_info = ENV_REGISTRY[env_name]
        history: list[AttemptRecord] = []
        best_code = ""
        best_rate = 0.0
        # human_feedback from CLI interactive mode (kept for backward compat)
        cli_human_feedback = ""

        # ── Introspect environment (ground truth obs shapes) ──────────
        status("Inspecting environment observations...")
        introspection_report = introspect_env(env_name, robot=self.config.robot)
        introspection_str = format_obs_report(introspection_report)
        self._log_verbose(f"Introspection:\n{introspection_str}")

        for attempt in range(1, self.config.max_attempts + 1):
            self._log(f"\n--- Attempt {attempt}/{self.config.max_attempts} ---", bold=True)

            # ── Generate or revise ───────────────────────────────────────
            if attempt == 1:
                status(f"✍️  Writing script for attempt {attempt}/{self.config.max_attempts}…")
                code, llm_resp = self._generate(task_description, env_info, introspection_str)
            else:
                prev = history[-1]
                # Collect human feedback: UI hook takes priority over CLI interactive mode
                human_feedback = prev._human_feedback or cli_human_feedback
                cli_human_feedback = ""

                status(f"✍️  Revising script for attempt {attempt}/{self.config.max_attempts}…")
                if human_feedback:
                    status(f"💬  Incorporating human feedback into revision…")

                code, llm_resp = self._revise(
                    task_description, env_info,
                    prev.code, prev.diagnosis, prev.result.trajectory_summary,
                    human_feedback=human_feedback,
                )

            status(f"✅  Script ready ({len(code)} chars) — launching simulation…")
            self._log_verbose(f"Generated code ({len(code)} chars)")

            # ── Simulate ────────────────────────────────────────────────
            status(f"🤖  Simulating attempt {attempt} ({self.config.num_episodes} episodes)…")
            self._log("Running simulation...")

            sim_result = self.runner.run_policy(
                code, env_name, frame_callback=on_frame,
            )

            video_path = None
            if getattr(self.config, "record_attempt_videos", False):
                try:
                    from roboscribe.sim.recorder import record_policy
                    import tempfile, os, textwrap
                    tmp_policy_path = os.path.join(
                        tempfile.gettempdir(), f"roboscribe_attempt_{attempt}.py"
                    )
                    with open(tmp_policy_path, "w") as f:
                        f.write(textwrap.dedent(code).lstrip())
                    video_path = os.path.join(
                        tempfile.gettempdir(), f"roboscribe_attempt_{attempt}.mp4"
                    )
                    record_policy(tmp_policy_path, env_name, output_path=video_path, num_episodes=1)
                except Exception as e:
                    self._log_verbose(f"Video recording failed: {e}")
                    video_path = None

            # ── Track best ──────────────────────────────────────────────
            if sim_result.success_rate > best_rate:
                best_rate = sim_result.success_rate
                best_code = code

            if sim_result.error:
                self._log(f"Error: {sim_result.error_type} — {sim_result.error[:200]}", fg="red")
                status(f"❌  Attempt {attempt} error: {sim_result.error_type}")
            else:
                rate_str = (f"{sim_result.success_rate:.0%} "
                            f"({sim_result.successes}/{sim_result.total_episodes})")
                self._log(
                    f"Success rate: {rate_str}",
                    fg="green" if sim_result.success_rate >= self.config.success_threshold else "yellow",
                )
                status(f"📊  Attempt {attempt} done — success rate: {rate_str}")

            # ── Diagnose ────────────────────────────────────────────────
            diagnosis = None
            if sim_result.success_rate >= self.config.success_threshold:
                self._log("Target success rate reached!", fg="green", bold=True)
                status(f"🎉  Target reached! Attempt {attempt} succeeded.")
                record = AttemptRecord(attempt, code, sim_result, diagnosis, llm_resp)
                record.video_path = video_path
                history.append(record)
                if on_attempt:
                    on_attempt(record)
                break

            diagnosis = diagnose_failure(sim_result, env_name)
            self._log(f"Diagnosis: [{diagnosis.category}] {diagnosis.summary}", fg="yellow")
            self._log_verbose(f"Suggestions: {diagnosis.suggestions}")
            status(f"🔍  Diagnosis: {diagnosis.category} — {diagnosis.summary}")

            record = AttemptRecord(attempt, code, sim_result, diagnosis, llm_resp)
            record.video_path = video_path
            history.append(record)

            # ── Call on_attempt hook ─────────────────────────────────────
            # The UI hook may block here to collect human feedback,
            # and will set record._human_feedback before returning.
            if on_attempt:
                on_attempt(record)

            # ── CLI interactive review (only when not using UI hook) ──────
            if self.reviewer and on_attempt is None and attempt < self.config.max_attempts:
                review = self.reviewer.review(
                    code, sim_result, diagnosis, attempt, self.config.max_attempts,
                )
                if review.action == ReviewAction.QUIT:
                    self._log("Stopped by user.")
                    break
                elif review.action == ReviewAction.SAVE:
                    self._log("Saving best result and stopping.")
                    break
                elif review.action == ReviewAction.FEEDBACK:
                    cli_human_feedback = review.human_feedback
                elif review.action == ReviewAction.EDIT:
                    edited = review.edited_code
                    self._log("Re-running simulation with edited code...")
                    sim_result = self.runner.run_policy(edited, env_name)
                    if sim_result.success_rate > best_rate:
                        best_rate = sim_result.success_rate
                        best_code = edited
                    history[-1] = AttemptRecord(attempt, edited, sim_result, diagnosis, llm_resp)
                    if sim_result.success_rate >= self.config.success_threshold:
                        self._log("Target success rate reached with edited code!", fg="green", bold=True)
                        break

        # ── Write output ─────────────────────────────────────────────────
        output_path = ""
        if best_code:
            output_path = self.writer.write(
                best_code,
                env_name=env_name,
                task_description=task_description,
                success_rate=best_rate,
                attempts=len(history),
                robot=self.config.robot,
            )

        status(f"🏁  All done — best rate: {best_rate:.0%} over {len(history)} attempt(s).")

        return GenerationResult(
            success=best_rate >= self.config.success_threshold,
            policy_code=best_code,
            success_rate=best_rate,
            attempts=len(history),
            output_path=output_path,
            history=history,
        )

    def select_environment(self, task_description: str) -> str:
        """Ask the LLM to pick the best environment for a task description.

        Returns the environment name (a key in ENV_REGISTRY).
        """
        prompt = build_env_selection_prompt(task_description, ENV_REGISTRY)
        resp = self.llm.generate(ENV_SELECTION_SYSTEM_PROMPT, prompt)
        raw = resp.raw_text.strip().strip("`").strip()
        # Match against known environments (case-insensitive)
        for name in ENV_REGISTRY:
            if raw.lower() == name.lower():
                return name
        # Fallback: check if the response contains an environment name
        for name in ENV_REGISTRY:
            if name.lower() in raw.lower():
                return name
        raise RoboScribeError(
            f"LLM returned unknown environment '{raw}'. "
            f"Available: {', '.join(sorted(ENV_REGISTRY.keys()))}"
        )

    def _generate(self, task_description: str, env_info, introspection: str = "") -> tuple[str, LLMResponse]:
        self._log("Generating initial policy...")
        prompt = build_generation_prompt(task_description, env_info, introspection=introspection)
        resp = self.llm.generate(SYSTEM_PROMPT, prompt)
        self._log_verbose(f"Tokens used: {resp.tokens_used}")
        return resp.code, resp

    def _revise(
        self, task_description: str, env_info,
        prev_code: str, diagnosis: Diagnosis | None, trajectory: str,
        human_feedback: str = "",
    ) -> tuple[str, LLMResponse]:
        self._log("Revising policy based on diagnosis...")
        if diagnosis is None:
            diagnosis = Diagnosis("UNKNOWN", "Unknown failure", "", "Try a different approach")
        prompt = build_revision_prompt(
            task_description, env_info, prev_code, diagnosis, trajectory,
            human_feedback=human_feedback,
        )
        resp = self.llm.generate(SYSTEM_PROMPT, prompt)
        self._log_verbose(f"Tokens used: {resp.tokens_used}")
        return resp.code, resp

    def _log(self, msg: str, fg: str | None = None, bold: bool = False, nl: bool = True):
        click.secho(msg, fg=fg, bold=bold, nl=nl)

    def _log_verbose(self, msg: str):
        if self.config.verbose:
            click.secho(f"  {msg}", fg="bright_black")