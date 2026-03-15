"""Interactive review mode for the agent loop."""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from enum import Enum

import click

from roboscribe.sim.diagnostics import Diagnosis
from roboscribe.sim.trajectory import TrajectoryResult


class ReviewAction(Enum):
    """Actions the user can take during interactive review."""

    CONTINUE = "c"
    FEEDBACK = "f"
    EDIT = "e"
    SAVE = "s"
    QUIT = "q"


@dataclass
class ReviewResult:
    """Result of an interactive review session."""

    action: ReviewAction
    human_feedback: str = ""
    edited_code: str = ""


@dataclass
class _Phase:
    """A detected state machine phase."""

    name: str
    line_number: int


class InteractiveReviewer:
    """Displays generated code, simulation results, and prompts the user for review."""

    def review(
        self,
        code: str,
        sim_result: TrajectoryResult,
        diagnosis: Diagnosis | None,
        attempt: int,
        max_attempts: int,
    ) -> ReviewResult:
        """Run a full interactive review cycle and return the user's decision."""
        self._display_header(attempt, max_attempts)
        self._display_code_with_phases(code)
        phases = self._extract_phases(code)
        self._display_phase_summary(phases)
        self._display_simulation_results(sim_result)
        if diagnosis:
            self._display_diagnosis(diagnosis)
        return self._prompt_menu(code)

    # ---- Display helpers ----

    def _display_header(self, attempt: int, max_attempts: int) -> None:
        click.echo()
        click.secho("=" * 60, fg="bright_white", bold=True)
        click.secho(
            f"  INTERACTIVE REVIEW  --  Attempt {attempt}/{max_attempts}",
            fg="bright_white",
            bold=True,
        )
        click.secho("=" * 60, fg="bright_white", bold=True)

    def _display_code_with_phases(self, code: str) -> None:
        click.echo()
        click.secho("Generated Policy Code:", bold=True)
        click.secho("-" * 40)

        lines = code.splitlines()
        for i, line in enumerate(lines, 1):
            num_prefix = click.style(f"{i:4d} | ", fg="bright_black")
            styled = self._style_code_line(line)
            click.echo(num_prefix + styled)

        click.secho("-" * 40)

    def _style_code_line(self, line: str) -> str:
        """Apply syntax highlighting to a single line of policy code."""
        stripped = line.lstrip()

        # State constant definitions (e.g. APPROACH = 0)
        if re.match(r"^[A-Z_]+\s*=\s*\d+", stripped):
            return click.style(line, fg="yellow")

        # Phase check lines: if state == PHASE or elif state == PHASE
        if re.match(r"^(?:el)?if\s+state\s*==\s*[A-Z_]+", stripped):
            return click.style(line, fg="cyan", bold=True)

        # State transitions: state = PHASE
        if re.match(r"^state\s*=\s*[A-Z_]+", stripped):
            return click.style(line, fg="green")

        # Function definitions
        if re.match(r"^def\s+", stripped):
            return click.style(line, bold=True)

        return line

    def _extract_phases(self, code: str) -> list[_Phase]:
        """Extract state machine phases from policy code."""
        phases: list[_Phase] = []
        seen: set[str] = set()
        for i, line in enumerate(code.splitlines(), 1):
            # Match constant definitions: APPROACH = 0
            m = re.match(r"^\s*([A-Z][A-Z_0-9]+)\s*=\s*\d+", line)
            if m:
                name = m.group(1)
                if name not in seen:
                    seen.add(name)
                    phases.append(_Phase(name=name, line_number=i))
                continue
            # Match phase checks: if state == APPROACH
            m = re.match(r"^\s*(?:el)?if\s+state\s*==\s*([A-Z][A-Z_0-9]+)", line)
            if m:
                name = m.group(1)
                if name not in seen:
                    seen.add(name)
                    phases.append(_Phase(name=name, line_number=i))
        return phases

    def _display_phase_summary(self, phases: list[_Phase]) -> None:
        if not phases:
            return
        click.echo()
        click.secho("State Machine Phases:", bold=True)
        for p in phases:
            click.echo(f"  {p.name:<16} (line {p.line_number})")

    def _display_simulation_results(self, result: TrajectoryResult) -> None:
        click.echo()
        click.secho("Simulation Results:", bold=True)
        rate_color = "green" if result.success_rate >= 0.8 else "yellow" if result.success_rate > 0 else "red"
        click.secho(
            f"  Success rate: {result.success_rate:.0%} "
            f"({result.successes}/{result.total_episodes})",
            fg=rate_color,
        )
        if result.episode_rewards:
            avg_r = sum(result.episode_rewards) / len(result.episode_rewards)
            max_r = max(result.episode_rewards)
            click.echo(f"  Avg reward: {avg_r:.2f}  |  Max reward: {max_r:.2f}")
        if result.error:
            click.secho(f"  Error: {result.error_type} — {result.error[:200]}", fg="red")

    def _display_diagnosis(self, diagnosis: Diagnosis) -> None:
        click.echo()
        click.secho("Diagnosis:", bold=True)
        click.echo(f"  Category: {diagnosis.category}")
        click.echo(f"  {diagnosis.summary}")
        if diagnosis.suggestions:
            click.secho(f"  Suggestions: {diagnosis.suggestions}", fg="yellow")

    # ---- Menu ----

    def _prompt_menu(self, code: str) -> ReviewResult:
        """Show the action menu and handle the user's choice."""
        click.echo()
        click.secho("What would you like to do?", bold=True)
        click.echo("  [c] Continue   (auto-revise with diagnosis)")
        click.echo("  [f] Feedback   (add your hints for the next revision)")
        click.echo("  [e] Edit       (open code in $EDITOR)")
        click.echo("  [s] Save       (save best result so far and stop)")
        click.echo("  [q] Quit")
        click.echo()

        while True:
            choice = click.prompt("Choice", default="c").strip().lower()
            if choice == "c":
                return ReviewResult(action=ReviewAction.CONTINUE)
            elif choice == "f":
                return self._handle_feedback()
            elif choice == "e":
                return self._handle_edit(code)
            elif choice == "s":
                return ReviewResult(action=ReviewAction.SAVE)
            elif choice == "q":
                return ReviewResult(action=ReviewAction.QUIT)
            else:
                click.secho("Invalid choice. Enter c, f, e, s, or q.", fg="red")

    def _handle_feedback(self) -> ReviewResult:
        """Prompt the user for free-text feedback."""
        click.echo("Enter your feedback (press Enter twice to finish):")
        lines: list[str] = []
        while True:
            line = input()
            if line == "" and lines and lines[-1] == "":
                break
            lines.append(line)
        feedback = "\n".join(lines).strip()
        if feedback:
            click.secho(f"Feedback recorded ({len(feedback)} chars).", fg="green")
        else:
            click.secho("No feedback entered — continuing with auto-revise.", fg="yellow")
        return ReviewResult(action=ReviewAction.FEEDBACK, human_feedback=feedback)

    def _handle_edit(self, code: str) -> ReviewResult:
        """Open the code in $EDITOR and return the edited version."""
        editor = os.environ.get("EDITOR", os.environ.get("VISUAL", ""))
        if not editor:
            click.secho(
                "No $EDITOR set. Set EDITOR environment variable to use this feature.",
                fg="red",
            )
            return self._prompt_menu(code)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix="_policy.py", delete=False,
        ) as f:
            f.write(code)
            tmp_path = f.name

        try:
            subprocess.run([editor, tmp_path], check=True)
            with open(tmp_path) as f:
                edited = f.read()
            if edited != code:
                click.secho("Code edited successfully.", fg="green")
            else:
                click.secho("No changes detected.", fg="yellow")
            return ReviewResult(action=ReviewAction.EDIT, edited_code=edited)
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            click.secho(f"Editor failed: {exc}", fg="red")
            return self._prompt_menu(code)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
