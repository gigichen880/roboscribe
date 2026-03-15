"""RoboScribe CLI."""

from __future__ import annotations

import sys

import click

from roboscribe import __version__


@click.group()
@click.version_option(__version__, prog_name="roboscribe")
def cli():
    """RoboScribe: Generate robosuite scripted policies from natural language."""


@cli.command()
@click.argument("task_description")
@click.option("--env", "-e", default=None, help="Robosuite environment name (auto-detected from task if omitted)")
@click.option("--robot", "-r", default="Panda", help="Robot to use (default: Panda)")
@click.option("--backend", "-b", default=None, help="LLM backend: openai, anthropic, qwen, qwen-beijing, qwen-singapore, or deepseek")
@click.option("--model", "-m", default=None, help="LLM model name")
@click.option("--api-key", default=None, help="API key (or set ROBOSCRIBE_API_KEY)")
@click.option("--base-url", default=None, help="Custom API base URL for OpenAI-compatible providers")
@click.option("--max-turns", default=15, help="Max agent iteration turns")
@click.option("--episodes", default=10, help="Episodes per evaluation")
@click.option("--output", "-o", default=".", help="Output directory")
@click.option("--classic", is_flag=True, help="Use classic generate→diagnose→revise loop instead of phase-design agent")
@click.option("--interactive", "-i", is_flag=True, help="Interactive review (classic mode only)")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def generate(task_description, env, robot, backend, model, api_key, base_url,
             max_turns, episodes, output, classic, interactive, verbose):
    """Generate a scripted policy from a natural language description."""
    from roboscribe.config import Config

    cfg = Config.from_env(
        llm_backend=backend,
        llm_model=model,
        api_key=api_key,
        base_url=base_url,
        robot=robot,
        max_attempts=max_turns,
        num_episodes=episodes,
        output_dir=output,
        interactive=interactive,
        verbose=verbose,
    )

    try:
        cfg.validate()
    except Exception as e:
        click.secho(f"Config error: {e}", fg="red", err=True)
        sys.exit(1)

    if classic:
        _generate_classic(cfg, task_description, env, robot, episodes, verbose)
    else:
        _generate_agent(cfg, task_description, env, robot, max_turns, episodes, verbose)


def _generate_classic(cfg, task_description, env, robot, episodes, verbose):
    """Classic generate→diagnose→revise loop."""
    from roboscribe.agent.loop import AgentLoop

    if env is None:
        click.secho("Classic mode requires --env. Use without --classic for auto-detection.", fg="red", err=True)
        sys.exit(1)

    loop = AgentLoop(cfg)
    click.secho(f"Generating policy for: {task_description}", fg="cyan")
    click.secho(f"Environment: {env} | Robot: {robot} | Backend: {cfg.llm_backend}/{cfg.llm_model}", fg="cyan")
    click.secho(f"Mode: classic loop", fg="bright_black")
    click.echo()

    try:
        result = loop.run(task_description, env)
    except Exception as e:
        click.secho(f"Error: {e}", fg="red", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

    if result.success:
        click.secho(f"\nSuccess! Policy saved to: {result.output_path}", fg="green", bold=True)
        click.secho(f"Success rate: {result.success_rate:.0%} over {episodes} episodes", fg="green")
        click.secho(f"Attempts: {result.attempts}", fg="green")
    else:
        click.secho(f"\nFailed to generate a working policy after {result.attempts} attempts.", fg="red")
        click.secho(f"Best success rate: {result.success_rate:.0%}", fg="yellow")
        if result.output_path:
            click.secho(f"Best attempt saved to: {result.output_path}", fg="yellow")
        sys.exit(1)


def _generate_agent(cfg, task_description, env, robot, max_turns, episodes, verbose):
    """Phase-design-first agent with tool-use iteration."""
    from roboscribe.agent.tool_loop import ToolAgentLoop

    loop = ToolAgentLoop(cfg)

    click.secho(f"Generating policy for: {task_description}", fg="cyan")
    click.secho(f"Robot: {robot} | Backend: {cfg.llm_backend}/{cfg.llm_model}", fg="cyan")
    click.secho(f"Mode: phase-design agent (max {max_turns} turns)", fg="bright_black")
    click.echo()

    try:
        # Step 1: Phase design (with auto env detection)
        click.secho("Step 1: Designing phase plan...", fg="cyan", bold=True)
        design = loop.run_phase_design(
            task_description, env,
            on_status=lambda msg: click.secho(f"  {msg}", fg="bright_black"),
        )

        click.secho(f"\nEnvironment: {design.env_name}", fg="green")
        click.secho(f"Phase plan ({len(design.phase_plan)} phases):", fg="green")
        for i, p in enumerate(design.phase_plan):
            click.secho(f"  {i+1}. {p['name']}", fg="green", bold=True, nl=False)
            click.echo(f" — {p['goal']}")
            if p.get("exit_condition"):
                click.echo(f"     Exit: {p['exit_condition']}")
        click.echo()

        # Interactive phase review
        if sys.stdin.isatty():
            while True:
                choice = click.prompt(
                    "Approve this plan? [y]es / [r]egenerate with feedback / [q]uit",
                    type=click.Choice(["y", "r", "q"], case_sensitive=False),
                    default="y",
                )
                if choice == "q":
                    click.secho("Aborted.", fg="yellow")
                    sys.exit(0)
                elif choice == "y":
                    break
                elif choice == "r":
                    feedback = click.prompt("Your feedback")
                    design = loop.redesign_phases(
                        task_description, design.env_name,
                        design.introspection_str, feedback,
                        previous_plan=design.phase_plan,
                        on_status=lambda msg: click.secho(f"  {msg}", fg="bright_black"),
                    )
                    click.echo()
                    click.secho(f"Revised plan ({len(design.phase_plan)} phases):", fg="green")
                    for i, p in enumerate(design.phase_plan):
                        click.secho(f"  {i+1}. {p['name']}", fg="green", bold=True, nl=False)
                        click.echo(f" — {p['goal']}")
                        if p.get("exit_condition"):
                            click.echo(f"     Exit: {p['exit_condition']}")
                    click.echo()

        # Step 2: Generate + iterate
        click.secho("Step 2: Generating and iterating...", fg="cyan", bold=True)
        click.echo()

        def on_tool_call(name, args, result_text):
            click.secho(f"  [{name}] ", fg="blue", bold=True, nl=False)
            # Show first line of result
            first_line = result_text.split("\n")[0][:120]
            click.echo(first_line)

        result = loop.run_with_phases(
            task_description, design.env_name,
            design.phase_plan, design.introspection_str,
            max_turns=max_turns,
            on_tool_call=on_tool_call,
            on_status=lambda msg: click.secho(f"  {msg}", fg="bright_black"),
        )

    except KeyboardInterrupt:
        click.secho("\nInterrupted.", fg="yellow")
        sys.exit(1)
    except Exception as e:
        click.secho(f"Error: {e}", fg="red", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

    click.echo()
    if result.success:
        click.secho(f"Success! Rate: {result.success_rate:.0%}", fg="green", bold=True)
        click.secho(f"Turns: {result.total_turns} | Tokens: {result.total_tokens}", fg="green")
    else:
        click.secho(f"Best rate: {result.success_rate:.0%} after {result.total_turns} turns", fg="yellow")
        if result.policy_code:
            click.secho("Best attempt saved.", fg="yellow")
        sys.exit(1)


@cli.command()
@click.argument("policy_file")
@click.option("--env", "-e", default=None, help="Override environment (auto-detected from file)")
@click.option("--robot", "-r", default="Panda", help="Robot to use")
@click.option("--episodes", "-n", default=20, help="Number of test episodes")
@click.option("--render", is_flag=True, help="Render simulation visually")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def test(policy_file, env, robot, episodes, render, verbose):
    """Test a policy file and report success rate."""
    import os
    from roboscribe.config import Config
    from roboscribe.sim.runner import SimulationRunner

    if not os.path.isfile(policy_file):
        click.secho(f"File not found: {policy_file}", fg="red", err=True)
        sys.exit(1)

    with open(policy_file) as f:
        code = f.read()

    # Try to detect env from file metadata
    if env is None:
        for line in code.split("\n"):
            if "environment:" in line.lower():
                env = line.split(":")[-1].strip()
                break
        if env is None:
            click.secho("Could not detect environment. Use --env to specify.", fg="red", err=True)
            sys.exit(1)

    cfg = Config.from_env(robot=robot, num_episodes=episodes, verbose=verbose)
    runner = SimulationRunner(cfg)

    click.secho(f"Testing {policy_file} on {env} ({episodes} episodes)...", fg="cyan")

    try:
        result = runner.run_policy(code, env, render=render)
    except Exception as e:
        click.secho(f"Error: {e}", fg="red", err=True)
        sys.exit(1)

    rate = result.success_rate
    color = "green" if rate >= 0.8 else "yellow" if rate >= 0.5 else "red"
    click.secho(f"\nSuccess rate: {rate:.0%} ({result.successes}/{result.total_episodes})", fg=color, bold=True)

    if verbose and result.error:
        click.secho(f"Errors: {result.error}", fg="red")


@cli.command()
def backends():
    """Show supported LLM backends, how to configure them, and current status."""
    import os
    from roboscribe.config import Config

    click.secho("LLM Backends:\n", fg="cyan", bold=True)

    for name, (default_model, base_url, env_var) in sorted(Config.PROVIDERS.items()):
        key_set = bool(os.environ.get(env_var, ""))
        status = click.style("ready", fg="green") if key_set else click.style("no key", fg="red")

        click.secho(f"  {name:<12}", fg="green", bold=True, nl=False)
        click.echo(f" [{status}]")
        click.echo(f"  {'':12}  Model:   {default_model}")
        if base_url:
            click.echo(f"  {'':12}  URL:     {base_url}")
        click.echo(f"  {'':12}  Key env: {env_var}", nl=False)
        if key_set:
            # show first 8 chars masked
            val = os.environ[env_var]
            click.echo(f"  ({val[:4]}...{val[-4:]})" if len(val) > 8 else "  (set)")
        else:
            click.echo(f"  (not set)")
        click.echo()

    # Show current effective config
    click.secho("Current config (from env):", fg="cyan", bold=True)
    backend = os.environ.get("ROBOSCRIBE_LLM_BACKEND", "openai")
    model = os.environ.get("ROBOSCRIBE_LLM_MODEL", "")
    base = os.environ.get("ROBOSCRIBE_BASE_URL", "")
    click.echo(f"  ROBOSCRIBE_LLM_BACKEND = {backend or '(not set, defaults to openai)'}")
    click.echo(f"  ROBOSCRIBE_LLM_MODEL   = {model or '(not set, uses provider default)'}")
    click.echo(f"  ROBOSCRIBE_BASE_URL    = {base or '(not set, uses provider default)'}")
    click.echo()

    click.secho("Quick start:", fg="cyan", bold=True)
    click.echo("  # OpenAI (default)")
    click.echo("  export OPENAI_API_KEY=sk-...")
    click.echo("  roboscribe generate 'lift the cube' -e Lift")
    click.echo()
    click.echo("  # Qwen")
    click.echo("  export DASHSCOPE_API_KEY=sk-...")
    click.echo("  roboscribe generate 'lift the cube' -e Lift --backend qwen")
    click.echo()
    click.echo("  # Anthropic")
    click.echo("  export ANTHROPIC_API_KEY=sk-...")
    click.echo("  roboscribe generate 'lift the cube' -e Lift --backend anthropic")
    click.echo()
    click.echo("  # DeepSeek")
    click.echo("  export DEEPSEEK_API_KEY=sk-...")
    click.echo("  roboscribe generate 'lift the cube' -e Lift --backend deepseek")
    click.echo()


@cli.command()
def envs():
    """List supported robosuite environments."""
    from roboscribe.sim.env_registry import ENV_REGISTRY

    click.secho("Supported environments:\n", fg="cyan", bold=True)
    for name, info in sorted(ENV_REGISTRY.items()):
        click.secho(f"  {name:<20}", fg="green", nl=False)
        click.echo(f" {info.description}")
        if info.objects:
            click.echo(f"  {'':20}  Objects: {', '.join(info.objects)}")
    click.echo()


@cli.command()
@click.argument("policy_file")
@click.option("--env", "-e", default=None, help="Environment (auto-detected from file)")
@click.option("--output", "-o", default="rollout.mp4", help="Output video path")
@click.option("--episodes", "-n", default=1, help="Number of episodes to record")
@click.option("--camera", default="agentview", help="Camera name (agentview, frontview, sideview)")
@click.option("--fps", default=30, help="Video FPS")
def record(policy_file, env, output, episodes, camera, fps):
    """Record a policy rollout as an MP4 video."""
    import os

    if not os.path.isfile(policy_file):
        click.secho(f"File not found: {policy_file}", fg="red", err=True)
        sys.exit(1)

    # Auto-detect env from file
    if env is None:
        with open(policy_file) as f:
            for line in f:
                if "environment:" in line.lower():
                    env = line.split(":")[-1].strip()
                    break
        if env is None:
            click.secho("Could not detect environment. Use --env to specify.", fg="red", err=True)
            sys.exit(1)

    click.secho(f"Recording {policy_file} on {env} ({episodes} episode(s))...", fg="cyan")
    click.secho(f"Camera: {camera} | Output: {output}", fg="cyan")

    try:
        from roboscribe.sim.recorder import record_policy
        record_policy(
            policy_file, env,
            output_path=output,
            num_episodes=episodes,
            camera=camera,
            fps=fps,
        )
        click.secho(f"\nVideo saved to: {output}", fg="green", bold=True)
    except Exception as e:
        click.secho(f"Error: {e}", fg="red", err=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()
