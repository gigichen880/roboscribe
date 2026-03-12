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
@click.option("--env", "-e", required=True, help="Robosuite environment name (e.g. Lift, Stack)")
@click.option("--robot", "-r", default="Panda", help="Robot to use (default: Panda)")
@click.option("--backend", "-b", default=None, help="LLM backend: openai, anthropic, qwen, qwen-beijing, qwen-singapore, or deepseek")
@click.option("--model", "-m", default=None, help="LLM model name")
@click.option("--api-key", default=None, help="API key (or set ROBOSCRIBE_API_KEY)")
@click.option("--base-url", default=None, help="Custom API base URL for OpenAI-compatible providers")
@click.option("--max-attempts", default=5, help="Max self-debugging iterations")
@click.option("--episodes", default=10, help="Episodes per evaluation")
@click.option("--output", "-o", default=".", help="Output directory")
@click.option("--interactive", "-i", is_flag=True, help="Interactive review: inspect code, provide feedback, or edit between iterations")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def generate(task_description, env, robot, backend, model, api_key, base_url,
             max_attempts, episodes, output, interactive, verbose):
    """Generate a scripted policy from a natural language description."""
    from roboscribe.config import Config
    from roboscribe.agent.loop import AgentLoop

    cfg = Config.from_env(
        llm_backend=backend,
        llm_model=model,
        api_key=api_key,
        base_url=base_url,
        robot=robot,
        max_attempts=max_attempts,
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

    loop = AgentLoop(cfg)
    click.secho(f"Generating policy for: {task_description}", fg="cyan")
    click.secho(f"Environment: {env} | Robot: {robot} | Backend: {cfg.llm_backend}/{cfg.llm_model}", fg="cyan")
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
