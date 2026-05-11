# src/nithinlang/cli.py
"""
NithinLang Command-Line Interface & Package Manager
=====================================================

Global command: `nithin`

Subcommands
-----------
  nithin run    <file.nl>          Run a NithinLang program
  nithin check  <file.nl>          Syntax-check without running
  nithin new    <project_name>     Scaffold a new project
  nithin info   <file.nl>          Show parse metadata (language, keywords)
  nithin version                   Print version info
  nithin langs                     List supported languages
  nithin bench  <file.nl>          Run + display execution timing
  nithin repl                      Interactive REPL (experimental)

Options
-------
  --verbose / -v    Enable compiler diagnostics
  --no-jit          Disable LLVM JIT (pure CPython execution)
  --help            Show help message
"""

from __future__ import annotations

import os
import sys
import time
import textwrap
from typing import Optional

# ── Third-party (installed with the package) ──────────────────────────────
try:
    import click
    _CLICK = True
except ImportError:
    _CLICK = False

try:
    from rich.console   import Console
    from rich.panel     import Panel
    from rich.syntax    import Syntax
    from rich.table     import Table
    from rich           import print as rprint
    _RICH = True
except ImportError:
    _RICH = False

# ── NithinLang internals ──────────────────────────────────────────────────
from nithinlang             import __version__
from nithinlang.parser      import NithinParser
from nithinlang.compiler    import NithinCompiler
from nithinlang.dicts       import list_languages


# ---------------------------------------------------------------------------
# Console helper (Rich or plain)
# ---------------------------------------------------------------------------

if _RICH:
    _console = Console(highlight=True)
    def _print(msg: str, style: str = "") -> None:
        _console.print(msg, style=style)
    def _print_error(msg: str) -> None:
        _console.print(f"[bold red]✗[/bold red] {msg}")
    def _print_success(msg: str) -> None:
        _console.print(f"[bold green]✓[/bold green] {msg}")
    def _print_info(msg: str) -> None:
        _console.print(f"[bold cyan]ℹ[/bold cyan] {msg}")
    def _print_banner() -> None:
        banner = Panel(
            f"[bold yellow]NithinLang[/bold yellow] [white]v{__version__}[/white]\n"
            "[dim]100% Free · Open-Source · Zero-Cloud · Multi-Lingual · Ultra-Fast[/dim]",
            border_style="bright_blue",
            expand=False,
        )
        _console.print(banner)
else:
    def _print(msg: str, style: str = "") -> None:       # type: ignore[misc]
        print(msg)
    def _print_error(msg: str) -> None:                  # type: ignore[misc]
        print(f"[ERROR] {msg}", file=sys.stderr)
    def _print_success(msg: str) -> None:                # type: ignore[misc]
        print(f"[OK] {msg}")
    def _print_info(msg: str) -> None:                   # type: ignore[misc]
        print(f"[INFO] {msg}")
    def _print_banner() -> None:                         # type: ignore[misc]
        print(f"NithinLang v{__version__}")
        print("100% Free · Open-Source · Zero-Cloud · Multi-Lingual · Ultra-Fast")


# ---------------------------------------------------------------------------
# Core execution helper (shared by run / bench / check)
# ---------------------------------------------------------------------------

def _execute_file(
    filepath : str,
    verbose  : bool = False,
    jit      : bool = True,
    dry_run  : bool = False,   # check only, don't execute
) -> int:
    """
    Parse, compile, and optionally execute a .nl file.

    Returns:
        Exit code: 0 = success, 1 = error.
    """
    # ── Validate file ──────────────────────────────────────────────────────
    if not os.path.isfile(filepath):
        _print_error(f"File not found: '{filepath}'")
        return 1

    if not filepath.endswith(".nl"):
        _print_error(
            f"NithinLang files must have the '.nl' extension. Got: '{filepath}'"
        )
        return 1

    # ── Parse ──────────────────────────────────────────────────────────────
    parser = NithinParser()
    t0     = time.perf_counter()
    result = parser.parse_file(filepath)
    parse_time = time.perf_counter() - t0

    if not result.success:
        _print_error(f"Parse errors in '{filepath}':")
        for err in result.errors:
            _print(str(err), style="red")
        return 1

    if verbose:
        _print_info(
            f"Parsed in {parse_time*1000:.2f} ms  "
            f"[language: {result.language}]  "
            f"[keywords translated: {len(result.keyword_map)}]"
        )

    if dry_run:
        _print_success(f"Syntax OK — '{filepath}' (language: {result.language})")
        return 0

    # ── Compile + Run ──────────────────────────────────────────────────────
    compiler = NithinCompiler(
        verbose       = verbose,
        jit_threshold = 1 if jit else 999_999,
    )
    compile_result = compiler.compile_and_run(result)

    if not compile_result.success:
        _print_error(f"Execution failed in '{filepath}':")
        for err in compile_result.errors:
            _print(err, style="red")
        return 1

    if verbose:
        jit_info = (
            f"  JIT-compiled functions: {compile_result.jit_functions}"
            if compile_result.jit_functions
            else "  Engine: CPython bytecode"
        )
        _print_info(
            f"Executed in {compile_result.execution_time*1000:.2f} ms{jit_info}"
        )

    return 0


# ---------------------------------------------------------------------------
# Project scaffolding
# ---------------------------------------------------------------------------

_HELLO_TEMPLATES = {
    "english": textwrap.dedent("""\
        nithin
        lang+ english

        # Hello World in NithinLang (English)
        print("Hello, World!")
        print("Welcome to NithinLang V1!")

        # Simple loop
        for i in range(5):
            print("Iteration:", i)

        # Function definition
        def greet(name: str) -> None:
            print("Hello,", name)

        greet("Developer")

        end nithin
    """),
    "telugu": textwrap.dedent("""\
        nithin
        lang+ telugu

        # Telugu lo Hello World
        raayi("Namaskaram, Praapanicham!")
        raayi("NithinLang ki swaagatam!")

        # Loop
        ki_varaku i lo range_lo(5):
            raayi("Iterationi:", i)

        # Function
        cheyyandi greet(name):
            raayi("Namaskaram,", name)

        greet("Developer")

        end nithin
    """),
    "hindi": textwrap.dedent("""\
        nithin
        lang+ hindi

        # Hindi mein Hello World
        likho("Namaste, Duniya!")
        likho("NithinLang mein aapka swagat hai!")

        # Loop
        ke_liye i mein seema_mein(5):
            likho("Avrutti:", i)

        # Function
        kaam greet(naam):
            likho("Namaste,", naam)

        greet("Developer")

        end nithin
    """),
}

_MAIN_NL_TEMPLATE = _HELLO_TEMPLATES["english"]

_GITIGNORE = textwrap.dedent("""\
    __pycache__/
    *.pyc
    *.pyo
    .nithin_cache/
    *.pkl
    *.model
    output/
    .DS_Store
""")

_README_TEMPLATE = textwrap.dedent("""\
    # {project_name}

    A NithinLang V1 project.

    ## Running

    ```bash
    nithin run main.nl
    ```

    ## Project Structure

    ```
    {project_name}/
    ├── main.nl          # Entry point
    ├── README.md
    └── .gitignore
    ```

    ## Language

    This project uses **English** syntax by default.
    Change `lang+ english` to `lang+ telugu` or `lang+ hindi` on line 2.
""")


def _scaffold_project(name: str) -> int:
    """Create a new NithinLang project directory."""
    proj_dir = os.path.join(os.getcwd(), name)
    if os.path.exists(proj_dir):
        _print_error(f"Directory already exists: '{proj_dir}'")
        return 1

    os.makedirs(proj_dir)

    # main.nl
    with open(os.path.join(proj_dir, "main.nl"), "w", encoding="utf-8") as f:
        f.write(_MAIN_NL_TEMPLATE)

    # README.md
    with open(os.path.join(proj_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(_README_TEMPLATE.format(project_name=name))

    # .gitignore
    with open(os.path.join(proj_dir, ".gitignore"), "w", encoding="utf-8") as f:
        f.write(_GITIGNORE)

    # examples/
    examples_dir = os.path.join(proj_dir, "examples")
    os.makedirs(examples_dir)
    for lang, template in _HELLO_TEMPLATES.items():
        with open(
            os.path.join(examples_dir, f"hello_{lang}.nl"),
            "w", encoding="utf-8"
        ) as f:
            f.write(template)

    _print_success(f"Project created: '{proj_dir}'")
    _print_info("Run it with:")
    _print(f"  cd {name} && nithin run main.nl")
    return 0


# ---------------------------------------------------------------------------
# REPL
# ---------------------------------------------------------------------------

def _run_repl(verbose: bool = False) -> None:
    """
    Interactive NithinLang REPL.
    Wraps each entered statement in the nithin/end nithin envelope
    with lang+ english, accumulates definitions, and executes.
    """
    _print_banner()
    _print_info("NithinLang REPL (type 'exit' or Ctrl-C to quit)")
    _print_info("Language: english (use 'setlang telugu' to switch)")

    parser      = NithinParser()
    compiler    = NithinCompiler(verbose=verbose)
    accumulated = ""    # accumulated definitions (defs, classes, etc.)
    lang        = "english"

    while True:
        try:
            try:
                line = input("nithin> ").strip()
            except EOFError:
                print()
                break

            if not line:
                continue
            if line.lower() in ("exit", "quit", "bye"):
                _print("Goodbye! 👋")
                break
            if line.lower().startswith("setlang "):
                new_lang = line.split(None, 1)[1].strip()
                lang = new_lang
                _print_info(f"Language set to: {lang}")
                continue
            if line.lower() == "clear":
                accumulated = ""
                _print_info("Accumulated state cleared.")
                continue

            # Wrap in envelope
            snippet = (
                f"nithin\n"
                f"lang+ {lang}\n"
                f"{accumulated}\n"
                f"{line}\n"
                f"end nithin"
            )

            result = parser.parse(snippet)
            if not result.success:
                for err in result.errors:
                    _print_error(str(err))
                continue

            cr = compiler.compile_and_run(result)
            if not cr.success:
                for err in cr.errors:
                    _print(err, style="red")
            else:
                # If line is a def/class, accumulate it
                stripped = line.lstrip()
                if stripped.startswith(("def ", "class ")):
                    accumulated += "\n" + line

        except KeyboardInterrupt:
            print()
            _print_info("KeyboardInterrupt. Type 'exit' to quit.")


# ---------------------------------------------------------------------------
# CLI entry points (Click or argparse fallback)
# ---------------------------------------------------------------------------

if _CLICK:

    @click.group(
        invoke_without_command=True,
        context_settings={"help_option_names": ["-h", "--help"]},
    )
    @click.version_option(__version__, "--version", "-V", prog_name="NithinLang")
    @click.pass_context
    def main(ctx: click.Context) -> None:
        """
        NithinLang V1 — Multi-Lingual Ultra-Fast Programming Language.

        \b
        Examples:
          nithin run   hello.nl
          nithin check hello.nl
          nithin new   my_project
          nithin repl
          nithin langs
        """
        if ctx.invoked_subcommand is None:
            _print_banner()
            click.echo(ctx.get_help())

    @main.command("run")
    @click.argument("filepath",      type=click.Path(exists=False))
    @click.option("--verbose", "-v", is_flag=True, help="Enable compiler diagnostics.")
    @click.option("--no-jit",        is_flag=True, help="Disable LLVM JIT compilation.")
    def cmd_run(filepath: str, verbose: bool, no_jit: bool) -> None:
        """Run a NithinLang (.nl) file."""
        code = _execute_file(filepath, verbose=verbose, jit=not no_jit)
        sys.exit(code)

    @main.command("check")
    @click.argument("filepath", type=click.Path(exists=False))
    @click.option("--verbose", "-v", is_flag=True)
    def cmd_check(filepath: str, verbose: bool) -> None:
        """Check syntax of a NithinLang file without executing it."""
        code = _execute_file(filepath, verbose=verbose, dry_run=True)
        sys.exit(code)

    @main.command("new")
    @click.argument("project_name")
    def cmd_new(project_name: str) -> None:
        """Scaffold a new NithinLang project."""
        code = _scaffold_project(project_name)
        sys.exit(code)

    @main.command("info")
    @click.argument("filepath", type=click.Path(exists=False))
    def cmd_info(filepath: str) -> None:
        """Show metadata about a NithinLang file (language, keywords, etc.)."""
        if not os.path.isfile(filepath):
            _print_error(f"File not found: '{filepath}'")
            sys.exit(1)

        parser = NithinParser()
        result = parser.parse_file(filepath)

        if _RICH:
            table = Table(title=f"NithinLang File Info: {filepath}")
            table.add_column("Property", style="cyan bold")
            table.add_column("Value",    style="white")
            table.add_row("Language",           result.language)
            table.add_row("Parse Success",      "✓ Yes" if result.success else "✗ No")
            table.add_row("Keywords in Dict",   str(len(result.keyword_map)))
            table.add_row("Errors",             str(len(result.errors)))
            table.add_row("Warnings",           str(len(result.warnings)))
            table.add_row("Source lines",       str(len(result.raw_source.splitlines())))
            _console.print(table)

            if result.keyword_map:
                kw_table = Table(title="Active Keyword Translations")
                kw_table.add_column("Native Keyword", style="yellow")
                kw_table.add_column("NithinLang Token", style="green")
                for k, v in sorted(result.keyword_map.items()):
                    kw_table.add_row(k, v)
                _console.print(kw_table)
        else:
            print(f"File:       {filepath}")
            print(f"Language:   {result.language}")
            print(f"Success:    {result.success}")
            print(f"Keywords:   {len(result.keyword_map)}")
            print(f"Errors:     {len(result.errors)}")

        if not result.success:
            for err in result.errors:
                _print_error(str(err))
            sys.exit(1)

    @main.command("bench")
    @click.argument("filepath", type=click.Path(exists=False))
    @click.option("--runs", "-n", default=1, help="Number of benchmark runs.", show_default=True)
    @click.option("--verbose", "-v", is_flag=True)
    def cmd_bench(filepath: str, runs: int, verbose: bool) -> None:
        """Run a NithinLang file and print detailed timing statistics."""
        if not os.path.isfile(filepath):
            _print_error(f"File not found: '{filepath}'")
            sys.exit(1)

        parser   = NithinParser()
        compiler = NithinCompiler(verbose=verbose)

        parse_result = parser.parse_file(filepath)
        if not parse_result.success:
            for err in parse_result.errors:
                _print_error(str(err))
            sys.exit(1)

        times = []
        for run_n in range(runs):
            cr = compiler.compile_and_run(parse_result)
            if not cr.success:
                _print_error(f"Run {run_n+1} failed.")
                for err in cr.errors:
                    _print(err, style="red")
                sys.exit(1)
            times.append(cr.execution_time)

        avg   = sum(times) / len(times)
        best  = min(times)
        worst = max(times)

        if _RICH:
            table = Table(title=f"Benchmark: {filepath} ({runs} run{'s' if runs>1 else ''})")
            table.add_column("Metric",      style="cyan")
            table.add_column("Time (ms)",   style="yellow", justify="right")
            table.add_row("Best",  f"{best*1000:.3f}")
            table.add_row("Worst", f"{worst*1000:.3f}")
            table.add_row("Average", f"{avg*1000:.3f}")
            if cr.jit_functions:
                table.add_row("JIT Functions", ", ".join(cr.jit_functions))
            _console.print(table)
        else:
            print(f"Benchmark: {filepath}  ({runs} run{'s' if runs>1 else ''})")
            print(f"  Best:    {best*1000:.3f} ms")
            print(f"  Worst:   {worst*1000:.3f} ms")
            print(f"  Average: {avg*1000:.3f} ms")

    @main.command("version")
    def cmd_version() -> None:
        """Print NithinLang version information."""
        _print_banner()
        if _RICH:
            table = Table(show_header=False, box=None, padding=(0, 2))
            table.add_column("Key",   style="cyan")
            table.add_column("Value", style="white")
            table.add_row("Version",  __version__)
            table.add_row("Python",   sys.version.split()[0])
            table.add_row("Platform", sys.platform)
            import nithinlang
            table.add_row("Install",  os.path.dirname(nithinlang.__file__))
            _console.print(table)
        else:
            print(f"NithinLang v{__version__}")
            print(f"Python: {sys.version.split()[0]}")

    @main.command("langs")
    def cmd_langs() -> None:
        """List all supported programming languages."""
        langs = list_languages()
        if _RICH:
            table = Table(title="Supported Languages")
            table.add_column("#",        style="dim", width=4)
            table.add_column("Language", style="cyan bold")
            table.add_column("Example Keyword (print)", style="yellow")
            examples = {
                "english": "print",
                "telugu" : "raayi",
                "hindi"  : "likho",
            }
            for i, lang in enumerate(langs, 1):
                table.add_row(
                    str(i),
                    lang.capitalize(),
                    examples.get(lang, "(see dicts/)"),
                )
            _console.print(table)
        else:
            for lang in langs:
                print(f"  {lang}")

    @main.command("repl")
    @click.option("--verbose", "-v", is_flag=True)
    def cmd_repl(verbose: bool) -> None:
        """Start the interactive NithinLang REPL."""
        _run_repl(verbose=verbose)

else:
    # ── Fallback: argparse-based CLI (when click is not installed) ─────────

    import argparse

    def main() -> None:  # type: ignore[misc]
        """Argparse fallback CLI for when click is not installed."""
        parser_cli = argparse.ArgumentParser(
            prog        = "nithin",
            description = f"NithinLang v{__version__} — Multi-Lingual Ultra-Fast Programming Language",
        )
        parser_cli.add_argument("--version", action="version", version=f"NithinLang v{__version__}")

        sub = parser_cli.add_subparsers(dest="command")

        # run
        p_run = sub.add_parser("run",   help="Run a .nl file")
        p_run.add_argument("filepath")
        p_run.add_argument("--verbose", "-v", action="store_true")
        p_run.add_argument("--no-jit",        action="store_true")

        # check
        p_chk = sub.add_parser("check", help="Syntax-check a .nl file")
        p_chk.add_argument("filepath")

        # new
        p_new = sub.add_parser("new",   help="Scaffold a new project")
        p_new.add_argument("project_name")

        # info
        p_inf = sub.add_parser("info",  help="Show file metadata")
        p_inf.add_argument("filepath")

        # bench
        p_ben = sub.add_parser("bench", help="Benchmark a .nl file")
        p_ben.add_argument("filepath")
        p_ben.add_argument("--runs", "-n", type=int, default=1)

        # langs
        sub.add_parser("langs",   help="List supported languages")

        # repl
        sub.add_parser("repl",    help="Interactive REPL")

        args = parser_cli.parse_args()

        if args.command == "run":
            code = _execute_file(
                args.filepath,
                verbose = args.verbose,
                jit     = not args.no_jit,
            )
            sys.exit(code)

        elif args.command == "check":
            code = _execute_file(args.filepath, dry_run=True)
            sys.exit(code)

        elif args.command == "new":
            code = _scaffold_project(args.project_name)
            sys.exit(code)

        elif args.command == "info":
            p    = NithinParser()
            res  = p.parse_file(args.filepath)
            print(f"Language: {res.language}")
            print(f"Success:  {res.success}")
            print(f"Keywords: {len(res.keyword_map)}")
            if not res.success:
                for err in res.errors:
                    _print_error(str(err))
                sys.exit(1)

        elif args.command == "bench":
            _execute_file(args.filepath, verbose=True)

        elif args.command == "langs":
            for lang in list_languages():
                print(f"  {lang}")

        elif args.command == "repl":
            _run_repl()

        else:
            _print_banner()
            parser_cli.print_help()