#!/usr/bin/env python3

from pathlib import Path

import typer
from typer.main import get_command_from_info
from typer.models import CommandInfo

from .app import TDApp
from .core import DTYPES, parse_shape


def run(
    file1: Path,
    file2: Path | None = typer.Argument(None, metavar="FILE2"),
    dtype: str = typer.Option("f32", "--dtype", "-t"),
    shape: str = typer.Option("auto", "--shape", "-s", help="shape like 10,10,10 or 2,-1,2 or auto"),
) -> None:
    dtype = dtype.lower()
    if dtype not in DTYPES:
        raise typer.BadParameter(f"unsupported dtype: {dtype}")
    for path in (file1, file2):
        if path is None:
            continue
        if not path.exists():
            raise typer.BadParameter(f"missing file: {path}")
        if path.suffix != ".bin":
            raise typer.BadParameter(f"unsupported file: {path}")
    try:
        dims = parse_shape(shape)
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error
    TDApp(str(file1), str(file2) if file2 is not None else None, dtype=dtype, shape=dims).run()


app = get_command_from_info(
    CommandInfo(
        name="td",
        callback=run,
        help="td: keyboard-first tensor diff TUI",
        context_settings={"help_option_names": ["-h", "--help"]},
    ),
    pretty_exceptions_short=True,
    rich_markup_mode="rich",
)


if __name__ == "__main__":
    app(prog_name="td")
