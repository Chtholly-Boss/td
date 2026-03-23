# td

Minimal keyboard-first tensor diff TUI for inspecting one `.bin` file or comparing it against a second `.bin` file.

## Install

```bash
pip install textual numpy typer
```

## Run

```bash
./td test_data/test1.bin
./td test_data/test1.bin test_data/test2.bin
./td test_data/test1.bin test_data/test2.bin -t f32 -s 10,10
./td test_data/test1.bin test_data/test2.bin -s 2,2,-1
```

Short name: **td**

If `/root/.local/bin` is on your `PATH`, you can also run:

```bash
td test_data/test1.bin [file2.bin]
```

## Keys

- `Arrow Up` / `Arrow Down`: move
- `g`: goto flat index or coordinates
- `n` / `p`: next / previous diff
- `s`: slice view, for example `0:63` or `1, 0:63`
- `d`: toggle a diff-only view
- `t`: change dtype
- `r`: reshape, or use `auto`
- `Esc`: on the main screen, reset to the original dtype, shape, and full view
- `q`: quit
- `Esc`: in a prompt, cancel the current input
- `Enter`: confirm current input

Supported dtypes include `b8`, `b16`, and `b32` as binary display types shown in fixed-width hex.

Reshape supports a single inferred dimension with `-1`, similar to torch. For example, `2,2,-1` over 100 elements becomes `2,2,25`.

## Design

- One table for both files, so scrolling is always synchronized
- Real filenames shown in the UI
- No mouse-only controls
- Small pure comparison core in `core.py`
- Thin Textual shell in `app.py`

## Files

- `core.py`: loading, reshape, dtype handling, diff tracking, cursor state
- `app.py`: keyboard-first TUI
- `main.py`: CLI entrypoint
- `tests/test_golden_cmp.py`: regression tests
