# td

Minimal keyboard-first tensor diff TUI for inspecting one `.bin` file or comparing it against a second `.bin` file.

## Install

From a local clone:

```bash
pip install .
```

Or install directly from GitHub without cloning:

```bash
pip install "git+https://github.com/Chtholly-Boss/td.git"
```

## Run

```bash
td a.bin
td a.bin b.bin
td a.bin b.bin -t f32 -s 10,10
td a.bin b.bin -s 2,2,-1
```

## Keys

- `Arrow Up` / `Arrow Down`: move
- `g`: goto flat index or coordinates
- `n` / `p`: next / previous diff
- `s`: slice view, for example `0:63` or `1, 0:63`
- `d`: toggle a diff-only view
- `t`: change dtype
- `r`: reshape, for example `2,-1` to reshape a `[100]` tensor to `[2,50]`
- `Esc`: on the main screen, reset to the original dtype, shape, and full view
- `q`: quit
- `Esc`: in a prompt, cancel the current input
- `Enter`: confirm current input


