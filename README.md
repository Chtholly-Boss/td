# td

Tensor Diff Tool for a keyboard-first TUI experience.

![presentation](https://github.com/user-attachments/assets/29757103-8add-4dd9-8987-620277e6af9f)

## Install

From a local clone:

```bash
pip install .
```

Or install directly from GitHub without cloning:

```bash
pip install "git+https://github.com/Chtholly-Boss/td.git"
```

## Build a binary

```bash
pyinstaller td.spec
```

## Run

```bash
td a.bin
td a.bin b.bin
td a.data b.bin
td a.npy
td a.bin b.bin -t f32 -s 10,10
td a.bin b.bin -s 2,2,-1
```

`.npy` files use their embedded dtype and shape by default.

## Diff Modes

Press `m` to choose how `td` decides whether a row is a diff:

- `raw`: any unequal values are diffs
- `abs`: a row is a diff when its absolute difference meets or exceeds the chosen threshold
- `rel`: a row is a diff when its relative difference meets or exceeds the chosen threshold

`abs` and `rel` prompt for a threshold when selected. Each numeric mode remembers its last threshold for the current session.

## Keys

- `Arrow Up` / `Arrow Down`: move
- `g`: goto flat index or coordinates
- `n` / `p`: next / previous diff in the active diff mode
- `s`: slice view, for example `0:63` or `1, 0:63`
- `d`: toggle a diff-only view for the active diff mode
- `m`: choose the diff mode (`raw`, `abs`, or `rel`)
- `t`: change dtype
- `r`: reshape, for example `2,-1` to reshape a `[100]` tensor to `[2,50]`
- `Esc`: on the main screen, reset to the original dtype, shape, and full view
- `q`: quit
- `Esc`: in a prompt, cancel the current input
- `Enter`: confirm current input
