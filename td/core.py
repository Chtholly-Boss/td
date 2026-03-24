from __future__ import annotations

from dataclasses import dataclass
from math import prod
from pathlib import Path

import numpy as np


BINARY_DTYPES = {
    "b8": np.uint8,
    "b16": np.uint16,
    "b32": np.uint32,
}

SUPPORTED_FILE_SUFFIXES = {".bin", ".data", ".npy"}


DTYPES = {
    "i8": np.int8,
    "i16": np.int16,
    "i32": np.int32,
    "i64": np.int64,
    "u8": np.uint8,
    "u16": np.uint16,
    "u32": np.uint32,
    "u64": np.uint64,
    **BINARY_DTYPES,
    "f16": np.float16,
    "f32": np.float32,
    "f64": np.float64,
}

NUMPY_DTYPES = {}
for name, dtype in DTYPES.items():
    NUMPY_DTYPES.setdefault(np.dtype(dtype), name)


def parse_shape(shape: str | tuple[int, ...] | None) -> tuple[int, ...] | None:
    if shape in (None, "", "auto"):
        return None
    if isinstance(shape, tuple):
        dims = shape
    else:
        dims = tuple(int(part.strip()) for part in str(shape).split(","))
    if not dims:
        raise ValueError(f"invalid shape: {shape}")
    if any(dim == 0 or dim < -1 for dim in dims):
        raise ValueError(f"invalid shape: {shape}")
    if sum(dim == -1 for dim in dims) > 1:
        raise ValueError(f"invalid shape: {shape}")
    return dims


def resolve_shape(shape: tuple[int, ...] | None, size: int) -> tuple[int, ...]:
    if shape is None:
        return (int(size),)

    if -1 not in shape:
        return shape

    known_prod = prod(dim for dim in shape if dim != -1)
    if known_prod <= 0 or size % known_prod != 0:
        raise ValueError(f"shape {','.join(map(str, shape))} does not fit {size} elements")

    inferred = size // known_prod
    return tuple(inferred if dim == -1 else dim for dim in shape)


def parse_slice_part(part: str) -> int | slice:
    part = part.strip()
    if not part:
        return slice(None)
    if ":" not in part:
        return int(part)
    values = part.split(":")
    if len(values) > 3:
        raise ValueError(f"invalid slice: {part}")
    parsed = [None if value.strip() == "" else int(value.strip()) for value in values]
    return slice(*parsed)


def parse_slice_spec(spec: str | None, ndim: int) -> tuple[int | slice, ...]:
    if spec in (None, ""):
        return (slice(None),) * ndim
    parts = [part.strip() for part in str(spec).split(",")]
    if len(parts) > ndim:
        raise ValueError(f"expected at most {ndim} slice parts")
    selection = [parse_slice_part(part) for part in parts]
    selection.extend([slice(None)] * (ndim - len(selection)))
    return tuple(selection)


@dataclass(frozen=True)
class Row:
    flat: int
    coords: tuple[int, ...]
    left: object
    right: object | None
    abs_diff: float
    rel_diff: float
    equal: bool

    @property
    def coords_text(self) -> str:
        return ",".join(map(str, self.coords))


class CompareSession:
    def __init__(
        self,
        file1: str | Path,
        file2: str | Path | None,
        dtype: str | None = None,
        shape: str | tuple[int, ...] | None = None,
    ) -> None:
        self.file1 = Path(file1)
        self.file2 = Path(file2) if file2 is not None else None
        self.compare_mode = self.file2 is not None
        inferred_dtype, inferred_shape = self._infer_initial_view()
        self.original_dtype = self._parse_dtype(dtype or inferred_dtype or "f32")
        self.original_shape_spec = parse_shape(inferred_shape if shape in (None, "", "auto") else shape)
        self.original_slice_spec: str | None = None
        self.original_diff_only = False
        self.dtype = self.original_dtype
        self.shape_spec = self.original_shape_spec
        self.slice_spec: str | None = None
        self.diff_only = False
        self.cursor = 0
        self.reload()

    def _parse_dtype(self, dtype: str) -> str:
        dtype = dtype.lower()
        if dtype not in DTYPES:
            raise ValueError(f"unsupported dtype: {dtype}")
        return dtype

    def _infer_initial_view(self) -> tuple[str | None, tuple[int, ...] | None]:
        for path in (self.file1, self.file2):
            if path is None or path.suffix != ".npy":
                continue
            array = np.load(path, allow_pickle=False, mmap_mode="r")
            dtype = NUMPY_DTYPES.get(np.dtype(array.dtype))
            if dtype is None:
                raise ValueError(f"unsupported dtype in {path}: {array.dtype}")
            shape = tuple(int(dim) for dim in array.shape) or (int(array.size),)
            return dtype, shape
        return None, None

    def _load_npy_raw(self, path: Path) -> np.ndarray:
        array = np.load(path, allow_pickle=False)
        buffer = np.ascontiguousarray(array).tobytes()
        dtype = np.dtype(DTYPES[self.dtype])
        size = len(buffer) // dtype.itemsize * dtype.itemsize
        if size == 0:
            return np.empty(0, dtype=dtype)
        return np.frombuffer(buffer[:size], dtype=dtype).copy()

    def _load_raw(self, path: Path) -> np.ndarray:
        if not path.exists():
            raise FileNotFoundError(path)
        if path.suffix not in SUPPORTED_FILE_SUFFIXES:
            raise ValueError(f"unsupported file: {path}")
        if path.suffix == ".npy":
            return self._load_npy_raw(path)
        return np.fromfile(path, dtype=DTYPES[self.dtype])

    def reload(self) -> None:
        left = self._load_raw(self.file1)
        right = self._load_raw(self.file2) if self.compare_mode and self.file2 is not None else None
        if right is not None and left.size != right.size:
            raise ValueError(f"size mismatch: {left.size} vs {right.size}")

        self.shape = resolve_shape(self.shape_spec, int(left.size))
        if prod(self.shape) != int(left.size):
            raise ValueError(f"shape {self.shape_text} does not fit {left.size} elements")

        self.left = left.reshape(self.shape)
        self.right = right.reshape(self.shape) if right is not None else None
        self._flat_left = self.left.reshape(-1)
        self._flat_right = self.right.reshape(-1) if self.right is not None else None
        if self._flat_right is None:
            self._equal = np.ones(self._flat_left.size, dtype=bool)
        else:
            self._equal = self._equal_mask(self._flat_left, self._flat_right)
        self._visible_flats = self._visible_flats_for_view()
        self._visible_index_by_flat = {flat: index for index, flat in enumerate(self._visible_flats)}
        self.diffs = [flat for flat in np.flatnonzero(~self._equal).astype(int).tolist() if flat in self._visible_index_by_flat]
        if self._visible_flats and self.cursor not in self._visible_index_by_flat:
            self.cursor = self._visible_flats[0]
        if not self._visible_flats:
            self.cursor = 0

    def _equal_mask(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        if self.dtype.startswith("f"):
            return (
                (left == right)
                | (np.isnan(left) & np.isnan(right))
                | (np.isposinf(left) & np.isposinf(right))
                | (np.isneginf(left) & np.isneginf(right))
            )
        return left == right

    @property
    def size(self) -> int:
        return int(self._flat_left.size)

    @property
    def view_size(self) -> int:
        return len(self._visible_flats)

    @property
    def shape_text(self) -> str:
        return ",".join(map(str, self.shape))

    @property
    def coords(self) -> tuple[int, ...]:
        return tuple(int(x) for x in np.unravel_index(self.cursor, self.shape))

    @property
    def view_row(self) -> int:
        if not self._visible_flats:
            return 0
        return self._visible_index_by_flat[self.cursor]

    def format_value(self, value: object) -> str:
        if self.dtype in BINARY_DTYPES:
            width = int(self.dtype[1:]) // 4
            return f"0x{int(value):0{width}x}"
        return f"{float(value):.6g}" if self.dtype.startswith("f") else str(value)

    def format_metric(self, value: object) -> str:
        return f"{float(value):.6g}"

    def diff_metrics(self, left: object, right: object, equal: bool) -> tuple[float, float]:
        if equal:
            return 0.0, 0.0
        abs_diff = float(abs(left - right))
        scale = max(float(abs(left)), float(abs(right)))
        rel_diff = 0.0 if scale == 0.0 else abs_diff / scale
        return abs_diff, rel_diff

    def _slice_flats(self) -> list[int]:
        selection = parse_slice_spec(self.slice_spec, len(self.shape))
        indices = np.arange(self.size).reshape(self.shape)
        try:
            selected = indices[selection]
        except (IndexError, TypeError, ValueError) as error:
            raise ValueError(f"invalid slice: {self.slice_spec}") from error
        flats = np.asarray(selected, dtype=int).reshape(-1).tolist()
        if not flats:
            raise ValueError(f"slice matched no elements: {self.slice_spec}")
        return [int(flat) for flat in flats]

    def _visible_flats_for_view(self) -> list[int]:
        flats = self._slice_flats()
        if self.diff_only:
            return [flat for flat in flats if not self._equal[flat]]
        return flats

    def row(self, flat: int) -> Row:
        left = self._flat_left[flat].item()
        if self._flat_right is None:
            right = None
            equal = True
            abs_diff = 0.0
            rel_diff = 0.0
        else:
            right = self._flat_right[flat].item()
            equal = bool(self._equal[flat])
            abs_diff, rel_diff = self.diff_metrics(left, right, equal)
        return Row(
            flat=flat,
            coords=tuple(int(x) for x in np.unravel_index(flat, self.shape)),
            left=left,
            right=right,
            abs_diff=abs_diff,
            rel_diff=rel_diff,
            equal=equal,
        )

    def rows(self) -> list[Row]:
        return [self.row(flat) for flat in self._visible_flats]

    def move(self, delta: int) -> None:
        if not self._visible_flats:
            return
        row_index = min(max(self.view_row + delta, 0), self.view_size - 1)
        self.cursor = self._visible_flats[row_index]

    def goto_view_row(self, row: int) -> None:
        if not 0 <= row < self.view_size:
            raise ValueError(f"row out of range: {row}")
        self.cursor = self._visible_flats[row]

    def goto_flat(self, flat: int) -> None:
        if not 0 <= flat < self.size:
            raise ValueError(f"index out of range: {flat}")
        if flat not in self._visible_index_by_flat:
            raise ValueError(f"index not visible: {flat}")
        self.cursor = flat

    def goto(self, target: str) -> None:
        target = target.strip()
        if "," not in target:
            self.goto_flat(int(target))
            return
        coords = tuple(int(part.strip()) for part in target.split(","))
        if len(coords) != len(self.shape):
            raise ValueError(f"expected {len(self.shape)} coordinates")
        for axis, (index, limit) in enumerate(zip(coords, self.shape)):
            if not 0 <= index < limit:
                raise ValueError(f"axis {axis} out of range: {index}")
        self.cursor = int(np.ravel_multi_index(coords, self.shape))

    def next_diff(self) -> bool:
        if not self.diffs:
            return False
        self.cursor = next((diff for diff in self.diffs if diff > self.cursor), self.diffs[0])
        return True

    def prev_diff(self) -> bool:
        if not self.diffs:
            return False
        self.cursor = next((diff for diff in reversed(self.diffs) if diff < self.cursor), self.diffs[-1])
        return True

    def set_dtype(self, dtype: str) -> None:
        previous_dtype = self.dtype
        previous_cursor = self.cursor
        self.dtype = self._parse_dtype(dtype)
        try:
            self.reload()
        except Exception:
            self.dtype = previous_dtype
            self.cursor = previous_cursor
            self.reload()
            raise

    def set_shape(self, shape: str | tuple[int, ...] | None) -> None:
        previous_shape_spec = self.shape_spec
        previous_cursor = self.cursor
        self.shape_spec = parse_shape(shape)
        try:
            self.reload()
        except Exception:
            self.shape_spec = previous_shape_spec
            self.cursor = previous_cursor
            self.reload()
            raise

    def set_slice(self, spec: str | None) -> None:
        previous_slice_spec = self.slice_spec
        previous_cursor = self.cursor
        spec = None if spec is None or str(spec).strip() == "" else str(spec).strip()
        self.slice_spec = spec
        try:
            self.reload()
        except Exception:
            self.slice_spec = previous_slice_spec
            self.cursor = previous_cursor
            self.reload()
            raise

    def toggle_diff_only(self) -> None:
        previous_diff_only = self.diff_only
        previous_cursor = self.cursor
        self.diff_only = not self.diff_only
        try:
            self.reload()
        except Exception:
            self.diff_only = previous_diff_only
            self.cursor = previous_cursor
            self.reload()
            raise

    def reset(self) -> None:
        self.dtype = self.original_dtype
        self.shape_spec = self.original_shape_spec
        self.slice_spec = self.original_slice_spec
        self.diff_only = self.original_diff_only
        self.cursor = 0
        self.reload()
