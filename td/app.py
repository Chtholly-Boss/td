from __future__ import annotations

from rich.text import Text

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container
from textual.screen import ModalScreen
from textual.widgets import DataTable, Input, Label, Select, Static

from .core import CompareSession, DIFF_MODES, DTYPES


class InputScreen(ModalScreen[str | None]):
    BINDINGS = [
        Binding("enter", "confirm", show=False),
        Binding("escape", "cancel", show=False),
    ]

    def __init__(
        self,
        title: str,
        default: str = "",
        placeholder: str = "",
        dialog_class: str = "",
    ) -> None:
        super().__init__()
        self.title = title
        self.default = default
        self.placeholder = placeholder
        self.dialog_class = dialog_class

    def compose(self) -> ComposeResult:
        with Container(id="dialog", classes=self.dialog_class):
            yield Label(self.title)
            yield Input(value=self.default, placeholder=self.placeholder, id="input-field")

    def on_mount(self) -> None:
        self.query_one(Input).focus()

    def action_confirm(self) -> None:
        self.dismiss(self.query_one(Input).value)

    def action_cancel(self) -> None:
        self.dismiss(None)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.dismiss(event.value)


class DTypeScreen(ModalScreen[str | None]):
    BINDINGS = [
        Binding("escape", "cancel", show=False),
    ]

    def __init__(self, default: str, choices: tuple[str, ...]) -> None:
        super().__init__()
        self.default = default
        self.choices = choices
        self._ready = False

    def compose(self) -> ComposeResult:
        with Container(id="dialog", classes="dtype-dialog"):
            yield Label("dtype")
            yield Select.from_values(
                self.choices,
                value=self.default,
                allow_blank=False,
                id="dtype-field",
            )

    def on_mount(self) -> None:
        self.call_after_refresh(self._activate_select)

    def _activate_select(self) -> None:
        select = self.query_one(Select)
        select.focus()
        select.action_show_overlay()
        self._ready = True

    def action_cancel(self) -> None:
        self.dismiss(None)

    def on_select_changed(self, event: Select.Changed) -> None:
        if self._ready and event.value != Select.NULL:
            self.dismiss(str(event.value))


class ModeScreen(ModalScreen[str | None]):
    BINDINGS = [
        Binding("escape", "cancel", show=False),
    ]

    def __init__(self, default: str, choices: tuple[str, ...]) -> None:
        super().__init__()
        self.default = default
        self.choices = choices
        self._ready = False

    def compose(self) -> ComposeResult:
        with Container(id="dialog", classes="mode-dialog"):
            yield Label("diff mode")
            yield Select.from_values(
                self.choices,
                value=self.default,
                allow_blank=False,
                id="mode-field",
            )

    def on_mount(self) -> None:
        self.call_after_refresh(self._activate_select)

    def _activate_select(self) -> None:
        select = self.query_one(Select)
        select.focus()
        select.action_show_overlay()
        self._ready = True

    def action_cancel(self) -> None:
        self.dismiss(None)

    def on_select_changed(self, event: Select.Changed) -> None:
        if self._ready and event.value != Select.NULL:
            self.dismiss(str(event.value))


class TDApp(App):
    TITLE = "td"
    CSS = ""
    CSS_PATH = "app.tcss"
    TABLE_WINDOW_ROWS = 1000

    BINDINGS = [
        Binding("up", "cursor_up", show=False, priority=True),
        Binding("down", "cursor_down", show=False, priority=True),
        Binding("pageup", "page_up", show=False, priority=True),
        Binding("pagedown", "page_down", show=False, priority=True),
        Binding("home", "cursor_home", show=False, priority=True),
        Binding("end", "cursor_end", show=False, priority=True),
        Binding("q", "quit", "quit"),
        Binding("escape", "reset", "reset"),
        Binding("g", "goto", "goto"),
        Binding("s", "set_slice", "slice"),
        Binding("d", "toggle_diff_only", "diff-only"),
        Binding("m", "set_diff_mode", "mode"),
        Binding("t", "set_dtype", "dtype"),
        Binding("r", "reshape", "reshape"),
        Binding("n", "next_diff", "next diff"),
        Binding("p", "prev_diff", "prev diff"),
    ]

    def __init__(self, file1: str, file2: str | None, dtype: str | None = None, shape: tuple[int, ...] | None = None) -> None:
        super().__init__()
        self.session = CompareSession(file1, file2, dtype=dtype, shape=shape)
        self._window_start = 0
        self._window_end = 0
        self._pending_mode_key: str | None = None

    def compose(self) -> ComposeResult:
        yield DataTable(id="table")
        yield Static(id="status")
        yield Static(self._help_text(), id="help")

    def on_mount(self) -> None:
        table = self.query_one(DataTable)
        table.cursor_type = "row"
        table.zebra_stripes = True
        table.focus()
        self.refresh_table()

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        if self.screen is not self.screen_stack[0]:
            return False
        return super().check_action(action, parameters)

    def refresh_table(self) -> None:
        table = self.query_one(DataTable)
        table.clear(columns=True)
        table.add_columns(
            "#",
            "coord",
            self.session.file1.name,
            self.session.file2.name if self.session.file2 is not None else "",
            self.session.diff_column_label,
        )
        self._window_start, self._window_end = self._window_bounds()
        for row in self.session.rows_for_view(self._window_start, self._window_end):
            left = self._cell(row.left, row.equal)
            if self.session.compare_mode:
                right = self._cell(row.right, row.equal)
                diff_value = self._diff_cell(row)
            else:
                right = Text("")
                diff_value = Text("")
            table.add_row(
                str(row.flat),
                row.coords_text,
                left,
                right,
                diff_value,
                key=str(row.flat),
            )
        self._sync_cursor()

    def _window_bounds(self) -> tuple[int, int]:
        total = self.session.view_size
        if total <= 0:
            return 0, 0
        limit = min(self.TABLE_WINDOW_ROWS, total)
        start = max(0, self.session.view_row - limit // 2)
        end = min(total, start + limit)
        start = max(0, end - limit)
        return start, end

    def _sync_cursor(self) -> None:
        table = self.query_one(DataTable)
        if self.session.view_size:
            if not self._window_start <= self.session.view_row < self._window_end:
                self.refresh_table()
                return
            table.move_cursor(row=self.session.view_row - self._window_start, column=0)
        self._update_status()

    def _update_status(self, message: str | None = None) -> None:
        if message is None:
            message = self._status_text()
        self.query_one("#status", Static).update(message)

    def _cell(self, value: object, equal: bool) -> Text:
        style = "" if equal else "bold red"
        return Text(self.session.format_value(value), style=style)

    def _metric_cell(self, value: object, equal: bool) -> Text:
        style = "" if equal else "bold red"
        return Text(self.session.format_metric(value), style=style)

    def _diff_cell(self, row) -> Text:
        value = self.session.diff_display_value(row)
        if value is None:
            return Text("")
        if isinstance(value, str):
            rendered = value
        else:
            rendered = self.session.format_metric(value)
        return Text(rendered, style="bold red")

    def _status_text(self) -> str:
        parts = []
        if self.session.view_size:
            row = self.session.row(self.session.cursor)
            parts.append(f"row {self.session.view_row + 1}/{self.session.view_size} @ {row.flat} ({row.coords_text})")
        else:
            parts.append("row 0/0")
        parts.extend(
            [
                f"dtype {self.session.dtype}",
                f"shape {self.session.shape_text}",
            ]
        )
        parts.append(f"mode {self.session.diff_mode.key}")
        if self.session.active_threshold is not None:
            parts.append(f"threshold {self.session.format_metric(self.session.active_threshold)}")
        if self.session.slice_spec:
            parts.append(f"slice {self.session.slice_spec}")
        if self.session.diff_only:
            parts.append("view diffs only")
        parts.append(f"diffs {len(self.session.diffs)}")
        return " | ".join(parts)

    def _help_text(self) -> Text:
        text = Text()
        key_style = "bold #f5f5f5 on #4a4a4a"
        label_style = "dim"
        separator_style = "dim"
        parts = [
            ("Arrow Up/Down", "move"),
            ("g", "goto"),
            ("s", "slice"),
            ("d", "diff-only"),
            ("m", "mode"),
            ("Esc", "reset"),
            ("n/p", "diff"),
            ("t", "dtype"),
            ("r", "reshape"),
            ("q", "quit"),
        ]
        for index, (key, label) in enumerate(parts):
            if index:
                text.append(" | ", style=separator_style)
            text.append(f" {key} ", style=key_style)
            text.append(f" {label}", style=label_style)
        return text

    def _prompt(
        self,
        title: str,
        default: str,
        placeholder: str,
        callback,
        dialog_class: str = "",
    ) -> None:
        self.push_screen(
            InputScreen(
                title,
                default=default,
                placeholder=placeholder,
                dialog_class=dialog_class,
            ),
            callback,
        )

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if self.session.view_size == 0:
            self._update_status()
            return
        row = self._window_start + event.cursor_row
        if 0 <= row < self.session.view_size:
            self.session.goto_view_row(row)
        self._update_status()

    def _move_cursor(self, delta: int) -> None:
        self.session.move(delta)
        self._sync_cursor()

    def action_cursor_up(self) -> None:
        self._move_cursor(-1)

    def action_cursor_down(self) -> None:
        self._move_cursor(1)

    def action_page_up(self) -> None:
        self._move_cursor(-(max(1, self.TABLE_WINDOW_ROWS - 1)))

    def action_page_down(self) -> None:
        self._move_cursor(max(1, self.TABLE_WINDOW_ROWS - 1))

    def action_cursor_home(self) -> None:
        if self.session.view_size:
            self.session.goto_view_row(0)
            self._sync_cursor()

    def action_cursor_end(self) -> None:
        if self.session.view_size:
            self.session.goto_view_row(self.session.view_size - 1)
            self._sync_cursor()

    def action_goto(self) -> None:
        self._prompt("goto flat index or coords", ",".join(map(str, self.session.coords)), "12 or 1,2", self._apply_goto)

    def _apply_goto(self, value: str | None) -> None:
        if value is None:
            return
        try:
            self.session.goto(value)
            self._sync_cursor()
        except Exception as error:
            self._update_status(f"error: {error}")

    def action_set_slice(self) -> None:
        default = self.session.slice_spec or ""
        placeholder = "0:63" if len(self.session.shape) == 1 else "0:63 or 1, 0:63"
        self._prompt("slice", default, placeholder, self._apply_slice)

    def _apply_slice(self, value: str | None) -> None:
        if value is None:
            return
        try:
            self.session.set_slice(value)
            self.refresh_table()
        except Exception as error:
            self._update_status(f"error: {error}")

    def action_toggle_diff_only(self) -> None:
        if not self.session.diff_only and not self.session.diffs:
            self._update_status("no diffs")
            return
        try:
            self.session.toggle_diff_only()
            self.refresh_table()
        except Exception as error:
            self._update_status(f"error: {error}")

    def action_set_dtype(self) -> None:
        self.push_screen(DTypeScreen(self.session.dtype, tuple(DTYPES)), self._apply_dtype)

    def action_set_diff_mode(self) -> None:
        self.push_screen(ModeScreen(self.session.diff_mode.key, tuple(DIFF_MODES)), self._apply_diff_mode)

    def _apply_diff_mode(self, value: str | None) -> None:
        if value is None:
            return
        try:
            needs_threshold = DIFF_MODES[value].requires_threshold
            current_threshold = self.session.numeric_threshold(value) if needs_threshold else None
            if needs_threshold and (value == self.session.diff_mode.key or current_threshold is None):
                self._pending_mode_key = value
                default = "" if current_threshold is None else self.session.format_metric(current_threshold)
                self._prompt(
                    f"{value} threshold",
                    default,
                    "0.01",
                    self._apply_mode_threshold,
                    dialog_class="threshold-dialog",
                )
                return
            self.session.set_diff_mode(value)
            self.refresh_table()
        except Exception as error:
            self._pending_mode_key = None
            self._update_status(f"error: {error}")

    def _apply_mode_threshold(self, value: str | None) -> None:
        mode_key = self._pending_mode_key
        self._pending_mode_key = None
        if value is None or mode_key is None:
            return
        try:
            threshold = float(value.strip())
            self.session.set_diff_mode(mode_key, threshold=threshold)
            self.refresh_table()
        except Exception as error:
            self._update_status(f"error: {error}")

    def _apply_dtype(self, value: str | None) -> None:
        if value is None:
            return
        try:
            self.session.set_dtype(value)
            self.refresh_table()
        except Exception as error:
            self._update_status(f"error: {error}")

    def action_reshape(self) -> None:
        self._prompt(
            "shape",
            self.session.shape_text,
            "10,10,-1 or auto",
            self._apply_shape,
            dialog_class="shape-dialog",
        )

    def _apply_shape(self, value: str | None) -> None:
        if value is None:
            return
        try:
            self.session.set_shape(value)
            self.refresh_table()
        except Exception as error:
            self._update_status(f"error: {error}")

    def action_reset(self) -> None:
        try:
            self.session.reset()
            self.refresh_table()
        except Exception as error:
            self._update_status(f"error: {error}")

    def action_next_diff(self) -> None:
        if self.session.next_diff():
            self._sync_cursor()
        else:
            self._update_status("no diffs")

    def action_prev_diff(self) -> None:
        if self.session.prev_diff():
            self._sync_cursor()
        else:
            self._update_status("no diffs")
