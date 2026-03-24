from __future__ import annotations

from rich.text import Text

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container
from textual.screen import ModalScreen
from textual.widgets import DataTable, Input, Label, Select, Static

from .core import CompareSession, DTYPES


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
        self.query_one(Select).focus()
        self.call_after_refresh(self._mark_ready)

    def _mark_ready(self) -> None:
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

    BINDINGS = [
        Binding("q", "quit", "quit"),
        Binding("escape", "reset", "reset"),
        Binding("g", "goto", "goto"),
        Binding("s", "set_slice", "slice"),
        Binding("d", "toggle_diff_only", "diff-only"),
        Binding("t", "set_dtype", "dtype"),
        Binding("r", "reshape", "reshape"),
        Binding("n", "next_diff", "next diff"),
        Binding("p", "prev_diff", "prev diff"),
    ]

    def __init__(self, file1: str, file2: str | None, dtype: str = "f32", shape: tuple[int, ...] | None = None) -> None:
        super().__init__()
        self.session = CompareSession(file1, file2, dtype=dtype, shape=shape)

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

    def refresh_table(self) -> None:
        table = self.query_one(DataTable)
        table.clear(columns=True)
        table.add_columns(
            "#",
            "coord",
            self.session.file1.name,
            self.session.file2.name if self.session.file2 is not None else "",
            "abs diff",
            "rel diff",
        )
        for row in self.session.rows():
            left = self._cell(row.left, row.equal)
            if self.session.compare_mode:
                right = self._cell(row.right, row.equal)
                abs_diff = self._metric_cell(row.abs_diff, row.equal)
                rel_diff = self._metric_cell(row.rel_diff, row.equal)
            else:
                right = Text("")
                abs_diff = Text("")
                rel_diff = Text("")
            table.add_row(
                str(row.flat),
                row.coords_text,
                left,
                right,
                abs_diff,
                rel_diff,
                key=str(row.flat),
            )
        self._sync_cursor()

    def _sync_cursor(self) -> None:
        table = self.query_one(DataTable)
        if self.session.view_size:
            table.move_cursor(row=self.session.view_row, column=0)
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
        self.session.goto_view_row(event.cursor_row)
        self._update_status()

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
