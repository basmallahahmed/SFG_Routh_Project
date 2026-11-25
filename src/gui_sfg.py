"""
gui_sfg.py  --  Part 1 GUI (Signal Flow Graph + Mason's gain formula), PyQt5.

Window layout
    left  : input panel  (nodes, branches table, input/output nodes, examples, Solve)
    right : the drawn signal flow graph (+ optional path/loop highlight)
            and the scrollable step-by-step results panel.

The main widget is `SFGWidget`, a plain QWidget, so Person 4 can drop it into a
tab of the main window:      tabs.addTab(SFGWidget(), "Part 1: Signal Flow Graph")
`SFGWindow` / `main()` run it on its own:     python -m src.gui_sfg

It uses the other modules only through their public API:
    graph_model.SignalFlowGraph      add_branch / remove_branch / clear ...
    mason.analyze / mason.explain    paths, loops, delta, delta_i, T(s) + text steps
    sfg_draw.draw_graph              the picture
"""
import functools
import html
import json
import math
import os
import re
import sys
import traceback

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QFont, QIcon, QKeySequence, QPainter, QPalette
from PyQt5.QtWidgets import (
    QAbstractItemView, QApplication, QComboBox, QFileDialog, QFrame,
    QGridLayout, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QListWidget,
    QMainWindow, QMessageBox, QPushButton, QScrollArea, QShortcut,
    QSizePolicy, QSplitter, QTableWidget, QTableWidgetItem, QTextEdit,
    QVBoxLayout, QWidget)
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg, NavigationToolbar2QT)
from matplotlib.figure import Figure

try:                                   # normal use:  python -m src.gui_sfg
    from .graph_model import SignalFlowGraph
    from .mason import analyze, explain
    from .sfg_draw import COLORS, draw_graph, format_gain
except ImportError:                    # run directly: python src/gui_sfg.py
    from graph_model import SignalFlowGraph
    from mason import analyze, explain
    from sfg_draw import COLORS, draw_graph, format_gain


# =========================================================================== #
#  Pure helpers (no Qt) - input checking, graph editing, text formatting
# =========================================================================== #
MAX_NAME_LEN = 12
MANY_BRANCHES = 40        # warn before solving graphs bigger than this
ARROW = " \u2192 "


def parse_gain(text):
    """Text -> float, or ValueError with a message meant for the user."""
    raw = (text or "").strip()
    if not raw:
        raise ValueError("Please enter a gain.\nUse a number such as 2, -0.5 or 1.25.")
    try:
        value = float(raw.replace(",", ".").replace("\u2212", "-"))
    except ValueError:
        raise ValueError(f"'{raw}' is not a valid number.\n"
                         "Use digits only, for example 2, -0.5 or 1.25.")
    if not math.isfinite(value):
        raise ValueError("The gain must be a finite number.")
    return value


def check_name(text, what="Node name"):
    """Strip a node name and make sure it is usable."""
    name = (text or "").strip()
    if not name:
        raise ValueError(f"{what} cannot be empty.")
    if len(name) > MAX_NAME_LEN:
        raise ValueError(f"{what} '{name}' is too long "
                         f"(at most {MAX_NAME_LEN} characters).")
    return name


def remove_nodes(graph, names):
    """Delete nodes together with every branch that touches them."""
    names = set(names)
    graph.nodes[:] = [n for n in graph.nodes if n not in names]
    graph.branches[:] = [b for b in graph.branches
                         if b[0] not in names and b[1] not in names]


def replace_branch(graph, index, src, dst, gain):
    """Edit one branch in place (keeps its row). Restores the old one on error."""
    old = graph.branches[index]
    graph.remove_branch(old[0], old[1])
    try:
        graph.add_branch(src, dst, gain)
    except ValueError:
        graph.add_branch(*old)
        graph.branches.insert(index, graph.branches.pop())
        raise
    graph.branches.insert(index, graph.branches.pop())


def friendly_error(exc):
    """Turn a model/solver ValueError into a short, helpful sentence."""
    msg = str(exc)
    m = re.search(r"exactly one (input|output) node, found (\d+): \[(.*?)\]", msg)
    if m:
        which, count, names = m.group(1), int(m.group(2)), m.group(3)
        names = names.replace("'", "").replace('"', "")
        rule = "no incoming branch" if which == "input" else "no outgoing branch"
        if count == 0:
            head = (f"No node can be the {which} node: every node has an "
                    f"{'incoming' if which == 'input' else 'outgoing'} branch.")
        else:
            head = (f"The graph needs exactly one {which} node (a node with "
                    f"{rule}), but {count} were found: {names}.")
        return (f"{head}\n\nConnect or remove the extra nodes, or choose the "
                f"{which} node yourself under \u201cInput / Output nodes\u201d.")
    return msg


# --------------------------------------------------------------------------- #
#  Graph files  (load / save)
#
#  Text format (.txt / .csv), one item per line, '#' starts a comment:
#       node: R, a, b, C        optional - only to fix the node order
#       R, a, 1                 branch:  from, to, gain
#       a -> b : 2              (also accepted)
#       input: R                optional - otherwise auto-detected
#       output: C               optional
#  JSON format (.json):
#       {"nodes": ["R","a"], "branches": [["R","a",1]],
#        "input": "R", "output": "a"}
# --------------------------------------------------------------------------- #
_SPLIT = re.compile(r"\s*(?:,|;|:|->|=>|\u2192)\s*|\s+")


def _parse_text_graph(text):
    nodes, branches = [], []
    src = snk = None
    for no, raw in enumerate(text.splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        key, sep, rest = line.partition(":")
        key = key.strip().lower()
        try:
            if sep and key in ("node", "nodes", "input", "output"):
                names = [check_name(n) for n in re.split(r"[,;\s]+", rest.strip())
                         if n]
                if key in ("node", "nodes"):
                    nodes += names
                else:
                    if len(names) != 1:
                        raise ValueError(f"'{key}:' needs exactly one node name.")
                    if key == "input":
                        src = names[0]
                    else:
                        snk = names[0]
                continue
            tokens = [t for t in _SPLIT.split(line) if t]
            if len(tokens) == 3 and tokens[0].lower() == "from" \
                    and tokens[2].lower() == "gain":
                continue                         # CSV header row
            if len(tokens) != 3:
                raise ValueError("expected  from, to, gain  (for example  R, a, 2.5).")
            branches.append((check_name(tokens[0]), check_name(tokens[1]),
                             parse_gain(tokens[2])))
        except ValueError as exc:
            raise ValueError(f"Line {no}: {exc}")
    return nodes, branches, src, snk


def _parse_json_graph(text):
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"The file is not valid JSON: {exc}")
    if not isinstance(data, dict) or not isinstance(data.get("branches"), list):
        raise ValueError('The JSON file needs a "branches" list.')
    nodes = [check_name(str(n)) for n in data.get("nodes", [])]
    branches = []
    for i, b in enumerate(data["branches"], 1):
        if isinstance(b, dict):
            if not {"from", "to", "gain"} <= set(b):
                raise ValueError(f'Branch {i}: needs "from", "to" and "gain".')
            s, d, g = b["from"], b["to"], b["gain"]
        elif isinstance(b, (list, tuple)) and len(b) == 3:
            s, d, g = b
        else:
            raise ValueError(f"Branch {i}: expected [from, to, gain].")
        try:
            branches.append((check_name(str(s)), check_name(str(d)),
                             parse_gain(str(g))))
        except ValueError as exc:
            raise ValueError(f"Branch {i}: {exc}")
    src, snk = data.get("input"), data.get("output")
    return (nodes, branches, str(src).strip() if src else None,
            str(snk).strip() if snk else None)


def parse_graph_file(text, filename=""):
    """File text -> (nodes, branches, input_or_None, output_or_None)."""
    if filename.lower().endswith(".json") or text.lstrip().startswith("{"):
        result = _parse_json_graph(text)
    else:
        result = _parse_text_graph(text)
    if not result[1]:
        raise ValueError("No branches were found in the file.")
    return result


def graph_to_text(graph, src=None, snk=None):
    lines = ["# Signal flow graph  -  branch format:  from, to, gain",
             "node: " + ", ".join(graph.nodes)]
    lines += [f"{s}, {d}, {format_gain(g)}" for s, d, g in graph.branches]
    if src:
        lines.append(f"input: {src}")
    if snk:
        lines.append(f"output: {snk}")
    return "\n".join(lines) + "\n"


def graph_to_json(graph, src=None, snk=None):
    data = {"nodes": list(graph.nodes),
            "branches": [[s, d, g] for s, d, g in graph.branches]}
    if src:
        data["input"] = src
    if snk:
        data["output"] = snk
    return json.dumps(data, indent=2, ensure_ascii=False) + "\n"


def results_to_html(text):
    """Colour the plain-text explanation produced by mason.explain()."""
    out = []
    for line in text.split("\n"):
        if not line.strip():
            continue
        if line.startswith("==="):
            title = html.escape(line.strip("= ").strip())
            out.append('<p style="margin-top:16px; margin-bottom:2px;">'
                       f'<b><span style="color:#F2A93B; font-size:12pt;">{title}'
                       '</span></b></p><hr/>')
            continue
        body = html.escape(line).replace(" ", "&nbsp;")
        stripped = line.strip()
        if stripped.startswith("T =") or stripped.startswith("Result:"):
            body = f'<b><span style="color:#34D399;">{body}</span></b>'
        elif stripped.startswith("Note:"):
            body = f'<b><span style="color:#FF9F6E;">{body}</span></b>'
        elif stripped.startswith(("Substituting:", "Symbolic:", "Formula:")):
            body = f'<span style="color:#8C99A6;">{body}</span>'
        out.append(f'<div style="margin:1px 0;">{body}</div>')
    return "".join(out)


# --------------------------------------------------------------------------- #
#  Built-in examples (all checked against an independent linear-equation solve)
# --------------------------------------------------------------------------- #
EXAMPLES = [
    {"name": "Example 1 \u2013 2 paths, 5 loops, self-loop",
     "nodes": ["R", "a", "b", "c", "d", "C"],
     "branches": [("R", "a", 1), ("a", "b", 2), ("b", "c", 3), ("c", "d", 4),
                  ("d", "C", 1), ("a", "c", 5), ("b", "a", -1), ("c", "b", -2),
                  ("d", "c", -3), ("d", "d", 0.5)],
     "note": "2 forward paths, 5 loops, four non-touching pairs."},
    {"name": "Example 2 \u2013 simple feedback loop",
     "nodes": ["R", "x1", "x2", "C"],
     "branches": [("R", "x1", 1), ("x1", "x2", 2), ("x2", "C", 3),
                  ("x2", "x1", -1)],
     "note": "1 forward path and 1 loop. A good first test."},
    {"name": "Example 3 \u2013 2 paths, \u0394\u2082 \u2260 1",
     "nodes": ["R", "a", "b", "c", "d", "C"],
     "branches": [("R", "a", 1), ("a", "b", 2), ("b", "c", 3), ("c", "d", 4),
                  ("d", "C", 1), ("R", "c", 6), ("b", "a", -1), ("d", "c", -2),
                  ("c", "b", -1)],
     "note": "The second path skips the first loop, so its cofactor is not 1."},
    {"name": "Example 4 \u2013 three non-touching loops",
     "nodes": ["R", "a", "b", "c", "d", "e", "f", "C"],
     "branches": [("R", "a", 1), ("a", "b", 2), ("b", "c", 1), ("c", "d", 3),
                  ("d", "e", 1), ("e", "f", 4), ("f", "C", 1), ("b", "a", -1),
                  ("d", "c", -2), ("f", "e", -3)],
     "note": "Three loops that never touch, so \u0394 has a triple-product term."},
]


# =========================================================================== #
#  Style
# =========================================================================== #
STYLE = """
QWidget#root { background: #14181C; }
QWidget { font-family: "Bahnschrift", "Segoe UI", "Helvetica Neue", Arial, sans-serif;
          font-size: 10pt; color: #D6DEE6; }
QLabel { background: transparent; }
QToolTip { background: #0F1317; color: #D6DEE6; border: 1px solid #F2A93B;
           padding: 4px; }
QMessageBox, QDialog { background: #1B2026; }

QFrame#topbar { background: #1B2026; border: none; border-bottom: 1px solid #2B333B; }
QLabel#mark { font-family: "Cascadia Mono", "Consolas", "Courier New", monospace; font-size: 16pt; font-weight: 700;
              color: #F2A93B; }
QFrame#vsep { background: #2B333B; border: none; }
QFrame#hline { background: #2B333B; border: none; }
QFrame#panel { background: #181D22; border: 1px solid #2B333B; border-radius: 0; }
QScrollArea#leftScroll, QScrollArea#leftScroll > QWidget > QWidget
    { background: transparent; border: none; }
QLabel#caption { color: #6B7785; font-size: 8pt; font-weight: 700;
                 padding-bottom: 4px; border-bottom: 1px solid #2B333B; }
QLabel#captionPlain { color: #8C99A6; font-size: 8pt; font-weight: 700; }
QLabel#muted { color: #6B7785; font-size: 9pt; }
QLabel#fieldLabel { color: #6B7785; font-size: 8pt; font-weight: 700; }

QLineEdit, QComboBox {
    background: #0F1317; border: 1px solid #2F3943; border-radius: 0;
    padding: 5px 8px; min-height: 18px; selection-background-color: #F2A93B;
    selection-color: #14181C; }
QLineEdit:focus, QComboBox:focus { border: 1px solid #4FD1C5; }
QComboBox QLineEdit { border: none; padding: 0; background: transparent;
                      min-height: 0; }
QComboBox QAbstractItemView { background: #0F1317; border: 1px solid #2F3943;
    selection-background-color: #26343F; selection-color: #FFFFFF; outline: 0; }

QPushButton { background: #1F262D; border: 1px solid #2F3943; border-radius: 0;
              padding: 6px 12px; }
QPushButton:hover { border-color: #4FD1C5; color: #FFFFFF; }
QPushButton:pressed { background: #14181C; }
QPushButton:disabled { color: #4D5864; background: #161B20; border-color: #232A31; }
QPushButton#primary { background: transparent; color: #4FD1C5;
                      border: 1px solid #4FD1C5; font-weight: 700; }
QPushButton#primary:hover { background: #4FD1C5; color: #14181C; }
QPushButton#primary:pressed { background: #3AB5AA; color: #14181C; }
QPushButton#primary:disabled { color: #3A6B67; border-color: #2A4A47;
                               background: transparent; }
QPushButton#solve { background: #F2A93B; color: #14181C; border: none;
                    font-size: 11pt; font-weight: 700; padding: 8px 24px; }
QPushButton#solve:hover { background: #FFBE5C; color: #14181C; }
QPushButton#solve:pressed { background: #D98F22; }
QPushButton#danger { color: #FF6B81; border-color: #4A2A33; background: transparent; }
QPushButton#danger:hover { background: #2B1217; border-color: #FF6B81; }
QPushButton#danger:disabled { color: #5A3A42; border-color: #2A1C20; }
QPushButton#small { padding: 4px 10px; font-size: 9pt; }

QTableWidget, QListWidget { background: #0F1317; border: 1px solid #2B333B;
    border-radius: 0; alternate-background-color: #131920; gridline-color: #1B2026;
    selection-background-color: #26343F; selection-color: #FFFFFF; outline: 0;
    font-family: "Cascadia Mono", "Consolas", "Courier New", monospace; }
QTableWidget::item, QListWidget::item { padding: 3px; }
QHeaderView::section { background: #1B2026; border: none;
    border-bottom: 1px solid #2B333B; padding: 5px; font-weight: 700;
    color: #6B7785; }
QTableCornerButton::section { background: #1B2026; border: none; }

QTextEdit#results { background: #0F1317; border: 1px solid #2B333B;
    border-radius: 0; padding: 6px; font-family: "Cascadia Mono", "Consolas", "Courier New", monospace; font-size: 10pt; }

QScrollBar:vertical { background: transparent; width: 10px; margin: 2px; }
QScrollBar::handle:vertical { background: #2F3943; min-height: 30px; }
QScrollBar::handle:vertical:hover { background: #4D5864; }
QScrollBar:horizontal { background: transparent; height: 10px; margin: 2px; }
QScrollBar::handle:horizontal { background: #2F3943; min-width: 30px; }
QScrollBar::handle:horizontal:hover { background: #4D5864; }
QScrollBar::add-line, QScrollBar::sub-line { width: 0; height: 0; }
QScrollBar::add-page, QScrollBar::sub-page { background: transparent; }

QSplitter::handle { background: transparent; }
QSplitter::handle:hover { background: #2B333B; }
QLabel#status { color: #8C99A6; font-family: "Cascadia Mono", "Consolas", "Courier New", monospace; font-size: 9pt;
                padding: 5px 14px; background: #1B2026;
                border-top: 1px solid #2B333B; }
"""

STATUS_COLORS = {"info": "#8C99A6", "ok": "#34D399", "warn": "#F2A93B",
                 "error": "#FF6B81"}
BANNER_COLORS = {            # background, border, text
    "ok": ("#0B1F1B", "#1F5A4E", "#34D399"),
    "warn": ("#2A1F0B", "#6B4E14", "#F2A93B"),
    "error": ("#2B1217", "#7A2434", "#FF6B81"),
}


def _dark_palette():
    """Dark palette so combo arrows, text cursors etc. are drawn light."""
    pal = QPalette()
    colors = {QPalette.Window: "#14181C", QPalette.WindowText: "#D6DEE6",
              QPalette.Base: "#0F1317", QPalette.AlternateBase: "#131920",
              QPalette.Text: "#D6DEE6", QPalette.Button: "#1F262D",
              QPalette.ButtonText: "#D6DEE6", QPalette.Highlight: "#F2A93B",
              QPalette.HighlightedText: "#14181C",
              QPalette.ToolTipBase: "#0F1317", QPalette.ToolTipText: "#D6DEE6"}
    for role, c in colors.items():
        pal.setColor(role, QColor(c))
    placeholder = getattr(QPalette, "PlaceholderText", None)
    if placeholder is not None:
        pal.setColor(placeholder, QColor("#6B7785"))
    for role in (QPalette.WindowText, QPalette.Text, QPalette.ButtonText):
        pal.setColor(QPalette.Disabled, role, QColor("#5B6672"))
    return pal


def safe(method):
    """Run a slot; show any unexpected error in a message box instead of crashing."""
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        except Exception as exc:                       # noqa: BLE001
            traceback.print_exc()
            QApplication.restoreOverrideCursor()
            self._error("Something went wrong",
                        f"An unexpected error occurred:\n\n{exc}\n\n"
                        "If it keeps happening, press \u201cClear all\u201d "
                        "and try again.")
    return wrapper


def _paren(x):
    return f"({x})" if isinstance(x, (int, float)) and x < 0 else f"{x}"


# =========================================================================== #
#  The Part 1 widget
# =========================================================================== #
class SFGWidget(QWidget):
    """Complete Part 1 screen. Embed it anywhere (tab, window, dialog)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("root")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setPalette(_dark_palette())
        self.setStyleSheet(STYLE)
        self.graph = SignalFlowGraph()
        self.result = None            # last Mason result (or None)
        self._result_text = ""        # plain text of the last explanation
        self._build_ui()
        self._wire_up()
        self._changed()               # fills the widgets and draws the empty hint
        self.set_status("Ready \u2013 add nodes and branches, or load an example.")

    # ------------------------------------------------------------------ UI --
    @staticmethod
    def _card(title, subtitle=None):
        frame = QFrame()
        frame.setObjectName("card")
        lay = QVBoxLayout(frame)
        lay.setContentsMargins(16, 14, 16, 14)
        lay.setSpacing(10)
        head = QLabel(title)
        head.setObjectName("cardTitle")
        lay.addWidget(head)
        if subtitle:
            sub = QLabel(subtitle)
            sub.setObjectName("muted")
            sub.setWordWrap(True)
            lay.addWidget(sub)
        return frame, lay

    @staticmethod
    def _field(text):
        lab = QLabel(text)
        lab.setObjectName("fieldLabel")
        return lab

    @staticmethod
    def _button(text, name=None, tip=None):
        btn = QPushButton(text)
        if name:
            btn.setObjectName(name)
        if tip:
            btn.setToolTip(tip)
        btn.setCursor(Qt.PointingHandCursor)
        return btn

    @staticmethod
    def _caption(text):
        lab = QLabel(text.upper())
        lab.setObjectName("caption")
        return lab

    @staticmethod
    def _vsep():
        line = QFrame()
        line.setObjectName("vsep")
        line.setFixedWidth(1)
        return line

    @staticmethod
    def _hline():
        line = QFrame()
        line.setObjectName("hline")
        line.setFixedHeight(1)
        return line

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(self._build_header())

        body = QSplitter(Qt.Horizontal)
        body.setChildrenCollapsible(False)
        body.setHandleWidth(8)
        body.addWidget(self._build_left())
        body.addWidget(self._build_right())
        body.setStretchFactor(0, 0)
        body.setStretchFactor(1, 1)
        body.setSizes([390, 1000])
        wrap = QWidget()
        wl = QVBoxLayout(wrap)
        wl.setContentsMargins(10, 10, 10, 10)
        wl.addWidget(body)
        outer.addWidget(wrap, 1)

        self.status = QLabel("")
        self.status.setObjectName("status")
        outer.addWidget(self.status)

        # Combo boxes must not size themselves to their longest item, or the
        # panels become wider than their box and get cut off.
        for combo, chars in ((self.from_combo, 4), (self.to_combo, 4),
                             (self.src_combo, 8), (self.snk_combo, 8),
                             (self.example_combo, 10), (self.hl_combo, 6)):
            combo.setSizeAdjustPolicy(
                QComboBox.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(chars)

    def _build_header(self):
        bar = QFrame()
        bar.setObjectName("topbar")
        row = QHBoxLayout(bar)
        row.setContentsMargins(14, 6, 14, 6)
        row.setSpacing(12)
        mark = QLabel("SFG\u25b8")
        mark.setObjectName("mark")
        tag = QLabel("Signal flow graph  \u00b7  Mason\u2019s gain formula")
        tag.setObjectName("muted")
        row.addWidget(mark)
        row.addWidget(tag)
        row.addStretch(1)
        return bar

    def _build_left(self):
        """Input column, in the order you use it: start -> nodes -> branches
        -> input/output -> Solve."""
        holder = QFrame()
        holder.setObjectName("panel")
        holder.setMinimumWidth(350)
        holder.setMaximumWidth(450)
        outer = QVBoxLayout(holder)
        outer.setContentsMargins(12, 10, 12, 12)
        outer.setSpacing(10)

        scroll = QScrollArea()
        scroll.setObjectName("leftScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        inner = QWidget()
        lay = QVBoxLayout(inner)
        lay.setContentsMargins(0, 0, 6, 0)
        lay.setSpacing(8)
        scroll.setWidget(inner)
        outer.addWidget(scroll, 1)

        # -- start from an example or a file --
        lay.addWidget(self._caption("Start from"))
        row = QHBoxLayout()
        row.setSpacing(6)
        self.example_combo = QComboBox()
        for ex in EXAMPLES:
            self.example_combo.addItem(ex["name"])
        self.load_btn = self._button("Load example", None,
                                     "Load the chosen example.")
        row.addWidget(self.example_combo, 1)
        row.addWidget(self.load_btn)
        lay.addLayout(row)
        row = QHBoxLayout()
        row.setSpacing(6)
        self.load_file_btn = self._button(
            "Open file\u2026", None,
            "Read a graph from a .txt, .csv or .json file.")
        self.save_file_btn = self._button(
            "Save graph\u2026", None,
            "Save the current graph so it can be loaded again later.")
        row.addWidget(self.load_file_btn, 1)
        row.addWidget(self.save_file_btn, 1)
        lay.addLayout(row)

        # -- nodes --
        lay.addSpacing(8)
        lay.addWidget(self._caption("Nodes"))
        row = QHBoxLayout()
        row.setSpacing(6)
        self.node_edit = QLineEdit()
        self.node_edit.setPlaceholderText("node name")
        self.node_edit.setClearButtonEnabled(True)
        self.add_node_btn = self._button("Add node", "primary")
        row.addWidget(self.node_edit, 1)
        row.addWidget(self.add_node_btn)
        lay.addLayout(row)
        self.node_list = QListWidget()
        self.node_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.node_list.setAlternatingRowColors(True)
        self.node_list.setFixedHeight(62)
        lay.addWidget(self.node_list)
        self.del_node_btn = self._button(
            "Remove selected node(s)", "danger",
            "Also removes every branch connected to the node.")
        lay.addWidget(self.del_node_btn)

        # -- branches --
        lay.addSpacing(8)
        lay.addWidget(self._caption("Branches"))
        row = QHBoxLayout()
        row.setSpacing(6)
        self.from_combo = QComboBox()
        self.to_combo = QComboBox()
        for combo, hint in ((self.from_combo, "from"), (self.to_combo, "to")):
            combo.setEditable(True)
            combo.setInsertPolicy(QComboBox.NoInsert)
            combo.lineEdit().setPlaceholderText(hint)
        self.gain_edit = QLineEdit()
        self.gain_edit.setPlaceholderText("gain")
        row.addWidget(self.from_combo, 3)
        row.addWidget(self.to_combo, 3)
        row.addWidget(self.gain_edit, 2)
        lay.addLayout(row)
        row = QHBoxLayout()
        row.setSpacing(6)
        self.add_branch_btn = self._button(
            "Add branch", "primary",
            "Use the same node twice for a self-loop. New names are "
            "created automatically.")
        self.update_btn = self._button(
            "Update", None,
            "Replace the selected table row with the values above.")
        self.update_btn.setEnabled(False)
        row.addWidget(self.add_branch_btn, 1)
        row.addWidget(self.update_btn, 1)
        lay.addLayout(row)
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["FROM", "TO", "GAIN"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.setShowGrid(False)
        self.table.setMinimumHeight(150)
        lay.addWidget(self.table, 1)
        row = QHBoxLayout()
        row.setSpacing(6)
        self.del_branch_btn = self._button(
            "Delete", "danger", "Shortcut: Delete key inside the table.")
        self.del_branch_btn.setEnabled(False)
        self.clear_btn = self._button("Clear all", "danger")
        row.addWidget(self.del_branch_btn, 1)
        row.addWidget(self.clear_btn, 1)
        lay.addLayout(row)

        # -- input / output --
        lay.addSpacing(8)
        lay.addWidget(self._caption("Input / output"))
        grid = QGridLayout()
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(3)
        self.src_combo = QComboBox()
        self.snk_combo = QComboBox()
        self.src_combo.setToolTip("Auto-detect = the node with no incoming branch.")
        self.snk_combo.setToolTip("Auto-detect = the node with no outgoing branch.")
        grid.addWidget(self._field("INPUT NODE"), 0, 0)
        grid.addWidget(self._field("OUTPUT NODE"), 0, 1)
        grid.addWidget(self.src_combo, 1, 0)
        grid.addWidget(self.snk_combo, 1, 1)
        lay.addLayout(grid)
        self.io_hint = QLabel("")
        self.io_hint.setObjectName("muted")
        self.io_hint.setWordWrap(True)
        lay.addWidget(self.io_hint)

        # -- Solve: always visible, right after the inputs --
        self.solve_btn = self._button(
            "Solve  \u25b8", "solve",
            "Find paths, loops, \u0394, \u0394\u1d62 and T(s).   "
            "Shortcut: Ctrl+Enter")
        outer.addWidget(self.solve_btn)
        return holder

    def _tint_toolbar(self, color="#C5D0DB"):
        """matplotlib's toolbar icons are dark - recolour them for the dark UI."""
        try:
            for act in self.toolbar.actions():
                icon = act.icon()
                if icon.isNull():
                    continue
                pm = icon.pixmap(48, 48)
                if pm.isNull():
                    continue
                painter = QPainter(pm)
                painter.setCompositionMode(QPainter.CompositionMode_SourceIn)
                painter.fillRect(pm.rect(), QColor(color))
                painter.end()
                act.setIcon(QIcon(pm))
        except Exception:                                  # noqa: BLE001
            traceback.print_exc()

    def _build_right(self):
        holder = QSplitter(Qt.Vertical)
        holder.setChildrenCollapsible(False)
        holder.setHandleWidth(8)

        # -- graph --
        card = QFrame()
        card.setObjectName("panel")
        lay = QVBoxLayout(card)
        lay.setContentsMargins(12, 10, 12, 8)
        lay.setSpacing(6)
        head = QHBoxLayout()
        title = QLabel("GRAPH")
        title.setObjectName("captionPlain")
        head.addWidget(title)
        head.addStretch(1)
        head.addWidget(self._field("HIGHLIGHT"))
        self.hl_combo = QComboBox()
        self.hl_combo.setMinimumWidth(280)
        self.hl_combo.setToolTip(
            "Emphasise one forward path or loop (available after Solve).")
        head.addWidget(self.hl_combo)
        lay.addLayout(head)
        lay.addWidget(self._hline())
        self.fig = Figure(facecolor=COLORS["bg"])
        self.ax = self.fig.add_axes([0.0, 0.0, 1.0, 1.0])
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.setMinimumHeight(250)
        lay.addWidget(self.canvas, 1)
        self.toolbar = NavigationToolbar2QT(self.canvas, self, coordinates=False)
        self.toolbar.setStyleSheet(
            "QToolBar { background: transparent; border: none; }"
            "QToolButton { border: 1px solid transparent; padding: 2px; }"
            "QToolButton:hover { background: #26343F; border-color: #4FD1C5; }"
            "QToolButton:checked { background: #3A2F12; border-color: #F2A93B; }")
        self._tint_toolbar()
        lay.addWidget(self.toolbar)
        # re-fit the whole graph whenever the canvas changes size
        self.canvas.mpl_connect("resize_event", lambda evt: self._refit())
        holder.addWidget(card)

        # -- solution --
        card = QFrame()
        card.setObjectName("panel")
        lay = QVBoxLayout(card)
        lay.setContentsMargins(12, 10, 12, 10)
        lay.setSpacing(6)
        head = QHBoxLayout()
        title = QLabel("SOLUTION")
        title.setObjectName("captionPlain")
        head.addWidget(title)
        head.addStretch(1)
        self.copy_btn = self._button("Copy text", "small")
        self.save_btn = self._button("Save .txt", "small")
        head.addWidget(self.copy_btn)
        head.addWidget(self.save_btn)
        lay.addLayout(head)
        lay.addWidget(self._hline())
        self.banner = QLabel("")
        self.banner.setObjectName("banner")
        self.banner.setWordWrap(True)
        self.banner.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.banner.hide()
        lay.addWidget(self.banner)
        self.results = QTextEdit()
        self.results.setObjectName("results")
        self.results.setReadOnly(True)
        self.results.setLineWrapMode(QTextEdit.NoWrap)
        self.results.setPlaceholderText(
            "Press \u201cSolve\u201d to see every forward path, loop, "
            "non-touching combination, \u0394, \u0394\u1d62 and T(s) "
            "with formulas and numerical substitution.")
        lay.addWidget(self.results, 1)
        holder.addWidget(card)

        holder.setStretchFactor(0, 5)
        holder.setStretchFactor(1, 5)
        holder.setSizes([440, 400])
        return holder

    def _wire_up(self):
        self.add_node_btn.clicked.connect(lambda *_: self.add_node())
        self.node_edit.returnPressed.connect(lambda: self.add_node())
        self.del_node_btn.clicked.connect(lambda *_: self.delete_nodes())
        self.add_branch_btn.clicked.connect(lambda *_: self.add_branch())
        self.gain_edit.returnPressed.connect(lambda: self.add_branch())
        self.update_btn.clicked.connect(lambda *_: self.update_branch())
        self.del_branch_btn.clicked.connect(lambda *_: self.delete_branches())
        self.clear_btn.clicked.connect(lambda *_: self.clear_all())
        self.table.itemSelectionChanged.connect(lambda: self._on_row_selected())
        self.load_btn.clicked.connect(lambda *_: self.load_example())
        self.load_file_btn.clicked.connect(lambda *_: self.load_file())
        self.save_file_btn.clicked.connect(lambda *_: self.save_graph())
        self.solve_btn.clicked.connect(lambda *_: self.solve())
        self.copy_btn.clicked.connect(lambda *_: self.copy_results())
        self.save_btn.clicked.connect(lambda *_: self.save_results())
        self.hl_combo.currentIndexChanged.connect(lambda *_: self.redraw())
        self.src_combo.currentIndexChanged.connect(lambda *_: self._io_changed())
        self.snk_combo.currentIndexChanged.connect(lambda *_: self._io_changed())

        for keys in ("Ctrl+Return", "Ctrl+Enter"):
            sc = QShortcut(QKeySequence(keys), self)
            sc.setContext(Qt.WidgetWithChildrenShortcut)
            sc.activated.connect(lambda: self.solve())
        dl = QShortcut(QKeySequence("Delete"), self.table)
        dl.setContext(Qt.WidgetShortcut)
        dl.activated.connect(lambda: self.delete_branches())

    # ------------------------------------------------------- messages -----
    def set_status(self, text, kind="info"):
        self.status.setStyleSheet(f"color: {STATUS_COLORS[kind]};")
        self.status.setText(text)

    def _error(self, title, text):
        self.set_status(title, "error")
        QMessageBox.warning(self, title, text)

    def _info(self, title, text):
        self.set_status(title, "warn")
        QMessageBox.information(self, title, text)

    def _show_banner(self, text, kind):
        bg, border, fg = BANNER_COLORS[kind]
        self.banner.setStyleSheet(
            f"QLabel#banner {{ background: {bg}; border: 1px solid {border}; "
            f"border-left: 4px solid {fg}; color: {fg}; border-radius: 0; "
            f"padding: 10px 14px; font-family: 'Cascadia Mono', 'Consolas', "
            f"monospace; font-size: 12pt; font-weight: 700; }}")
        self.banner.setText(text)
        self.banner.show()

    # --------------------------------------------------- state refresh -----
    def _changed(self, status=None):
        """Call after ANY edit of the graph: old results are no longer valid."""
        had_results = self.result is not None
        self.result = None
        self._result_text = ""
        self.results.clear()
        self.banner.hide()
        self.hl_combo.blockSignals(True)
        self.hl_combo.clear()
        self.hl_combo.addItem("No highlight", None)
        self.hl_combo.blockSignals(False)
        self.hl_combo.setEnabled(False)
        self.copy_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        self._refresh_nodes()
        self._refresh_table()
        self.redraw()
        if status:
            self.set_status(status, "ok")
        elif had_results:
            self.set_status("Graph edited \u2013 press Solve to update the results.",
                            "warn")

    def _refresh_nodes(self):
        nodes = list(self.graph.nodes)
        self.node_list.clear()
        self.node_list.addItems(nodes)
        for combo in (self.from_combo, self.to_combo):
            text = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(nodes)
            combo.setEditText(text)
            combo.blockSignals(False)
        for combo in (self.src_combo, self.snk_combo):
            keep = combo.currentData()
            combo.blockSignals(True)
            combo.clear()
            combo.addItem("Auto-detect", None)
            for n in nodes:
                combo.addItem(n, n)
            idx = combo.findData(keep) if keep is not None else 0
            combo.setCurrentIndex(max(idx, 0))
            combo.blockSignals(False)
        self._refresh_io_hint()

    def _refresh_table(self):
        self.table.blockSignals(True)
        self.table.setRowCount(len(self.graph.branches))
        for r, (s, d, g) in enumerate(self.graph.branches):
            for c, text in enumerate((s, d, format_gain(g))):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(r, c, item)
        self.table.blockSignals(False)
        self._on_row_selected()

    def _auto_io(self):
        try:
            src = self.graph.get_source()
        except ValueError:
            src = None
        try:
            snk = self.graph.get_sink()
        except ValueError:
            snk = None
        return src, snk

    def _refresh_io_hint(self):
        if not self.graph.branches:
            self.io_hint.setText("")
            return
        src, snk = self._auto_io()
        self.io_hint.setText(f"Auto-detected \u2192 input: {src or '?'}   \u00b7   "
                             f"output: {snk or '?'}")

    def _current_io(self):
        """The user's explicit choice (None = auto-detect)."""
        return self.src_combo.currentData(), self.snk_combo.currentData()

    def _effective_io(self):
        src, snk = self._current_io()
        auto_src, auto_snk = self._auto_io()
        return (src if src is not None else auto_src,
                snk if snk is not None else auto_snk)

    @safe
    def _io_changed(self):
        if self.result is not None:
            self._changed("Input/output changed \u2013 press Solve to update.")
        else:
            self.redraw()

    @safe
    def redraw(self):
        src, snk = self._effective_io()
        draw_graph(self.ax, self.graph, src, snk, self.hl_combo.currentData())
        self.toolbar.update()          # forget old zoom/pan history so Home
        self.canvas.draw_idle()        # returns to THIS graph's full view

    def _refit(self):
        """Window resized: re-fit the whole graph to the new canvas size."""
        try:
            src, snk = self._effective_io()
            draw_graph(self.ax, self.graph, src, snk,
                       self.hl_combo.currentData())
            self.toolbar.update()
        except Exception:              # never let a resize crash the app
            traceback.print_exc()

    @safe
    def _on_row_selected(self):
        rows = self._selected_rows()
        self.update_btn.setEnabled(len(rows) == 1)
        self.del_branch_btn.setEnabled(len(rows) >= 1)
        if len(rows) == 1 and rows[0] < len(self.graph.branches):
            s, d, g = self.graph.branches[rows[0]]
            self.from_combo.setEditText(s)
            self.to_combo.setEditText(d)
            self.gain_edit.setText(format_gain(g))

    def _selected_rows(self):
        return sorted({item.row() for item in self.table.selectedItems()})

    # ----------------------------------------------------- editing --------
    @safe
    def add_node(self):
        try:
            name = check_name(self.node_edit.text())
        except ValueError as exc:
            self._error("Invalid node name", str(exc))
            return
        if name in self.graph.nodes:
            self._info("Node already exists",
                       f"Node '{name}' is already in the graph.")
            return
        self.graph.add_node(name)
        self.node_edit.clear()
        self._changed(f"Node '{name}' added.")
        self.node_edit.setFocus()

    def _read_branch_form(self):
        """Validated (src, dst, gain) from the form, or None after a message."""
        try:
            src = check_name(self.from_combo.currentText(),
                             "The \u201cFrom\u201d node")
            dst = check_name(self.to_combo.currentText(),
                             "The \u201cTo\u201d node")
            gain = parse_gain(self.gain_edit.text())
        except ValueError as exc:
            self._error("Check the branch", str(exc))
            return None
        return src, dst, gain

    @safe
    def add_branch(self):
        form = self._read_branch_form()
        if form is None:
            return
        src, dst, gain = form
        created = [n for n in dict.fromkeys((src, dst))
                   if n not in self.graph.nodes]
        try:
            self.graph.add_branch(src, dst, gain)
        except ValueError as exc:
            self._error("Cannot add branch",
                        f"{exc}\n\nTo change it, select its row in the table "
                        "and press \u201cUpdate selected\u201d.")
            return
        self.gain_edit.clear()
        extra = ""
        if created:
            extra = f" (new node{'s' * (len(created) > 1)}: {', '.join(created)})"
        self._changed(f"Branch {src} \u2192 {dst} (gain {format_gain(gain)}) "
                      f"added{extra}.")
        self.gain_edit.setFocus()

    @safe
    def update_branch(self):
        rows = self._selected_rows()
        if len(rows) != 1:
            self._info("Select one row", "Click one row in the table first.")
            return
        form = self._read_branch_form()
        if form is None:
            return
        src, dst, gain = form
        try:
            replace_branch(self.graph, rows[0], src, dst, gain)
        except ValueError as exc:
            self._error("Cannot update branch", str(exc))
            return
        self._changed(f"Branch updated: {src} \u2192 {dst} "
                      f"(gain {format_gain(gain)}).")

    @safe
    def delete_branches(self):
        rows = self._selected_rows()
        if not rows:
            self._info("Nothing selected",
                       "Select one or more rows in the table first.")
            return
        for src, dst in [self.graph.branches[r][:2] for r in rows]:
            self.graph.remove_branch(src, dst)
        self._changed(f"{len(rows)} branch{'es' * (len(rows) > 1)} deleted.")

    @safe
    def delete_nodes(self):
        names = [i.text() for i in self.node_list.selectedItems()]
        if not names:
            self._info("Nothing selected",
                       "Select one or more nodes in the list first.")
            return
        remove_nodes(self.graph, names)
        self._changed(f"Removed node{'s' * (len(names) > 1)}: {', '.join(names)}.")

    @safe
    def clear_all(self):
        if not self.graph.nodes and not self.graph.branches:
            self.set_status("The graph is already empty.")
            return
        ok = QMessageBox.question(
            self, "Clear everything?",
            "This removes all nodes and branches. Continue?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if ok != QMessageBox.Yes:
            return
        self.graph.clear()
        self.src_combo.setCurrentIndex(0)
        self.snk_combo.setCurrentIndex(0)
        self._changed("Cleared.")

    @safe
    def load_example(self):
        ex = EXAMPLES[self.example_combo.currentIndex()]
        if self.graph.branches:
            ok = QMessageBox.question(
                self, "Replace the current graph?",
                "Loading an example replaces the current graph. Continue?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            if ok != QMessageBox.Yes:
                return
        self.graph.clear()
        for n in ex["nodes"]:
            self.graph.add_node(n)
        for s, d, g in ex["branches"]:
            self.graph.add_branch(s, d, g)
        self.src_combo.setCurrentIndex(0)
        self.snk_combo.setCurrentIndex(0)
        self._changed(f"{ex['name']} loaded. {ex['note']} Press Solve.")

    @safe
    def load_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load graph from file", "",
            "Graph files (*.txt *.csv *.json);;All files (*)")
        if not path:
            return
        try:
            with open(path, encoding="utf-8-sig") as fh:
                text = fh.read()
        except (OSError, UnicodeDecodeError) as exc:
            self._error("Could not read the file", f"{exc}")
            return
        try:
            nodes, branches, src, snk = parse_graph_file(text, path)
        except ValueError as exc:
            self._error("Invalid graph file", str(exc))
            return
        known = set(nodes) | {n for s, d, _ in branches for n in (s, d)}
        for label, name in (("input", src), ("output", snk)):
            if name is not None and name not in known:
                self._error("Invalid graph file",
                            f"The {label} node '{name}' is not used in the graph.")
                return
        if src is not None and src == snk:
            self._error("Invalid graph file",
                        "The input and output nodes must be different.")
            return
        if self.graph.branches:
            ok = QMessageBox.question(
                self, "Replace the current graph?",
                "Loading a file replaces the current graph. Continue?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            if ok != QMessageBox.Yes:
                return

        old_nodes, old_branches = list(self.graph.nodes), list(self.graph.branches)
        self.graph.clear()
        try:
            for n in nodes:
                if n not in self.graph.nodes:
                    self.graph.add_node(n)
            for s, d, g in branches:
                self.graph.add_branch(s, d, g)
        except ValueError as exc:
            self.graph.clear()                      # put the old graph back
            for n in old_nodes:
                self.graph.add_node(n)
            for s, d, g in old_branches:
                self.graph.add_branch(s, d, g)
            self._changed()
            self._error("Invalid graph file",
                        f"{exc}\n\nThe previous graph was kept.")
            return

        self.src_combo.setCurrentIndex(0)
        self.snk_combo.setCurrentIndex(0)
        self._changed(f"Loaded {len(self.graph.nodes)} nodes and "
                      f"{len(self.graph.branches)} branches from "
                      f"{os.path.basename(path)}. Press Solve.")
        for combo, name in ((self.src_combo, src), (self.snk_combo, snk)):
            idx = combo.findData(name) if name is not None else -1
            if idx >= 0:
                combo.setCurrentIndex(idx)

    @safe
    def save_graph(self):
        if not self.graph.branches:
            self._info("Nothing to save", "Add at least one branch first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save graph", "my_graph.txt",
            "Text files (*.txt);;JSON files (*.json)")
        if not path:
            return
        src, snk = self._current_io()
        writer = graph_to_json if path.lower().endswith(".json") else graph_to_text
        try:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(writer(self.graph, src, snk))
        except OSError as exc:
            self._error("Could not save the file", f"{exc}")
            return
        self.set_status(f"Graph saved to {path}", "ok")

    # -------------------------------------------------------- solving -----
    @safe
    def solve(self):
        if not self.graph.branches:
            self._info("Nothing to solve",
                       "Add at least one branch first (or load an example).")
            return
        src, snk = self._current_io()
        if src is not None and src == snk:
            self._error("Input and output are the same",
                        "The input node and the output node must be "
                        "different nodes.")
            return
        if len(self.graph.branches) > MANY_BRANCHES:
            ok = QMessageBox.question(
                self, "Large graph",
                f"This graph has {len(self.graph.branches)} branches. Finding "
                "all loops can take a long time and the window may freeze for "
                "a while.\n\nContinue?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if ok != QMessageBox.Yes:
                return

        problem = None
        result = text = None
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            result = analyze(self.graph, src, snk)
            text = explain(result)
        except ValueError as exc:
            problem = ("Cannot solve this graph", friendly_error(exc))
        except RecursionError:
            problem = ("Graph too complex",
                       "The graph is too large or too tangled to analyse.")
        except MemoryError:
            problem = ("Graph too complex",
                       "There are too many loops or paths to list. "
                       "Try a smaller graph.")
        finally:
            QApplication.restoreOverrideCursor()
        if problem:
            self._error(*problem)
            return

        self.result = result
        self._result_text = text
        self.results.setHtml(results_to_html(text))
        self.results.verticalScrollBar().setValue(0)
        self.copy_btn.setEnabled(True)
        self.save_btn.setEnabled(True)
        self._fill_highlight_choices(result)
        self.redraw()
        self._announce(result)

    def _fill_highlight_choices(self, result):
        self.hl_combo.blockSignals(True)
        self.hl_combo.clear()
        self.hl_combo.addItem("No highlight", None)
        for p in result["paths"]:
            self.hl_combo.addItem(
                f"{p['id']}   {ARROW.join(p['nodes'])}",
                {"edges": p["edges"], "color": COLORS["path"], "label": p["id"]})
        for lp in result["loops"]:
            ring = lp["nodes"] + lp["nodes"][:1]
            self.hl_combo.addItem(
                f"{lp['id']}   {ARROW.join(ring)}",
                {"edges": lp["edges"], "color": COLORS["loop"], "label": lp["id"]})
        self.hl_combo.setCurrentIndex(0)
        self.hl_combo.blockSignals(False)
        self.hl_combo.setEnabled(self.hl_combo.count() > 1)

    def _announce(self, result):
        n_paths, n_loops = len(result["paths"]), len(result["loops"])
        delta = result["delta"]["value"]
        summary = (f"{n_paths} forward path{'s' * (n_paths != 1)}, "
                   f"{n_loops} loop{'s' * (n_loops != 1)}")
        if not result["valid"]:
            self._show_banner(
                "T(s) is undefined: \u0394 = 0 (division by zero)", "error")
            self.set_status(f"Solved ({summary}) \u2013 but \u0394 = 0.", "error")
            QMessageBox.warning(self, "T(s) is undefined", result["message"])
        elif not result["paths"]:
            self._show_banner(
                "No forward path from input to output, so T(s) = 0", "warn")
            self.set_status(f"Solved ({summary}) \u2013 no forward path.", "warn")
        else:
            self._show_banner(
                f"T(s) = \u03a3MiΔi / Δ = {_paren(result['numerator'])} / "
                f"{_paren(delta)} = {result['T_text']}", "ok")
            self.set_status(f"Solved: {summary}, \u0394 = {delta}.", "ok")

    # ------------------------------------------------- copy / save --------
    @safe
    def copy_results(self):
        if not self._result_text:
            self._info("Nothing to copy", "Press Solve first.")
            return
        QApplication.clipboard().setText(self._result_text)
        self.set_status("Results copied to the clipboard.", "ok")

    @safe
    def save_results(self):
        if not self._result_text:
            self._info("Nothing to save", "Press Solve first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save results", "sfg_results.txt", "Text files (*.txt)")
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(self._result_text + "\n")
        except OSError as exc:
            self._error("Could not save the file", f"{exc}")
            return
        self.set_status(f"Results saved to {path}", "ok")


# =========================================================================== #
#  Stand-alone window
# =========================================================================== #
class SFGWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Signal Flow Graph \u2013 Mason\u2019s Gain Formula")
        self.setCentralWidget(SFGWidget())
        self.resize(1440, 880)
        self.setMinimumSize(1000, 640)


def install_error_handler():
    """Show any uncaught exception in a message box instead of silently dying."""
    def hook(exc_type, exc, tb):
        traceback.print_exception(exc_type, exc, tb)
        if QApplication.instance() is not None:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(None, "Unexpected error",
                                 f"Something went wrong:\n\n{exc}")
    sys.excepthook = hook


def main():
    install_error_handler()
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = SFGWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()