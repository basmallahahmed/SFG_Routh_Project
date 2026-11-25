"""
gui_routh.py  --  Part 2 GUI (Routh stability criterion), PyQt5.

Window layout
    left  : input panel  (characteristic equation, examples, tolerance, Solve)
    right : the Routh table (first column coloured by sign, sign changes marked),
            the poles in the s-plane (right half plane shaded),
            and the scrollable step-by-step solution text.

The main widget is `RouthWidget`, a plain QWidget (same idea as SFGWidget), so
main.py puts it in a tab:      tabs.addTab(RouthWidget(), "Part 2: Routh")
`RouthWindow` / `main()` run it on its own:     python -m src.gui_routh

It talks to the algorithm only through routh.analyze() / routh.explain().
The colours, fonts and style sheet are shared with gui_sfg so both tabs match.
"""
import html
import re
import sys

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QBrush, QColor, QFont, QKeySequence
from PyQt5.QtWidgets import (
    QApplication, QComboBox, QFileDialog, QFrame, QHBoxLayout, QHeaderView,
    QLabel, QLineEdit, QMainWindow, QMessageBox, QPushButton, QShortcut,
    QSizePolicy, QSplitter, QTableWidget, QTableWidgetItem, QTextEdit,
    QVBoxLayout, QWidget, QAbstractItemView)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

try:                                   # normal use:  python -m src.gui_routh
    from .routh import (DEFAULT_TOL, analyze, explain, fmt, parse_polynomial,
                        sign_char, sign_of, validate)
    from .gui_sfg import (BANNER_COLORS, STATUS_COLORS, STYLE, _dark_palette,
                          install_error_handler, safe)
    from .sfg_draw import COLORS
except ImportError:                    # run directly: python src/gui_routh.py
    from routh import (DEFAULT_TOL, analyze, explain, fmt, parse_polynomial,
                       sign_char, sign_of, validate)
    from gui_sfg import (BANNER_COLORS, STATUS_COLORS, STYLE, _dark_palette,
                         install_error_handler, safe)
    from sfg_draw import COLORS

GOOD, BAD, WARN, MUTED = "#34D399", "#FF5C7A", "#F2A93B", "#6B7785"
_SUP = str.maketrans("0123456789", "\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079")

EXAMPLES = [
    {"name": "Assignment example \u2013 unstable (2 RHP poles)",
     "eq": "s^5 + s^4 + 10s^3 + 72s^2 + 152s + 240 = 0"},
    {"name": "Stable \u2013 3rd order",
     "eq": "s^3 + 6s^2 + 11s + 6 = 0"},
    {"name": "Stable \u2013 4th order",
     "eq": "s^4 + 10s^3 + 35s^2 + 50s + 24 = 0"},
    {"name": "Stable \u2013 5th order",
     "eq": "s^5 + 15s^4 + 85s^3 + 225s^2 + 274s + 120 = 0"},
    {"name": "Unstable \u2013 4th order (2 RHP poles)",
     "eq": "s^4 + 2s^3 + 3s^2 + 4s + 5 = 0"},
    {"name": "Unstable \u2013 6th order (2 RHP poles)",
     "eq": "s^6 + s^5 + 2s^4 + 3s^3 + 4s^2 + 5s + 6 = 0"},
]

EXTRA_STYLE = """
QLabel#hint { font-size: 9pt; }
QLabel#cardTitle { color: #D6DEE6; font-weight: 700; }
QFrame#card { background: transparent; border: none; }
"""


def sup(n):
    return str(n).translate(_SUP)


def row_label(power):
    return f"s{sup(power)}"


# =========================================================================== #
#  Text -> coloured HTML for the solution panel
# =========================================================================== #
_STEP = re.compile(r"^(\s+)([a-z]\d*_?\d* = .*)$")


def routh_to_html(text):
    out = []
    for line in text.split("\n"):
        if not line.strip():
            continue
        if line.startswith("==="):
            title = html.escape(line.strip("= ").strip())
            out.append('<p style="margin-top:16px; margin-bottom:2px;">'
                       f'<b><span style="color:{WARN}; font-size:12pt;">{title}'
                       '</span></b></p><hr/>')
            continue
        body = html.escape(line).replace(" ", "&nbsp;")
        s = line.strip()
        if s.startswith("Result: STABLE") or s.startswith("Result: VERIFIED"):
            body = f'<b><span style="color:{GOOD};">{body}</span></b>'
        elif s.startswith("Result: UNSTABLE"):
            body = f'<b><span style="color:{BAD};">{body}</span></b>'
        elif s.startswith("Note:"):
            body = f'<b><span style="color:#FF9F6E;">{body}</span></b>'
        elif s.startswith(("Formula:", "Rule:", "p1 =", "First two rows", "Every other",
                           "Check:", "Coefficients:", "A pole counts")):
            body = f'<span style="color:#8C99A6;">{body}</span>'
        elif "RIGHT HALF PLANE" in s:
            body = f'<b><span style="color:{BAD};">{body}</span></b>'
        elif _STEP.match(line) and s.count(" = ") >= 2:        # b1 = formula = subst = value
            head, value = line.rsplit(" = ", 1)
            body = (html.escape(head).replace(" ", "&nbsp;") + "&nbsp;=&nbsp;"
                    f'<b><span style="color:#4FD1C5;">{html.escape(value)}</span></b>')
        elif s.startswith("sign change"):
            body = f'<span style="color:{BAD};">{body}</span>'
        out.append(f'<div style="margin:1px 0;">{body}</div>')
    return "".join(out)


# =========================================================================== #
#  The Part 2 widget
# =========================================================================== #
class RouthWidget(QWidget):
    """Complete Part 2 screen. Embed it anywhere (tab, window, dialog)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("root")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setPalette(_dark_palette())
        self.setStyleSheet(STYLE + EXTRA_STYLE)
        self.result = None
        self._result_text = ""
        self._build_ui()
        self._wire_up()
        self._clear_results()
        self._update_hint()
        self.set_status("Ready \u2013 type a characteristic equation or load an example.")

    # ------------------------------------------------------------------ UI --
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
    def _hline():
        line = QFrame()
        line.setObjectName("hline")
        line.setFixedHeight(1)
        return line

    @staticmethod
    def _panel():
        card = QFrame()
        card.setObjectName("panel")
        lay = QVBoxLayout(card)
        lay.setContentsMargins(12, 10, 12, 10)
        lay.setSpacing(6)
        return card, lay

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

    def _build_header(self):
        bar = QFrame()
        bar.setObjectName("topbar")
        row = QHBoxLayout(bar)
        row.setContentsMargins(14, 6, 14, 6)
        row.setSpacing(12)
        mark = QLabel("s\u25b8")
        mark.setObjectName("mark")
        tag = QLabel("Routh stability criterion  \u00b7  right-half-plane poles")
        tag.setObjectName("muted")
        row.addWidget(mark)
        row.addWidget(tag)
        row.addStretch(1)
        return bar

    def _build_left(self):
        card, lay = self._panel()
        lay.setSpacing(8)

        lay.addWidget(self._caption("Characteristic equation"))
        self.eq_edit = QLineEdit()
        self.eq_edit.setPlaceholderText("s^5 + s^4 + 10s^3 + 72s^2 + 152s + 240 = 0")
        self.eq_edit.setToolTip("Polynomial in s, or just the coefficients "
                                "(highest power first): 1 1 10 72 152 240")
        self.eq_edit.setClearButtonEnabled(True)
        lay.addWidget(self.eq_edit)
        self.hint = QLabel("")
        self.hint.setObjectName("hint")
        self.hint.setWordWrap(True)
        lay.addWidget(self.hint)
        help_lbl = QLabel("Accepted:  s^3 + 6s^2 + 11s + 6 = 0   \u00b7   "
                          "6s\u00b2 + 5s + 1   \u00b7   a coefficient list "
                          "such as  1 6 11 6")
        help_lbl.setObjectName("muted")
        help_lbl.setWordWrap(True)
        lay.addWidget(help_lbl)

        lay.addSpacing(6)
        lay.addWidget(self._caption("Examples"))
        self.example_combo = QComboBox()
        for ex in EXAMPLES:
            self.example_combo.addItem(ex["name"], ex["eq"])
        self.example_combo.setSizeAdjustPolicy(
            QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.example_combo.setMinimumContentsLength(10)
        lay.addWidget(self.example_combo)
        self.load_btn = self._button("Load example", "primary")
        lay.addWidget(self.load_btn)

        lay.addSpacing(6)
        lay.addWidget(self._caption("Numerical check"))
        row = QHBoxLayout()
        row.addWidget(self._field("TOLERANCE"))
        self.tol_edit = QLineEdit(f"{DEFAULT_TOL:g}")
        self.tol_edit.setToolTip("A pole is counted as right-half-plane when "
                                 "Re(p) is greater than this value.")
        self.tol_edit.setMaximumWidth(110)
        row.addWidget(self.tol_edit)
        row.addStretch(1)
        lay.addLayout(row)
        tol_help = QLabel("Poles with Re(p) > tolerance count as right half plane "
                          "(absorbs numerical error of numpy.roots).")
        tol_help.setObjectName("muted")
        tol_help.setWordWrap(True)
        lay.addWidget(tol_help)

        lay.addStretch(1)
        self.solve_btn = self._button("Solve", "solve", "Ctrl+Enter")
        lay.addWidget(self.solve_btn)
        self.clear_btn = self._button("Clear", "danger")
        lay.addWidget(self.clear_btn)
        return card

    def _build_right(self):
        holder = QSplitter(Qt.Vertical)
        holder.setChildrenCollapsible(False)
        holder.setHandleWidth(8)

        top = QSplitter(Qt.Horizontal)
        top.setChildrenCollapsible(False)
        top.setHandleWidth(8)

        # -- Routh table --
        card, lay = self._panel()
        title = QLabel("ROUTH TABLE")
        title.setObjectName("captionPlain")
        lay.addWidget(title)
        lay.addWidget(self._hline())
        self.table = QTableWidget(0, 0)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionMode(QAbstractItemView.NoSelection)
        self.table.setAlternatingRowColors(True)
        self.table.setShowGrid(False)
        self.table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        lay.addWidget(self.table, 1)
        legend = QLabel("First column coloured by sign; rows where the sign "
                        "changes are marked.  Hover an entry to see how it was computed.")
        legend.setObjectName("muted")
        legend.setWordWrap(True)
        lay.addWidget(legend)
        top.addWidget(card)

        # -- pole plot --
        card, lay = self._panel()
        title = QLabel("POLES IN THE s-PLANE")
        title.setObjectName("captionPlain")
        lay.addWidget(title)
        lay.addWidget(self._hline())
        self.fig = Figure(facecolor=COLORS["bg"], constrained_layout=True)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.setMinimumSize(260, 220)
        lay.addWidget(self.canvas, 1)
        top.addWidget(card)
        top.setStretchFactor(0, 5)
        top.setStretchFactor(1, 4)
        top.setSizes([560, 440])
        holder.addWidget(top)

        # -- solution text --
        card, lay = self._panel()
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
            "Press \u201cSolve\u201d to see the Routh table with every calculation "
            "step, the stability verdict, the number of right-half-plane poles "
            "and the numerical pole check.")
        lay.addWidget(self.results, 1)
        holder.addWidget(card)
        holder.setStretchFactor(0, 5)
        holder.setStretchFactor(1, 5)
        holder.setSizes([380, 440])
        return holder

    def _wire_up(self):
        self.solve_btn.clicked.connect(lambda *_: self.solve())
        self.eq_edit.returnPressed.connect(lambda: self.solve())
        self.eq_edit.textChanged.connect(lambda *_: self._update_hint())
        self.load_btn.clicked.connect(lambda *_: self.load_example())
        self.clear_btn.clicked.connect(lambda *_: self.clear_all())
        self.copy_btn.clicked.connect(lambda *_: self.copy_results())
        self.save_btn.clicked.connect(lambda *_: self.save_results())
        for keys in ("Ctrl+Return", "Ctrl+Enter"):
            sc = QShortcut(QKeySequence(keys), self)
            sc.setContext(Qt.WidgetWithChildrenShortcut)
            sc.activated.connect(lambda: self.solve())

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

    # ------------------------------------------------------- input --------
    def _read_tol(self):
        raw = self.tol_edit.text().strip().replace(",", ".")
        try:
            tol = float(raw)
        except ValueError:
            raise ValueError(f"The tolerance '{raw}' is not a number. Use e.g. 1e-6.")
        if not (0 < tol < 1):
            raise ValueError("The tolerance must be a small positive number "
                             "(for example 1e-6).")
        return tol

    def _update_hint(self):
        text = self.eq_edit.text().strip()
        if not text:
            self.hint.setText("")
            return
        try:
            coeffs = parse_polynomial(text)
            validate(coeffs)
        except ValueError as exc:
            first = str(exc).split("\n")[0]
            self.hint.setStyleSheet(f"color: {WARN};")
            self.hint.setText(first)
            return
        self.hint.setStyleSheet(f"color: {GOOD};")
        self.hint.setText(f"\u2713 degree {len(coeffs) - 1}, coefficients: "
                          + ", ".join(fmt(c) for c in coeffs))

    # ------------------------------------------------------- actions ------
    def load_example(self):
     self.eq_edit.setText(self.example_combo.currentData())
     
     self.set_status("Example loaded \u2013 press Solve.")

    def clear_all(self):
        self.eq_edit.clear()
        self._clear_results()
        self.set_status("Cleared.", "ok")

    def _clear_results(self):
        self.result = None
        self._result_text = ""
        self.results.clear()
        self.banner.hide()
        self.table.clear()
        self.table.setRowCount(0)
        self.table.setColumnCount(0)
        self.copy_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        self._draw_poles(None)

    @safe
    def solve(self):
        text = self.eq_edit.text().strip()
        if not text:
            self._info("Nothing to solve",
                       "Type a characteristic equation first (or load an example).")
            return
        try:
            tol = self._read_tol()
            result = analyze(text, tol)
            explanation = explain(result)
        except ValueError as exc:
            self._clear_results()
            self._error("Cannot solve this equation", str(exc))
            return

        self.result = result
        self._result_text = explanation
        self._fill_table(result)
        self._draw_poles(result)
        self.results.setHtml(routh_to_html(explanation))
        self.results.verticalScrollBar().setValue(0)
        self.copy_btn.setEnabled(True)
        self.save_btn.setEnabled(True)
        self._announce(result)

    # ------------------------------------------------------- table --------
    def _fill_table(self, result):
        rows = result["rows"]
        n_rows = result["degree"] + 1          # unfinished tables still show all rows
        width = len(rows[0]["values"])
        t = self.table
        t.clear()
        t.setRowCount(n_rows)
        t.setColumnCount(width + 1)
        t.setHorizontalHeaderLabels([f"col {j + 1}" for j in range(width)] + ["1st-col sign"])
        t.setVerticalHeaderLabels([row_label(result["degree"] - r) for r in range(n_rows)])
        t.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        t.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)
        t.verticalHeader().setMinimumSectionSize(26)

        steps = {(s["row"], s["col"]): s["text"] for s in result["steps"]}
        changes = set(result["changes_at"])
        mono = QFont("Cascadia Mono")
        mono.setStyleHint(QFont.Monospace)

        for r in range(n_rows):
            if r >= len(rows):                 # never reached (special case)
                for c in range(width + 1):
                    it = QTableWidgetItem("\u2013")
                    it.setTextAlignment(Qt.AlignCenter)
                    it.setForeground(QBrush(QColor("#4D5864")))
                    t.setItem(r, c, it)
                continue
            row = rows[r]
            for c in range(width):
                live = c < row["live"]
                it = QTableWidgetItem(fmt(row["values"][c]) if live else "")
                it.setTextAlignment(Qt.AlignCenter)
                it.setFont(mono)
                if live and (r, c + 1) in steps:
                    it.setToolTip(steps[(r, c + 1)])
                elif live:
                    it.setToolTip(f"{row['names'][c]} = {fmt(row['values'][c])}  "
                                  "(copied from the coefficients)")
                if c == 0 and live:
                    sg = sign_of(row["values"][0])
                    it.setForeground(QBrush(QColor(GOOD if sg > 0 else BAD)))
                    f = QFont(mono)
                    f.setBold(True)
                    it.setFont(f)
                    if r in changes:
                        it.setBackground(QBrush(QColor("#3A1620")))
                t.setItem(r, c, it)
            # sign column
            if result["valid"] or r < len(rows):
                v = row["values"][0]
                mark = sign_char(v).replace("-", "\u2212")
                if r in changes:
                    mark += "   \u21af change"
                it = QTableWidgetItem(mark)
                it.setTextAlignment(Qt.AlignCenter)
                it.setForeground(QBrush(QColor(BAD if r in changes else
                                               (GOOD if sign_of(v) > 0 else BAD))))
                t.setItem(r, width, it)

    # ------------------------------------------------------- pole plot ----
    def _draw_poles(self, result):
        ax = self.ax
        ax.clear()
        ax.set_facecolor(COLORS["bg"])
        for sp in ax.spines.values():
            sp.set_color("#2F3943")
        ax.tick_params(colors=MUTED, labelsize=7)
        if result is None:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.text(0.5, 0.5, "Press Solve to plot the poles", ha="center",
                    va="center", color=MUTED, fontsize=10, transform=ax.transAxes)
            self.canvas.draw_idle()
            return
        poles = [p["value"] for p in result["poles"]]
        reach_x = max([abs(z.real) for z in poles] + [1.0]) * 1.3
        reach_y = max([abs(z.imag) for z in poles] + [1.0]) * 1.3
        ax.axvspan(0, reach_x, color=BAD, alpha=0.10, lw=0)
        ax.axhline(0, color="#4D5864", lw=0.8)
        ax.axvline(0, color="#7E8B99", lw=1.2)
        colours = {"LHP": COLORS["node_edge"], "RHP": BAD, "AXIS": WARN}
        for region, label in (("LHP", "left half plane"), ("AXIS", "on the axis"),
                              ("RHP", "right half plane")):
            pts = [p["value"] for p in result["poles"] if p["region"] == region]
            if pts:
                ax.scatter([z.real for z in pts], [z.imag for z in pts], marker="x",
                           s=55, linewidths=2, color=colours[region],
                           label=f"{label} ({len(pts)})", zorder=3)
        ax.set_xlim(-reach_x, reach_x)
        ax.set_ylim(-reach_y, reach_y)
        ax.set_xlabel(f"Re(s)     left: {result['lhp_numeric']}   "
                      f"axis: {result['axis_numeric']}   right: {result['rhp_numeric']}",
                      color=MUTED, fontsize=8)
        ax.set_ylabel("Im(s)", color=MUTED, fontsize=8)
        ax.text(0.98, 0.97, "RHP", transform=ax.transAxes, ha="right", va="top",
                color=BAD, alpha=0.7, fontsize=9, fontweight="bold")
        self.canvas.draw_idle()

    # ------------------------------------------------------- verdict ------
    def _announce(self, result):
        if not result["valid"]:
            self._show_banner(f"Special case: {result['message']} "
                              "Not covered by this program.", "warn")
            self.set_status("Table stopped at a special case.", "warn")
            return
        k, m = result["sign_changes"], result["rhp_numeric"]
        check = (f"  \u00b7  verified by numpy.roots ({m} RHP)" if result["agree"]
                 else f"  \u00b7  MISMATCH with numpy.roots ({m} RHP)")
        if result["stable"]:
            self._show_banner("STABLE \u2013 no sign change in the first column, "
                              "0 poles in the right half plane" + check,
                              "ok" if result["agree"] else "warn")
            self.set_status("Solved: stable system.", "ok")
        else:
            self._show_banner(f"UNSTABLE \u2013 {k} sign change{'s' * (k != 1)}, "
                              f"{k} pole{'s' * (k != 1)} in the right half plane"
                              + check, "error" if result["agree"] else "warn")
            self.set_status(f"Solved: unstable, {k} RHP pole{'s' * (k != 1)}.", "warn")

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
            self, "Save results", "routh_results.txt", "Text files (*.txt)")
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
class RouthWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Routh Stability Criterion")
        self.setCentralWidget(RouthWidget())
        self.resize(1440, 880)
        self.setMinimumSize(1000, 640)


def main():
    install_error_handler()
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = RouthWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
