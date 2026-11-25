"""
main.py  --  the application. It opens on a start screen where you choose the tool:

    01  Signal Flow Graph   (Mason's gain formula)   ->  gui_sfg.SFGWidget
    02  Routh Criterion     (stability)              ->  gui_routh.RouthWidget

Each tool opens in the same window (loaded the first time you pick it, then kept,
so you do not lose your work when you go back). "Menu" in the top-left corner,
or Ctrl+M, returns to the start screen.

Run from the project folder:
    python -m src.main          (or)      python src/main.py

Build the executable (see build_exe.bat / README):
    pyinstaller --noconfirm --windowed --name SFG_Routh src/main.py
"""
import sys
import logging
import traceback
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import (
    QApplication, QFrame, QHBoxLayout, QLabel, QMainWindow, QMessageBox,
    QPushButton, QShortcut, QStackedWidget, QVBoxLayout, QWidget)

try:                                   # normal use:  python -m src.main
    from .gui_sfg import _dark_palette, install_error_handler
except ImportError:                    # run directly / PyInstaller: python src/main.py
    from gui_sfg import _dark_palette, install_error_handler


# --------------------------------------------------------------------------- #
#  The tools. To add another one, write a factory and add an entry to TOOLS.
#  (The imports sit inside the factories so a problem in one part - for example
#  a missing library - never stops the other part from opening.)
# --------------------------------------------------------------------------- #
def _make_sfg():
    try:
        from .gui_sfg import SFGWidget
    except ImportError:
        from gui_sfg import SFGWidget
    return SFGWidget()


def _make_routh():
    try:
        from .gui_routh import RouthWidget
    except ImportError:
        from gui_routh import RouthWidget
    return RouthWidget()


TEAL, AMBER = "#4FD1C5", "#F2A93B"

TOOLS = [
    {"key": "1", "num": "01", "title": "Signal Flow Graph",
     "window": "Signal Flow Graph \u2013 Mason\u2019s Gain Formula",
     "desc": "Build a graph from nodes and branches, or load one from a file. "
             "Get every forward path, loop, \u0394, \u0394\u1d62 and the "
             "transfer function T(s), step by step.",
     "tags": "MASON  \u00b7  PATHS  \u00b7  LOOPS  \u00b7  T(s)",
     "accent": "teal", "color": TEAL, "make": _make_sfg},
    {"key": "2", "num": "02", "title": "Routh Criterion",
     "window": "Routh Stability Criterion",
     "desc": "Enter a characteristic equation to build the Routh table, count "
             "the sign changes and see how many poles lie in the right "
             "half-plane.",
     "tags": "ROUTH TABLE  \u00b7  STABILITY  \u00b7  POLES",
     "accent": "amber", "color": AMBER, "make": _make_routh},
]

HOME_STYLE = """
QWidget#home { background: #14181C; }
QLabel { background: transparent; color: #D6DEE6;
         font-family: "Bahnschrift", "Segoe UI", "Helvetica Neue", Arial, sans-serif; }
QLabel#homeMark { font-family: "Cascadia Mono", "Consolas", "Courier New", monospace;
                  font-size: 34pt; font-weight: 700; color: #F2A93B; }
QLabel#homeTitle { font-size: 24pt; font-weight: 700; color: #D6DEE6; }
QLabel#homeSub { font-size: 11pt; color: #6B7785; }
QLabel#homeHint { font-size: 9pt; color: #6B7785;
                  font-family: "Cascadia Mono", "Consolas", "Courier New", monospace; }

QPushButton#tile { background: #181D22; border: 1px solid #2B333B;
                   border-radius: 0; padding: 0; text-align: left; }
QPushButton#tile:hover, QPushButton#tile:focus { background: #1B2229; }
QPushButton#tile[accent="teal"]:hover, QPushButton#tile[accent="teal"]:focus
    { border: 1px solid #4FD1C5; }
QPushButton#tile[accent="amber"]:hover, QPushButton#tile[accent="amber"]:focus
    { border: 1px solid #F2A93B; }
QPushButton#tile:pressed { background: #14181C; }
QLabel#tileTitle { font-size: 17pt; font-weight: 700; color: #D6DEE6; }
QLabel#tileDesc { font-size: 10pt; color: #8C99A6; }
QLabel#tileTags { font-size: 8pt; font-weight: 700; color: #6B7785;
                  font-family: "Cascadia Mono", "Consolas", "Courier New", monospace; }
"""

BACK_STYLE = """
QFrame#backbar { background: #14181C; border: none;
                 border-bottom: 1px solid #2B333B; }
QPushButton#back { background: transparent; color: #8C99A6; border: none;
                   padding: 5px 12px; font-weight: 700; font-size: 9pt;
                   font-family: "Bahnschrift", "Segoe UI", Arial, sans-serif; }
QPushButton#back:hover { color: #14181C; background: #F2A93B; }
QLabel#backTitle { color: #6B7785; font-size: 9pt; background: transparent;
                   font-family: "Bahnschrift", "Segoe UI", Arial, sans-serif; }
"""


class Tile(QPushButton):
    """One big clickable choice on the start screen."""

    def __init__(self, tool, parent=None):
        super().__init__(parent)
        self.setObjectName("tile")
        self.setProperty("accent", tool["accent"])
        self.setCursor(Qt.PointingHandCursor)
        self.setMinimumSize(380, 250)
        self.setMaximumWidth(480)
        self.setToolTip(f"Open {tool['title']}  (press {tool['key']})")

        lay = QVBoxLayout(self)
        lay.setContentsMargins(26, 22, 26, 20)
        lay.setSpacing(8)

        num = QLabel(tool["num"])
        num.setStyleSheet(f"color: {tool['color']}; font-size: 12pt; "
                          "font-weight: 700; font-family: 'Cascadia Mono', "
                          "'Consolas', monospace;")
        title = QLabel(tool["title"])
        title.setObjectName("tileTitle")
        desc = QLabel(tool["desc"])
        desc.setObjectName("tileDesc")
        desc.setWordWrap(True)
        tags = QLabel(tool["tags"])
        tags.setObjectName("tileTags")
        go = QLabel("Open  \u25b8")
        go.setStyleSheet(f"color: {tool['color']}; font-size: 11pt; "
                         "font-weight: 700;")

        lay.addWidget(num)
        lay.addWidget(title)
        lay.addSpacing(4)
        lay.addWidget(desc)
        lay.addStretch(1)
        lay.addWidget(tags)
        lay.addWidget(go)
        for lab in (num, title, desc, tags, go):          # clicks reach the button
            lab.setAttribute(Qt.WA_TransparentForMouseEvents, True)


class HomeWidget(QWidget):
    """The start screen."""
    chosen = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("home")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet(HOME_STYLE)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(40, 30, 40, 24)
        lay.setSpacing(6)
        lay.addStretch(2)

        for text, name in (("SFG\u25b8", "homeMark"),
                           ("Control Systems Toolkit", "homeTitle"),
                           ("Choose a tool to start", "homeSub")):
            lab = QLabel(text)
            lab.setObjectName(name)
            lab.setAlignment(Qt.AlignCenter)
            lay.addWidget(lab)
        lay.addSpacing(30)

        row = QHBoxLayout()
        row.setSpacing(26)
        row.addStretch(1)
        self.tiles = []
        for i, tool in enumerate(TOOLS):
            tile = Tile(tool)
            tile.clicked.connect(lambda _=False, i=i: self.chosen.emit(i))
            row.addWidget(tile, 4)
            self.tiles.append(tile)
            sc = QShortcut(QKeySequence(tool["key"]), self)
            sc.setContext(Qt.WidgetWithChildrenShortcut)
            sc.activated.connect(lambda i=i: self.chosen.emit(i))
        row.addStretch(1)
        lay.addLayout(row)

        lay.addStretch(3)
        hint = QLabel("Press 1 or 2 to choose   \u00b7   "
                      "Ctrl+M brings you back to this menu")
        hint.setObjectName("homeHint")
        hint.setAlignment(Qt.AlignCenter)
        lay.addWidget(hint)

    def focus_first(self):
        if self.tiles:
            self.tiles[0].setFocus()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setPalette(_dark_palette())
        self.stack = QStackedWidget()
        self.home = HomeWidget()
        self.home.chosen.connect(self.open_tool)
        self.stack.addWidget(self.home)
        self.pages = {}                       # tool index -> page (kept alive)
        self.setCentralWidget(self.stack)

        sc = QShortcut(QKeySequence("Ctrl+M"), self)
        sc.setContext(Qt.WindowShortcut)
        sc.activated.connect(self.show_menu)

        self.resize(1440, 900)
        self.setMinimumSize(1000, 660)
        self.show_menu()

    def _wrap(self, widget, tool):
        """The tool plus a slim bar with the way back to the menu."""
        page = QWidget()
        page.setStyleSheet(BACK_STYLE)
        lay = QVBoxLayout(page)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        bar = QFrame()
        bar.setObjectName("backbar")
        row = QHBoxLayout(bar)
        row.setContentsMargins(6, 2, 14, 2)
        back = QPushButton("\u25c2  MENU")
        back.setObjectName("back")
        back.setCursor(Qt.PointingHandCursor)
        back.setToolTip("Back to the start screen (Ctrl+M). "
                        "Your work here is kept.")
        back.clicked.connect(self.show_menu)
        label = QLabel(f"{tool['num']}  \u00b7  {tool['title']}")
        label.setObjectName("backTitle")
        row.addWidget(back)
        row.addStretch(1)
        row.addWidget(label)
        lay.addWidget(bar)
        lay.addWidget(widget, 1)
        return page

    def open_tool(self, index):
        tool = TOOLS[index]
        if index not in self.pages:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            try:
                widget = tool["make"]()
            except Exception as exc:                       # noqa: BLE001
                traceback.print_exc()
                QApplication.restoreOverrideCursor()
                QMessageBox.critical(
                    self, "Could not open",
                    f"{tool['title']} failed to start:\n\n{exc}")
                return
            QApplication.restoreOverrideCursor()
            page = self._wrap(widget, tool)
            self.stack.addWidget(page)
            self.pages[index] = page
        self.stack.setCurrentWidget(self.pages[index])
        self.setWindowTitle(tool["window"])

    def show_menu(self):
        self.stack.setCurrentWidget(self.home)
        self.setWindowTitle("Control Systems \u2013 Signal Flow Graphs & Routh Stability")
        self.home.focus_first()


def main():
    install_error_handler()
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()