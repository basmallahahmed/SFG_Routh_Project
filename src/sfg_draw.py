"""
sfg_draw.py  --  Signal flow graph drawing (matplotlib only, no GUI toolkit).

Main entry point
    draw_graph(ax, graph, source=None, sink=None, highlight=None)

  ax         a matplotlib Axes (it is cleared and redrawn every call)
  graph      a SignalFlowGraph (graph_model.py)
  source     name of the input node  (coloured green)  - optional
  sink       name of the output node (coloured amber)  - optional
  highlight  optional dict that emphasises one forward path or loop:
                 {"edges": [(u, v, gain), ...],   # e.g. paths[i]["edges"]
                  "color": "#F97316",             # optional
                  "label": "M1"}                  # optional, shown in the legend

Layout idea
    * Back edges (feedback) are found with a DFS; the remaining edges form a DAG.
    * Every node gets a column = its longest-path depth in that DAG, so the
      signal flows left -> right and the output node sits in the last column.
    * The longest input -> output path is drawn on a straight horizontal line,
      other nodes are stacked above / below it.
    * Edges between neighbouring columns are straight; edges that skip a column
      curve above the chain; feedback edges curve below it; a pair of opposite
      edges (a->b and b->a) bend to opposite sides; self-loops are small
      loops above their node.
"""
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.path import Path

COLORS = {
    "bg": "#0F1317",
    "edge": "#7E8B99",
    "node": "#1B2229",          # node fill
    "node_edge": "#4FD1C5",     # node ring
    "source": "#10302B",
    "source_edge": "#34D399",
    "sink": "#33250F",
    "sink_edge": "#F2A93B",
    "text": "#D6DEE6",
    "label_edge": "#2F3943",
    "muted": "#6B7785",
    "path": "#F2A93B",     # default highlight colour for a forward path
    "loop": "#FF5C7A",     # default highlight colour for a loop
}

# monospace, technical look for node names, gains and the legend
FONT = ["Consolas", "DejaVu Sans Mono", "Menlo", "monospace"]

X_GAP = 2.7      # distance between columns
Y_GAP = 1.7      # distance between rows
LOOP_H = 1.0     # height of a self-loop above its node


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def format_gain(g):
    """6.0 -> '6', 0.5 -> '0.5', -2 -> '-2' (used for edge labels)."""
    if isinstance(g, float):
        return str(int(g)) if g.is_integer() else f"{g:.4g}"
    return str(g)


def _radius(name):
    """Node radius grows a little for long names."""
    return min(0.55, max(0.27, 0.07 * len(str(name)) + 0.13))


def _font_size(name):
    n = len(str(name))
    return 12 if n <= 2 else 10 if n <= 4 else 8 if n <= 7 else 7


# --------------------------------------------------------------------------- #
# layout
# --------------------------------------------------------------------------- #
def _structure(graph, source):
    """Adjacency lists plus the set of back edges (found by DFS)."""
    nodes = list(graph.nodes)
    adj = {n: [d for d, _ in graph.out_edges(n)] for n in nodes}
    state = {n: 0 for n in nodes}          # 0 new, 1 on stack, 2 done
    back = set()

    def visit(u):
        state[u] = 1
        for v in adj[u]:
            if v not in state:
                continue
            if state[v] == 0:
                visit(v)
            elif state[v] == 1:
                back.add((u, v))
        state[u] = 2

    roots = ([source] if source in state else []) + nodes
    for r in roots:
        if state[r] == 0:
            visit(r)
    return adj, back


def _columns(nodes, adj, back):
    """Column of every node = longest-path depth using forward edges only."""
    preds = {n: [] for n in nodes}
    for u in nodes:
        for v in adj[u]:
            if u != v and (u, v) not in back:
                preds[v].append(u)
    col = {}

    def depth(n):
        if n in col:
            return col[n]
        col[n] = 0                           # guard against accidental cycles
        col[n] = 1 + max((depth(p) for p in preds[n]), default=-1)
        return col[n]

    for n in nodes:
        depth(n)
    return col


def _main_chain(adj, back, source, sink):
    """Longest forward path source -> sink (list of node names) or []."""
    if source is None or sink is None or source == sink:
        return []
    memo = {}

    def best(u):
        if u == sink:
            return [u]
        if u in memo:
            return memo[u]
        memo[u] = None
        res = None
        for v in adj.get(u, []):
            if v == u or (u, v) in back:
                continue
            sub = best(v)
            if sub and (res is None or len(sub) + 1 > len(res)):
                res = [u] + sub
        memo[u] = res
        return res

    return best(source) or []


def compute_layout(graph, source=None, sink=None):
    """Return ({node: (x, y)}, {node: column}, set_of_back_edges)."""
    nodes = list(graph.nodes)
    if not nodes:
        return {}, {}, set()
    adj, back = _structure(graph, source)
    col = _columns(nodes, adj, back)
    if sink in col:                          # output node goes to the last column
        col[sink] = max(col.values())
    chain = set(_main_chain(adj, back, source, sink))

    by_col = {}
    for n in nodes:
        by_col.setdefault(col[n], []).append(n)

    pos = {}
    for c, members in by_col.items():
        on_chain = [n for n in members if n in chain]
        others = [n for n in members if n not in chain]
        if on_chain:
            rows = {on_chain[0]: 0.0}
            others = on_chain[1:] + others
            for i, n in enumerate(others):           # +1, -1, +2, -2, ...
                rows[n] = float((i // 2) + 1) * (1 if i % 2 == 0 else -1)
        else:
            k = len(members)
            rows = {n: (k - 1) / 2.0 - i for i, n in enumerate(members)}
        for n, r in rows.items():
            pos[n] = (c * X_GAP, r * Y_GAP)
    return pos, col, back


# --------------------------------------------------------------------------- #
# routing of one edge
# --------------------------------------------------------------------------- #
def _rad(u, v, col, back, edges):
    """
    Curvature (matplotlib 'arc3' rad) for edge u -> v.
    For a left->right edge a negative rad bends UP, a positive rad bends DOWN;
    the same rad on a right->left edge bends the opposite way, so feedback
    edges automatically go below the chain.
    """
    if (v, u) in edges:                  # opposite edge exists: separate them
        return -0.30
    if (u, v) in back or col[v] <= col[u]:
        return -0.35                     # feedback: below the chain
    if col[v] - col[u] >= 2:
        return -0.28                     # skips a column: above the chain
    return 0.0                           # neighbouring columns: straight


# --------------------------------------------------------------------------- #
# drawing
# --------------------------------------------------------------------------- #
def _label(ax, x, y, text, color, edge_color, size=9.5):
    # clip_on=True: the label is clipped together with the edges when panning
    ax.text(x, y, text, ha="center", va="center", fontsize=size,
            fontweight="bold", color=color, zorder=7, clip_on=True, fontfamily=FONT,
            bbox=dict(boxstyle="round,pad=0.22", fc=COLORS["bg"], ec=edge_color,
                      lw=0.9, alpha=0.96))


def draw_graph(ax, graph, source=None, sink=None, highlight=None):
    """Draw `graph` on `ax`. Returns the {node: (x, y)} positions."""
    ax.clear()
    ax.set_facecolor(COLORS["bg"])
    ax.axis("off")

    if not graph.nodes:
        ax.text(0.5, 0.5, "Your signal flow graph will appear here\n"
                          "Add nodes and branches, or load an example.",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=12, color=COLORS["muted"], fontfamily=FONT)
        return {}

    pos, col, back = compute_layout(graph, source, sink)
    edges = {(u, v) for u, v, _ in graph.branches}
    radius = {n: _radius(n) for n in graph.nodes}

    hl_edges, hl_color, hl_label = set(), None, None
    if highlight and highlight.get("edges"):
        hl_edges = {(e[0], e[1]) for e in highlight["edges"]}
        hl_color = highlight.get("color") or COLORS["path"]
        hl_label = highlight.get("label")
    hl_nodes = {n for e in hl_edges for n in e}
    dim = bool(hl_edges)

    xs, ys = [], []
    for n, (x, y) in pos.items():
        r = radius[n] + 0.15
        xs += [x - r, x + r]
        ys += [y - r, y + r]

    # ---- nodes (drawn first so arrows can be clipped by their outline) ----
    circles = {}
    for n, (x, y) in pos.items():
        if n == source:
            fc, ec = COLORS["source"], COLORS["source_edge"]
        elif n == sink:
            fc, ec = COLORS["sink"], COLORS["sink_edge"]
        else:
            fc, ec = COLORS["node"], COLORS["node_edge"]
        lw = 2.2
        if n in hl_nodes:
            ec, lw = hl_color, 3.2
        c = Circle((x, y), radius[n], fc=fc, ec=ec, lw=lw, zorder=4)
        ax.add_patch(c)
        circles[n] = c
        ax.text(x, y, str(n), ha="center", va="center", color="white",
                fontsize=_font_size(n), fontweight="bold", zorder=5,
                clip_on=True, fontfamily=FONT)

    # ---- edges ----
    for u, v, g in graph.branches:
        highlighted = (u, v) in hl_edges
        color = hl_color if highlighted else COLORS["edge"]
        lw = 3.0 if highlighted else 1.8
        alpha = 0.35 if (dim and not highlighted) else 1.0
        text = format_gain(g)
        edge_col = hl_color if highlighted else COLORS["label_edge"]
        txt_col = COLORS["text"]

        if u == v:                                   # self-loop above the node
            x, y = pos[u]
            r = radius[u]
            sx, sy = 0.55 * r, 0.83 * r
            p0 = (x + sx, y + sy)
            p3 = (x - sx, y + sy)
            c1 = (x + 0.85, y + LOOP_H + 0.45)
            c2 = (x - 0.85, y + LOOP_H + 0.45)
            path = Path([p0, c1, c2, p3],
                        [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4])
            ax.add_patch(FancyArrowPatch(
                path=path, arrowstyle="-|>", mutation_scale=16, color=color,
                lw=lw, alpha=alpha, zorder=2, shrinkA=0, shrinkB=0))
            lx = x
            ly = y + 0.125 * 2 * sy + 0.75 * (LOOP_H + 0.45) + 0.2
            _label(ax, lx, ly, text, txt_col, edge_col)
            xs += [x - 0.9, x + 0.9]
            ys += [ly + 0.3]
            continue

        rad = _rad(u, v, col, back, edges)
        (x1, y1), (x2, y2) = pos[u], pos[v]
        ax.add_patch(FancyArrowPatch(
            (x1, y1), (x2, y2), connectionstyle=f"arc3,rad={rad}",
            arrowstyle="-|>", mutation_scale=16, color=color, lw=lw,
            alpha=alpha, patchA=circles[u], patchB=circles[v],
            shrinkA=0, shrinkB=1.5, zorder=2))
        # mid-point of the quadratic curve (control point of 'arc3')
        mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        dx, dy = x2 - x1, y2 - y1
        lx, ly = mx + 0.5 * rad * dy, my - 0.5 * rad * dx
        _label(ax, lx, ly, text, txt_col, edge_col)
        xs += [lx - 0.35, lx + 0.35]
        ys += [ly - 0.3, ly + 0.3]

    # ---- limits / aspect so circles stay circles ----
    pad = 0.5
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)
    # 'datalim' keeps the axes filling the WHOLE canvas (so panning never hits
    # an invisible box edge) and widens the view limits to keep circles round.
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_anchor("C")

    # ---- small legend ----
    def dot(fc, ec, label):
        return Line2D([0], [0], marker="o", ls="", mfc=fc, mec=ec, mew=1.5,
                      ms=9, label=label)

    handles = []
    if source in pos:
        handles.append(dot(COLORS["source"], COLORS["source_edge"], "input"))
    if sink in pos:
        handles.append(dot(COLORS["sink"], COLORS["sink_edge"], "output"))
    handles.append(dot(COLORS["node"], COLORS["node_edge"], "node"))
    if hl_edges:
        handles.append(Line2D([0], [0], color=hl_color, lw=3,
                              label=hl_label or "selected"))
    leg = ax.legend(handles=handles, loc="upper right", prop={"family": FONT, "size": 8},
                    frameon=True, framealpha=0.95, borderpad=0.6,
                    handletextpad=0.4, labelspacing=0.5)
    leg.get_frame().set_edgecolor(COLORS["label_edge"])
    leg.get_frame().set_facecolor(COLORS["bg"])
    for t in leg.get_texts():
        t.set_color(COLORS["text"])
    leg.set_zorder(10)
    return pos