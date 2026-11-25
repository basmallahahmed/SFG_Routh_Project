import math


def _clean(x):
    """Remove float noise and turn 6.0 into 6."""
    x = round(x, 10)
    return int(x) if float(x).is_integer() else x


def _fmt(g):
    return f"({g})" if g < 0 else f"{g}"


def _make(label, index, nodes, gains, closed):
    total = _clean(math.prod(gains))
    name = f"{label}{index}"
    if len(gains) == 1:
        expr = f"{name} = {_fmt(gains[0])}"
    else:
        expr = f"{name} = " + " × ".join(_fmt(g) for g in gains) + f" = {total}"
    pairs = list(zip(nodes, nodes[1:] + nodes[:1])) if closed else list(zip(nodes, nodes[1:]))
    return {
        "id": name,
        "nodes": nodes,                  # in order of traversal
        "branches": gains,               # branch gains, in order
        "edges": [(u, v, g) for (u, v), g in zip(pairs, gains)],
        "gain": total,
        "expr": expr,
    }


def find_forward_paths(graph, source=None, sink=None):
    """All paths source -> sink that never visit a node twice."""
    source = source or graph.get_source()
    sink = sink or graph.get_sink()
    if source == sink:
        raise ValueError("Input and output nodes must be different.")

    found = []

    def dfs(node, nodes, gains):
        if node == sink:
            found.append((list(nodes), list(gains)))
            return
        for dst, g in graph.out_edges(node):
            if dst not in nodes:
                nodes.append(dst); gains.append(g)
                dfs(dst, nodes, gains)
                nodes.pop(); gains.pop()

    dfs(source, [source], [])
    return [_make("M", i, n, g, closed=False) for i, (n, g) in enumerate(found, 1)]


def find_loops(graph):
    """All individual loops (simple cycles, self-loops included), each exactly once.

    Each cycle is found only from its earliest node (in graph.nodes order),
    so no duplicates appear and no rotation check is needed.
    """
    order = {n: i for i, n in enumerate(graph.nodes)}
    found = []

    def search(start, node, nodes, gains):
        for dst, g in graph.out_edges(node):
            if dst == start:
                found.append((list(nodes), gains + [g]))
            elif order[dst] > order[start] and dst not in nodes:
                nodes.append(dst); gains.append(g)
                search(start, dst, nodes, gains)
                nodes.pop(); gains.pop()

    for start in graph.nodes:
        search(start, start, [start], [])

    return [_make("L", i, n, g, closed=True) for i, (n, g) in enumerate(found, 1)]


def get_forward_paths(graph, source=None, sink=None):
    return find_forward_paths(graph, source, sink)


def get_loops(graph):
    return find_loops(graph)