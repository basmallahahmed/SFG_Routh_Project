import math


class SignalFlowGraph:
    """Directed graph with numeric branch gains."""

    def __init__(self):
        self.nodes = []      # node names, in insertion order
        self.branches = []   # (src, dst, gain)

    def add_node(self, name):
        name = str(name).strip()
        if not name:
            raise ValueError("Node name cannot be empty.")
        if name not in self.nodes:
            self.nodes.append(name)
        return name

    def add_branch(self, src, dst, gain):
        try:
            gain = float(gain)
        except (TypeError, ValueError):
            raise ValueError(f"Gain '{gain}' is not a number.")
        if not math.isfinite(gain):
            raise ValueError("Gain must be a finite number.")
        if gain.is_integer():
            gain = int(gain)

        src, dst = str(src).strip(), str(dst).strip()
        if not src or not dst:
            raise ValueError("Branch endpoints cannot be empty.")
        if any(s == src and d == dst for s, d, _ in self.branches):
            raise ValueError(f"Branch {src} -> {dst} already exists.")

        self.add_node(src)
        self.add_node(dst)
        self.branches.append((src, dst, gain))

    def remove_branch(self, src, dst):
        self.branches = [b for b in self.branches if (b[0], b[1]) != (src, dst)]

    def clear(self):
        self.nodes, self.branches = [], []

    def out_edges(self, node):
        return [(d, g) for s, d, g in self.branches if s == node]

    def get_source(self):
        """The input node: the only node with no incoming branch."""
        targets = {d for _, d, _ in self.branches}
        cands = [n for n in self.nodes if n not in targets]
        if len(cands) != 1:
            raise ValueError(
                f"Expected exactly one input node, found {len(cands)}: {cands}. "
                "Pass the source explicitly.")
        return cands[0]

    def get_sink(self):
        """The output node: the only node with no outgoing branch."""
        sources = {s for s, _, _ in self.branches}
        cands = [n for n in self.nodes if n not in sources]
        if len(cands) != 1:
            raise ValueError(
                f"Expected exactly one output node, found {len(cands)}: {cands}. "
                "Pass the sink explicitly.")
        return cands[0]