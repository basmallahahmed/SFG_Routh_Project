"""
mason.py  --  Mason's gain formula

Input : the forward paths and loops returned by paths_loops.py
        (get_forward_paths(graph), get_loops(graph)).
Output: structured results (dicts) + formatted step-by-step text.

Each path / loop is a dict likrr this
    {"id": "L1", "nodes": ["a", "b"], "branches": [2, -3], "gain": -6,
     "expr": "L1 = 2 × (-3) = -6", ...}
Two items touch if they share at least one node.

"""
import math
from fractions import Fraction

EPS = 1e-9

def _clean(x):
    #Remove float noise and turn 6.0 into 6
    x = round(x, 10)
    return int(x) if float(x).is_integer() else x


def _s(x):
    """Number as text; negatives in parentheses: -2 -> (-2)."""
    x = _clean(x)
    return f"({x})" if x < 0 else f"{x}"


def _sum_str(values):
    return " + ".join(_s(v) for v in values)


def _to_fraction(x):
    # fraction from a number (0.5 -> 1/2)
    return Fraction(str(_clean(x)))


def _index_of(item):
    """'M2' -> '2', 'L3' -> '3'."""
    return str(item["id"])[1:]

# 1. Touching and non-touching loops
def touches(a, b):
    return bool(set(a["nodes"]) & set(b["nodes"]))


def _make_group(members):
    ids = [m["id"] for m in members]
    gains = [m["gain"] for m in members]
    total = _clean(math.prod(gains))
    return {
        "ids": ids,
        "name": "".join(ids),                      # e.g. "L1L3"
        "gains": gains,
        "gain": total,
        "expr": f"{' · '.join(ids)} = {' × '.join(_s(g) for g in gains)} = {total}",
    }


def non_touching_groups(loops):
    """
    Every combination of 2, 3, ... mutually non-touching loops.

    Returns {2: [group, ...], 3: [group, ...], ...}; sizes with no group are
    absent. Groups of size k are built by extending groups of size k-1, so
    the search stops at the first size that has no group.
    """
    groups = {}
    current = [(i,) for i in range(len(loops))]
    k = 2
    while current:
        nxt = []
        for combo in current:
            for j in range(combo[-1] + 1, len(loops)):
                if all(not touches(loops[i], loops[j]) for i in combo):
                    nxt.append(combo + (j,))
        if not nxt:
            break
        groups[k] = [_make_group([loops[i] for i in c]) for c in nxt]
        current = nxt
        k += 1
    return groups

# 2. delta and deltai
def _formula_text(name, kmax):
    parts = [f"{name} = 1"]
    for k in range(1, kmax + 1):
        sign = "−" if k % 2 else "+"
        label = "ΣL" if k == 1 else f"Σ(products of {k} non-touching loops)"
        parts.append(f"{sign} {label}")
    return " ".join(parts)


def compute_delta(loops, name="Δ"):
    """
    Δ = 1 - ΣL + Σ(2 non-touching) - Σ(3 non-touching) + ...
    Works on whatever list of loops it is given, so Δi reuses it.
    """
    groups = non_touching_groups(loops)

   #  (k = number of loops in the product)
    levels = {}
    if loops:
        levels[1] = [(l["id"], l["gain"]) for l in loops]
    for k, grp in groups.items():
        levels[k] = [(g["name"], g["gain"]) for g in grp]

    value = 1
    terms, term_lines = {}, []
    symbolic, numeric, evaluated = f"{name} = 1", f"{name} = 1", f"{name} = 1"
    for k in sorted(levels):
        sign = -1 if k % 2 else 1
        sym = "−" if k % 2 else "+"
        labels = " + ".join(lbl for lbl, _ in levels[k])
        vals = _sum_str([g for _, g in levels[k]])
        terms[k] = _clean(sum(g for _, g in levels[k]))
        value += sign * terms[k]

        title = "ΣL" if k == 1 else f"Σ({k} non-touching)"
        term_lines.append(f"{title} = {vals} = {terms[k]}")
        symbolic += f" {sym} ({labels})"
        numeric += f" {sym} {vals}" if len(levels[k]) == 1 else f" {sym} ({vals})"
        evaluated += f" {sym} {_s(terms[k])}"
    value = _clean(value)

    return {
        "name": name,
        "value": value,
        "terms": terms,                 # {1: ΣL, 2: Σpairs, ...}
        "levels": levels,
        "groups": groups,               # only sizes >= 2
        "loops": loops,
        "formula": _formula_text(name, max(levels) if levels else 0),
        "term_lines": term_lines,
        "symbolic": symbolic,
        "numeric": numeric,
        "evaluated": evaluated,
        "expr": f"{name} = {value}" if not levels else f"{evaluated} = {value}",
    }


def compute_delta_i(path, loops):
    """Δi = Δ computed only from the loops that do NOT touch path i."""
    excluded = [l for l in loops if touches(path, l)]
    remaining = [l for l in loops if not touches(path, l)]
    d = compute_delta(remaining, name=f"Δ{_index_of(path)}")
    d["path_id"] = path["id"]
    d["excluded"] = excluded
    d["remaining"] = remaining
    return d


def solve_mason(paths, loops):
    """
    T = (Σ Mi·Δi) / Δ.  Returns a dict with everything the GUI/report needs:
      paths, loops, groups, delta, delta_i, products, numerator,
      numerator_expr, valid, message, T, T_fraction, T_text
    """
    delta = compute_delta(loops, "Δ")
    delta_i = [compute_delta_i(p, loops) for p in paths]

    products = []
    for p, d in zip(paths, delta_i):
        val = _clean(p["gain"] * d["value"])
        name = f"{p['id']}{d['name']}"             # "M1Δ1"
        products.append({
            "name": name,
            "value": val,
            "expr": f"{name} = {_s(p['gain'])} × {_s(d['value'])} = {val}",
        })

    numerator = _clean(sum(pr["value"] for pr in products))
    if products:
        names = " + ".join(pr["name"] for pr in products)
        if len(products) == 1:
            numerator_expr = f"ΣMiΔi = {names} = {numerator}"
        else:
            numerator_expr = (
                f"ΣMiΔi = {names} = "
                + _sum_str([pr["value"] for pr in products])
                + f" = {numerator}"
            )
    else:
        numerator_expr = "ΣMiΔi = 0 (no forward paths)"

    result = {
        "paths": paths,
        "loops": loops,
        "groups": delta["groups"],
        "delta": delta,
        "delta_i": delta_i,
        "products": products,
        "numerator": numerator,
        "numerator_expr": numerator_expr,
        "valid": True,
        "message": "",
        "T": None,
        "T_fraction": None,
        "T_text": "undefined",
    }

    if abs(delta["value"]) < EPS:
        result["valid"] = False
        result["message"] = (
            "Δ = 0, so T(s) = ΣMiΔi / Δ is undefined (division by zero). "
            "Check the gains: the graph has a degenerate loop structure."
        )
        return result

    frac = _to_fraction(numerator) / _to_fraction(delta["value"])
    result["T_fraction"] = frac
    result["T"] = float(frac)
    if frac.denominator == 1:
        result["T_text"] = str(frac.numerator)
    else:
        result["T_text"] = f"{frac.numerator}/{frac.denominator} ≈ {float(frac):.6g}"
    if not paths:
        result["message"] = "No forward paths, so T(s) = 0."
    return result


def analyze(graph, source=None, sink=None):
    """Convenience: find paths and loops from a SignalFlowGraph, then solve."""
    try:
        from .paths_loops import get_forward_paths, get_loops
    except ImportError:
        from paths_loops import get_forward_paths, get_loops
    paths = get_forward_paths(graph, source, sink)
    loops = get_loops(graph)
    return solve_mason(paths, loops)

def _delta_lines(d, indent="  "):
    lines = [f"{indent}Formula:      {d['formula']}"]
    for tl in d["term_lines"]:
        lines.append(f"{indent}  {tl}")
    lines.append(f"{indent}Symbolic:     {d['symbolic']}")
    lines.append(f"{indent}Substituting: {d['numeric']}")
    lines.append(f"{indent}Result:       {d['expr']}")
    return lines


def explain(result):
    out = []
    add = out.append

    add("=== Forward paths ===")
    if result["paths"]:
        for p in result["paths"]:
            add(f"{p['expr']}    [{' → '.join(p['nodes'])}]")
    else:
        add("No forward paths found.")

    add("")
    add("=== Individual loops ===")
    if result["loops"]:
        for l in result["loops"]:
            add(f"{l['expr']}    [{' → '.join(l['nodes'] + l['nodes'][:1])}]")
    else:
        add("No loops found.")

    add("")
    add("=== Non-touching loop combinations ===")
    if result["groups"]:
        for k, grp in sorted(result["groups"].items()):
            add(f"{k} non-touching loops:")
            for g in grp:
                add(f"  {g['expr']}")
    else:
        add("None (no two loops are non-touching).")

    add("")
    add("=== Graph determinant Δ ===")
    out.extend(_delta_lines(result["delta"]))

    add("")
    add("=== Path cofactors Δi (loops touching path i are excluded) ===")
    for p, d in zip(result["paths"], result["delta_i"]):
        add(f"{p['id']}  [{' → '.join(p['nodes'])}]")
        ex = ", ".join(l["id"] for l in d["excluded"]) or "none"
        rem = ", ".join(l["id"] for l in d["remaining"]) or "none"
        add(f"  Loops touching {p['id']} (excluded): {ex}")
        add(f"  Loops kept:                       {rem}")
        out.extend(_delta_lines(d))
    if not result["paths"]:
        add("No forward paths, so there is no Δi.")

    add("")
    add("=== Mason's gain formula ===")
    add("T = (M1Δ1 + M2Δ2 + ... + MmΔm) / Δ")
    for pr in result["products"]:
        add(f"  {pr['expr']}")
    add(f"  {result['numerator_expr']}")
    add(f"  Δ = {result['delta']['value']}")
    if result["valid"]:
        add(f"T = {result['numerator']} / {_s(result['delta']['value'])} = {result['T_text']}")
    else:
        add(f"T = {result['numerator']} / 0  →  undefined")
    if result["message"]:
        add(f"Note: {result['message']}")

    return "\n".join(out)

if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    try:
        from .graph_model import SignalFlowGraph
    except ImportError:
        from graph_model import SignalFlowGraph

    g = SignalFlowGraph()
    for s, d, w in [("R", "a", 1), ("a", "b", 2), ("b", "c", 3), ("c", "d", 4),
                    ("d", "C", 1), ("a", "c", 5), ("b", "a", -1), ("c", "b", -2),
                    ("d", "c", -3), ("d", "d", 0.5)]:
        g.add_branch(s, d, w)
    print(explain(analyze(g)))