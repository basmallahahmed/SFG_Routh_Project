"""
routh.py  --  Part 2: Routh stability criterion (no GUI code in here).

Public API (same style as mason.py)
    parse_polynomial(text)      "s^5 + s^4 + ... = 0"  or  "1 1 10 72 152 240"  ->  [1, 1, 10, ...]
    analyze(poly, tol=1e-6)     poly = text or coefficient list (highest power first)  ->  result dict
    explain(result)             result dict  ->  step-by-step text (formulas, substitution, values)

Everything is computed with exact fractions, so a zero in the first column is
detected exactly (no float noise).  The numerical poles use numpy.roots and are
only used to double-check the Routh result.

Naming used in the table (textbook style)
    row s^n   : a0 a2 a4 ...         (a_i = coefficient of s^(n-i))
    row s^n-1 : a1 a3 a5 ...
    next rows : b1 b2 ..., c1 c2 ..., d1 d2 ...
    every new entry:   x = (p2[0]*p1[j+1] - p1[0]*p2[j+1]) / p2[0]
                       p1 = row two above, p2 = row just above
"""
import re
from fractions import Fraction

import numpy as np

MAX_DEGREE = 30
DEFAULT_TOL = 1e-6

_SUPERSCRIPTS = str.maketrans("\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079",
                              "0123456789")


# =========================================================================== #
#  Small helpers
# =========================================================================== #
def _frac(x):
    """Exact Fraction from an int / float / str (0.1 -> 1/10, not the float error)."""
    if isinstance(x, Fraction):
        return x
    return Fraction(str(x).strip())


def fmt(x):
    """Fraction -> short text: 6 -> '6', 1/3 -> '0.333333'."""
    x = _frac(x)
    if x.denominator == 1:
        return str(x.numerator)
    return f"{float(x):.6g}"


def fmt_p(x):
    """Same as fmt but negatives in parentheses (for substitutions)."""
    return f"({fmt(x)})" if _frac(x) < 0 else fmt(x)


def _num(x):
    """Fraction -> int when whole, else float (for the structured result)."""
    x = _frac(x)
    return x.numerator if x.denominator == 1 else float(x)


def sign_of(x):
    x = _frac(x)
    return 1 if x > 0 else (-1 if x < 0 else 0)


def sign_char(x):
    return {1: "+", -1: "-", 0: "0"}[sign_of(x)]


def equation_text(coeffs):
    """[1, 1, 10] -> 's^2 + s + 10 = 0'."""
    n = len(coeffs) - 1
    out = ""
    for i, c in enumerate(coeffs):
        c = _frac(c)
        power = n - i
        mag = abs(c)
        body = "" if (mag == 1 and power > 0) else fmt(mag)
        if power == 1:
            body += "s"
        elif power > 1:
            body += f"s^{power}"
        if i == 0:
            out += ("-" if c < 0 else "") + body
        else:
            out += (" - " if c < 0 else " + ") + body
    return out + " = 0"


# =========================================================================== #
#  1. Input parsing and validation
# =========================================================================== #
_TERM = re.compile(
    r"(?P<sign>[+-]?)"
    r"(?P<num>(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)?"
    r"\*?"
    r"(?P<s>s(?:\^(?P<pow>\d+))?)?")


def _normalise(text):
    text = (text or "")
    text = re.sub("[\u2070\u00b9\u00b2\u00b3\u2074-\u2079]+",       # s^5 written as s<superscript 5>
                  lambda m: "^" + m.group(0).translate(_SUPERSCRIPTS), text)
    text = (text.replace("\u2212", "-").replace("\u00d7", "*").replace("**", "^")
            .replace("\u00b7", "*").lower())
    return re.sub(r"\s+", "", text)


def _parse_list(text):
    parts = [p for p in re.split(r"[,;\s]+", text.strip("[]() \t")) if p]
    if not parts:
        raise ValueError("Please enter the characteristic equation.")
    try:
        return [Fraction(p.replace("\u2212", "-")) for p in parts]
    except (ValueError, ZeroDivisionError):
        raise ValueError("Could not read the coefficient list.\n"
                         "Use numbers separated by spaces or commas, "
                         "highest power first, e.g.  1 1 10 72 152 240")


def _parse_poly(text):
    t = _normalise(text)
    if "=" in t:
        lhs, rhs = t.split("=", 1)
        if rhs not in ("0", "0.0", "0.00"):
            raise ValueError("The right-hand side must be 0, "
                             "e.g.  s^3 + 6s^2 + 11s + 6 = 0")
        t = lhs
    if not t:
        raise ValueError("Please enter the characteristic equation.")
    coef = {}
    pos = 0
    while pos < len(t):
        m = _TERM.match(t, pos)
        if m is None or m.end() == pos or (m.group("num") is None and m.group("s") is None):
            raise ValueError(f"Could not read the equation near '{t[pos:pos + 8]}'.\n"
                             "Write it like  s^5 + s^4 + 10s^3 + 72s^2 + 152s + 240 = 0")
        if pos > 0 and not m.group("sign"):
            raise ValueError(f"Missing '+' or '-' before '{t[pos:pos + 8]}'.")
        value = Fraction(m.group("num")) if m.group("num") is not None else Fraction(1)
        if m.group("sign") == "-":
            value = -value
        power = (int(m.group("pow")) if m.group("pow") else 1) if m.group("s") else 0
        coef[power] = coef.get(power, Fraction(0)) + value
        pos = m.end()
    degree = max(coef)
    return [coef.get(p, Fraction(0)) for p in range(degree, -1, -1)]


def parse_polynomial(text):
    """Text (polynomial or coefficient list) -> list of Fractions, highest power first."""
    if re.search(r"[a-zA-Z]", text or ""):
        return _parse_poly(text)
    return _parse_list(text)


def validate(coeffs):
    """Check the assumptions of the assignment (same sign, every power present)."""
    n = len(coeffs) - 1
    if n < 1:
        raise ValueError("The polynomial must have degree 1 or higher.")
    if n > MAX_DEGREE:
        raise ValueError(f"Degree {n} is too large (maximum {MAX_DEGREE}).")
    if coeffs[0] == 0:
        raise ValueError("The leading coefficient (of the highest power) cannot be 0.")
    missing = [f"s^{n - i}" if n - i > 1 else ("s" if n - i == 1 else "the constant term")
               for i, c in enumerate(coeffs) if c == 0]
    if missing:
        raise ValueError("Every power from s^0 up to s^n must be present, "
                         f"but the coefficient of {', '.join(missing)} is 0.\n"
                         "(A missing power is a special case that this program "
                         "does not handle.)")
    if not (all(c > 0 for c in coeffs) or all(c < 0 for c in coeffs)):
        raise ValueError("All coefficients must have the same sign. "
                         "Mixed signs already prove the system is not stable.")


# =========================================================================== #
#  2. Routh table
# =========================================================================== #
def _row_name(r):
    """Letter used for the entries of row r (rows 0 and 1 use 'a')."""
    if r < 2:
        return "a"
    k = r - 2
    return chr(ord("b") + k) if k < 25 else f"r{r}_"


def _live_count(n, r):
    """How many entries of row r can be non-zero."""
    return (n - r + 2) // 2


def build_table(coeffs):
    """
    Returns (rows, steps, special)
        rows    list of dicts {"label","power","values","names","live"}  (values = Fractions)
        steps   one dict per computed entry (formula, substitution, value)
        special None, or a dict {"kind","row","text"} if the table cannot continue
    """
    n = len(coeffs) - 1
    width = _live_count(n, 0)
    rows, steps = [], []

    def add_row(r, values, names):
        rows.append({"label": f"s^{n - r}", "power": n - r, "values": values,
                     "names": names, "live": _live_count(n, r)})

    # rows 0 and 1 come straight from the coefficients
    for r in (0, 1):
        idx = list(range(r, n + 1, 2))
        vals = [coeffs[i] for i in idx] + [Fraction(0)] * (width - len(idx))
        add_row(r, vals, [f"a{i}" for i in idx] + ["0"] * (width - len(idx)))

    special = None
    for r in range(2, n + 1):
        p1, p2 = rows[r - 2], rows[r - 1]
        a, b = p1["values"][0], p2["values"][0]
        letter = _row_name(r)
        live = _live_count(n, r)

        def name_of(row, k):
            return row["names"][k] if k < width else "0"

        vals, names = [], []
        for j in range(width):
            if j >= live:
                vals.append(Fraction(0))
                names.append("0")
                continue
            c = p1["values"][j + 1] if j + 1 < width else Fraction(0)
            d = p2["values"][j + 1] if j + 1 < width else Fraction(0)
            value = (b * c - a * d) / b
            name = f"{letter}{j + 1}"
            symbolic = (f"({name_of(p2, 0)}\u00b7{name_of(p1, j + 1)} \u2212 "
                        f"{name_of(p1, 0)}\u00b7{name_of(p2, j + 1)}) / {name_of(p2, 0)}")
            subst = (f"({fmt_p(b)}\u00b7{fmt_p(c)} \u2212 {fmt_p(a)}\u00b7{fmt_p(d)})"
                     f" / {fmt_p(b)}")
            steps.append({"row": r, "label": f"s^{n - r}", "col": j + 1, "name": name,
                          "symbolic": symbolic, "subst": subst, "value": value,
                          "text": f"{name} = {symbolic} = {subst} = {fmt(value)}"})
            vals.append(value)
            names.append(name)
        add_row(r, vals, names)

        if all(v == 0 for v in vals[:live]):
            special = {"kind": "zero_row", "row": r,
                       "text": f"The row of s^{n - r} is all zeros (special case)."}
            break
        if vals[0] == 0 and r < n:
            special = {"kind": "zero_pivot", "row": r,
                       "text": f"The first entry of the row of s^{n - r} is 0 "
                               "(special case)."}
            break
    return rows, steps, special


# =========================================================================== #
#  3. Numerical poles
# =========================================================================== #
def numeric_poles(coeffs, tol=DEFAULT_TOL):
    """All roots with numpy and their region: 'RHP', 'LHP' or 'AXIS'."""
    roots = np.roots([float(c) for c in coeffs])
    roots = sorted(roots, key=lambda z: (-round(z.real, 9), -abs(z.imag)))
    poles = []
    for z in roots:
        kind = "RHP" if z.real > tol else ("AXIS" if abs(z.real) <= tol else "LHP")
        poles.append({"value": complex(z), "region": kind})
    return poles


# =========================================================================== #
#  4. Full analysis
# =========================================================================== #
def analyze(poly, tol=DEFAULT_TOL):
    """
    poly : text, or a list of coefficients (highest power first).
    Returns a dict; the keys other modules may want:
        valid, message, equation, coeffs, degree, rows, steps, first_column,
        sign_changes (= right-half-plane poles by Routh), stable,
        poles, rhp_numeric, lhp_numeric, axis_numeric, agree, tol
    Raises ValueError (with a user-friendly message) for bad input.
    """
    coeffs = parse_polynomial(poly) if isinstance(poly, str) else [_frac(c) for c in poly]
    validate(coeffs)
    n = len(coeffs) - 1
    rows, steps, special = build_table(coeffs)
    poles = numeric_poles(coeffs, tol)

    result = {
        "equation": equation_text(coeffs),
        "coeffs": [_num(c) for c in coeffs],
        "degree": n,
        "rows": rows,
        "steps": steps,
        "special": special,
        "poles": poles,
        "rhp_numeric": sum(p["region"] == "RHP" for p in poles),
        "lhp_numeric": sum(p["region"] == "LHP" for p in poles),
        "axis_numeric": sum(p["region"] == "AXIS" for p in poles),
        "tol": tol,
        "valid": special is None,
        "message": "" if special is None else special["text"],
        "first_column": [r["values"][0] for r in rows],
        "sign_changes": None, "changes_at": [], "stable": None, "agree": None,
    }
    if special is None:
        col = result["first_column"]
        changes = [i for i in range(1, len(col)) if sign_of(col[i]) != sign_of(col[i - 1])]
        result["changes_at"] = changes
        result["sign_changes"] = len(changes)
        result["stable"] = len(changes) == 0
        result["agree"] = (len(changes) == result["rhp_numeric"])
    return result


# =========================================================================== #
#  5. Text output
# =========================================================================== #
def table_text(result):
    """Aligned plain-text Routh table."""
    rows = result["rows"]
    cells = [[fmt(v) for v in r["values"][:r["live"]]] for r in rows]
    ncols = max(len(c) for c in cells)
    widths = [max([len(c[j]) for c in cells if j < len(c)] + [1]) for j in range(ncols)]
    lab = max(len(r["label"]) for r in rows)
    lines = []
    for r, c in zip(rows, cells):
        body = "   ".join(c[j].rjust(widths[j]) if j < len(c) else " " * widths[j]
                          for j in range(ncols))
        lines.append(f"{r['label'].ljust(lab)} |  {body}".rstrip())
    return "\n".join(lines)


def explain(result):
    """Step-by-step explanation (formulas, numerical substitution, final values)."""
    n = result["degree"]
    coeffs = result["coeffs"]
    rows = result["rows"]
    L = []

    L.append("=== 1. CHARACTERISTIC EQUATION ===")
    L.append(f"  {result['equation']}")
    L.append(f"  Degree n = {n}  ->  {n + 1} coefficients, the Routh table has {n + 1} rows "
             f"(s^{n} down to s^0).")
    L.append("  Coefficients: " + ", ".join(f"a{i} = {fmt(c)}" for i, c in enumerate(coeffs)))
    L.append("  Check: every power present and all coefficients have the same sign.")
    L.append("")

    L.append("=== 2. BUILDING THE ROUTH TABLE ===")
    L.append("  First two rows: copy the coefficients alternately.")
    for r in (0, 1):
        row = rows[r]
        L.append(f"    {row['label']} row:  " + ",  ".join(
            f"{nm} = {fmt(v)}" for nm, v in zip(row["names"][:row["live"]], row["values"])))
    L.append("  Every other entry is computed from the two rows above it:")
    L.append("    Formula: x = (p2[0]\u00b7p1[j+1] \u2212 p1[0]\u00b7p2[j+1]) / p2[0]")
    L.append("    p1 = row two above, p2 = row just above, p[0] = its first-column entry")
    last_row = None
    for s in result["steps"]:
        if s["row"] != last_row:
            L.append("")
            L.append(f"  Row {s['label']}:")
            last_row = s["row"]
        L.append(f"    {s['text']}")
    if result["special"]:
        L.append("")
        L.append(f"Note: {result['special']['text']}")
        L.append("Note: the table stops here - special cases are outside the "
                 "assignment (they need the epsilon method or an auxiliary polynomial).")
    L.append("")

    L.append("=== 3. ROUTH TABLE ===")
    for line in table_text(result).split("\n"):
        L.append("  " + line)
    L.append("")

    L.append("=== 4. ROUTH CRITERION ===")
    L.append("  Rule: stable  <=>  no sign change in the first column;")
    L.append("        number of sign changes = number of poles in the right half plane.")
    if result["valid"]:
        col = result["first_column"]
        L.append("  First column: " + ",  ".join(
            f"{r['label']}: {fmt(v)} ({sign_char(v)})" for r, v in zip(rows, col)))
        if result["changes_at"]:
            for i in result["changes_at"]:
                L.append(f"    sign change {rows[i - 1]['label']} ({sign_char(col[i - 1])})"
                         f" -> {rows[i]['label']} ({sign_char(col[i])})")
        else:
            L.append("    no sign change between consecutive entries")
        k = result["sign_changes"]
        L.append(f"  Number of sign changes = {k}")
        if result["stable"]:
            L.append("Result: STABLE - no sign change, 0 poles in the right half plane.")
        else:
            L.append(f"Result: UNSTABLE - {k} sign change{'s' * (k != 1)}, "
                     f"so {k} pole{'s' * (k != 1)} in the right half plane.")
    else:
        L.append("Note: stability cannot be decided here (special case).")
    L.append("")

    L.append("=== 5. NUMERICAL POLES (numpy.roots) ===")
    L.append(f"  A pole counts as right half plane when Re(p) > {result['tol']:g}")
    for i, p in enumerate(result["poles"], 1):
        z = p["value"]
        tag = {"RHP": "   <-- RIGHT HALF PLANE", "AXIS": "   (on the imaginary axis)",
               "LHP": ""}[p["region"]]
        L.append(f"    p{i} = {z.real:+.6f} {'+' if z.imag >= 0 else '-'} "
                 f"{abs(z.imag):.6f}j{tag}")
    L.append(f"  Right half plane: {result['rhp_numeric']},  "
             f"left half plane: {result['lhp_numeric']},  "
             f"imaginary axis: {result['axis_numeric']}")
    if result["valid"]:
        k, m = result["sign_changes"], result["rhp_numeric"]
        if result["agree"]:
            L.append(f"Result: VERIFIED - Routh sign changes ({k}) = numerical RHP poles ({m}).")
        else:
            L.append(f"Note: MISMATCH - Routh gives {k} but numpy finds {m} RHP poles "
                     "(a pole very near the imaginary axis or a repeated root; "
                     "try another tolerance).")
        if result["axis_numeric"]:
            L.append("Note: some poles lie within the tolerance of the imaginary axis.")
    return "\n".join(L)


if __name__ == "__main__":                       # quick demo:  python -m src.routh
    print(explain(analyze("s^5 + s^4 + 10s^3 + 72s^2 + 152s + 240 = 0")))
