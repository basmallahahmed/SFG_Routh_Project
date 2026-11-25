"""
tests/test_cases.py  --  hand-verified test cases for Part 1 (SFG) and Part 2 (Routh).

Run from the project root:
    python -m unittest discover -s tests -v          (no extra install needed)
    python -m pytest tests -v                        (if pytest is installed)

Print the full step-by-step output of one case :
    python tests/test_cases.py show A      (or B, C, D)

Every expected number below was obtained by hand with Mason's formula AND checked
independently by solving the node equations with SymPy (see test_sympy_reference).
"""
import os
import sys
import unittest
from fractions import Fraction


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.graph_model import SignalFlowGraph          # noqa: E402
from src import mason, routh                          # noqa: E402
from src.paths_loops import get_forward_paths, get_loops   # noqa: E402

# --------------------------------------------------------------------------- #
#  PART 1 - test graphs  (from, to, gain)
# --------------------------------------------------------------------------- #
GRAPH_A = dict(src="R", snk="C", branches=[
    ("R", "a", 1), ("a", "b", 2), ("b", "c", 3), ("c", "d", 2), ("d", "e", 1),
    ("e", "f", 2), ("f", "C", 1), ("a", "c", 4), ("c", "e", 3), ("b", "a", -1),
    ("d", "c", -2), ("f", "e", -1), ("e", "b", -1), ("R", "c", 2)])

GRAPH_B = dict(src="In", snk="Out", branches=[
    ("In", "a", 1), ("a", "b", 2), ("b", "c", 1), ("c", "d", 3), ("d", "e", 1),
    ("e", "f", 2), ("f", "Out", 1), ("a", "a", 0.5), ("c", "b", -1),
    ("d", "d", -2), ("f", "e", -0.5), ("In", "c", 4), ("b", "e", 3),
    ("f", "a", -1)])

# Expected values, solved by hand ------------------------------------------- #
EXPECTED = {
    "A": {
        "paths": {                        # node sequence -> gain Mi
            ("R", "a", "b", "c", "d", "e", "f", "C"): 24,
            ("R", "a", "b", "c", "e", "f", "C"): 36,
            ("R", "a", "c", "d", "e", "f", "C"): 16,
            ("R", "a", "c", "e", "f", "C"): 24,
            ("R", "c", "d", "e", "f", "C"): 8,
            ("R", "c", "e", "f", "C"): 12},
        "loops": {                        # set of nodes -> gain Li
            frozenset("ab"): -2, frozenset("acdeb"): 8, frozenset("aceb"): 12,
            frozenset("bcde"): -6, frozenset("bce"): -9,
            frozenset("cd"): -4, frozenset("ef"): -2},
        "groups": {                       # size -> {set of loop-node-sets: product}
            2: {frozenset([frozenset("ab"), frozenset("cd")]): 8,
                frozenset([frozenset("ab"), frozenset("ef")]): 4,
                frozenset([frozenset("cd"), frozenset("ef")]): 8},
            3: {frozenset([frozenset("ab"), frozenset("cd"),
                           frozenset("ef")]): -16}},
        "delta": 40,
        "delta_i": [1, 1, 1, 1, 3, 3],    # same order as the paths above
        "products": [24, 36, 16, 24, 24, 36],
        "numerator": 160,
        "T": Fraction(4, 1)},
    "B": {
        "paths": {
            ("In", "a", "b", "c", "d", "e", "f", "Out"): 12,
            ("In", "a", "b", "e", "f", "Out"): 12,
            ("In", "c", "d", "e", "f", "Out"): 24,
            ("In", "c", "b", "e", "f", "Out"): -24},
        "loops": {
            frozenset("abcdef"): -12, frozenset("abef"): -12,
            frozenset("a"): 0.5, frozenset("bc"): -1,
            frozenset("d"): -2, frozenset("ef"): -1},
        "groups": {
            2: {frozenset([frozenset("abef"), frozenset("d")]): 24,
                frozenset([frozenset("a"), frozenset("bc")]): -0.5,
                frozenset([frozenset("a"), frozenset("d")]): -1,
                frozenset([frozenset("a"), frozenset("ef")]): -0.5,
                frozenset([frozenset("bc"), frozenset("d")]): 2,
                frozenset([frozenset("bc"), frozenset("ef")]): 1,
                frozenset([frozenset("d"), frozenset("ef")]): 2},
            3: {frozenset([frozenset("a"), frozenset("bc"), frozenset("d")]): 1,
                frozenset([frozenset("a"), frozenset("bc"), frozenset("ef")]): 0.5,
                frozenset([frozenset("a"), frozenset("d"), frozenset("ef")]): 1,
                frozenset([frozenset("bc"), frozenset("d"), frozenset("ef")]): -2},
            4: {frozenset([frozenset("a"), frozenset("bc"), frozenset("d"),
                           frozenset("ef")]): -1}},
        "delta": 54,
        "delta_i": [1, 3, 0.5, 1.5],
        "products": [12, 36, 12, -36],
        "numerator": 24,
        "T": Fraction(4, 9)},
}

# --------------------------------------------------------------------------- #
#  PART 2 - test polynomials (coefficients, highest power first)
# --------------------------------------------------------------------------- #
POLY_C = [1, 12, 77, 272, 571, 740, 375]          # stable, 6th order
POLY_D = [1, 6, 29, 78, 182, 272, 464, 320]       # unstable, 7th order, 2 RHP poles

FIRST_COL_C = [Fraction(x) for x in ("1", "12", "163/3", "26000/163",
                                     "7137/25", "81920/183", "375")]
FIRST_COL_D = [Fraction(x) for x in ("1", "6", "16", "107/4", "21214/321",
                                     "310240/10607", "-976820/1939", "320")]
POLES_C = [-1 + 2j, -1 - 2j, -1, -3 + 4j, -3 - 4j, -3]      # (s+1)(s+3)(s^2+2s+5)(s^2+6s+25)
POLES_D = [0.5 + 1.936491673j, 0.5 - 1.936491673j, -1 + 3j, -1 - 3j, -1,
           -2 + 2j, -2 - 2j]                                  # (s^2-s+4)(s+1)(s^2+2s+10)(s^2+4s+8)


def build(spec):
    g = SignalFlowGraph()
    for s, d, w in spec["branches"]:
        g.add_branch(s, d, w)
    return g


def solve(key):
    spec = GRAPH_A if key == "A" else GRAPH_B
    g = build(spec)
    return mason.analyze(g, spec["src"], spec["snk"])


def same_poles(found, expected):
    """True if both lists hold the same complex numbers (any order)."""
    left = list(expected)
    for p in found:
        match = next((q for q in left if abs(p - q) < 1e-4), None)
        if match is None:
            return False
        left.remove(match)
    return not left


# =========================================================================== #
#  PART 1 tests
# =========================================================================== #
class SFGCase:
    key = None

    @classmethod
    def setUpClass(cls):
        cls.exp = EXPECTED[cls.key]
        cls.res = solve(cls.key)

    def test_forward_paths(self):
        got = {tuple(p["nodes"]): p["gain"] for p in self.res["paths"]}
        self.assertEqual(set(got), set(self.exp["paths"]))
        for nodes, gain in self.exp["paths"].items():
            self.assertAlmostEqual(got[nodes], gain, msg=f"path {nodes}")

    def test_individual_loops(self):
        got = {frozenset(l["nodes"]): l["gain"] for l in self.res["loops"]}
        self.assertEqual(len(self.res["loops"]), len(self.exp["loops"]),
                         "a loop was found twice or missed")
        self.assertEqual(set(got), set(self.exp["loops"]))
        for nodes, gain in self.exp["loops"].items():
            self.assertAlmostEqual(got[nodes], gain, msg=f"loop {sorted(nodes)}")

    def test_non_touching_groups(self):
        got = {k: {frozenset(frozenset(l["nodes"]) for l in self._members(g)): g["gain"]
                   for g in grp}
               for k, grp in self.res["groups"].items()}
        self.assertEqual(set(got), set(self.exp["groups"]), "wrong group sizes")
        for k, expected in self.exp["groups"].items():
            self.assertEqual(set(got[k]), set(expected), f"groups of size {k}")
            for key, val in expected.items():
                self.assertAlmostEqual(got[k][key], val)

    def _members(self, group):
        by_id = {l["id"]: l for l in self.res["loops"]}
        return [by_id[i] for i in group["ids"]]

    def test_delta(self):
        self.assertAlmostEqual(self.res["delta"]["value"], self.exp["delta"])

    def test_delta_i(self):
        order = [tuple(p["nodes"]) for p in self.res["paths"]]
        expected_order = list(self.exp["paths"])
        by_path = dict(zip(expected_order, self.exp["delta_i"]))
        for nodes, d in zip(order, self.res["delta_i"]):
            self.assertAlmostEqual(d["value"], by_path[nodes], msg=f"delta_i of {nodes}")

    def test_products_and_numerator(self):
        by_path = dict(zip(self.exp["paths"], self.exp["products"]))
        for p, pr in zip(self.res["paths"], self.res["products"]):
            self.assertAlmostEqual(pr["value"], by_path[tuple(p["nodes"])])
        self.assertAlmostEqual(self.res["numerator"], self.exp["numerator"])

    def test_transfer_function(self):
        self.assertTrue(self.res["valid"])
        self.assertEqual(self.res["T_fraction"], self.exp["T"])

    def test_explanation_text_has_every_section(self):
        text = mason.explain(self.res)
        for header in ("Forward paths", "Individual loops", "Non-touching",
                       "Graph determinant", "Path cofactors", "Mason's gain formula"):
            self.assertIn(header, text)

    def test_sympy_reference(self):
        """Independent check: solve the node equations, no Mason involved."""
        try:
            import sympy as sp
        except ImportError:
            self.skipTest("sympy not installed")
        spec = GRAPH_A if self.key == "A" else GRAPH_B
        nodes = []
        for s, d, _ in spec["branches"]:
            for n in (s, d):
                if n not in nodes:
                    nodes.append(n)
        X = {n: sp.Symbol(n) for n in nodes}
        eqs = [sp.Eq(X[n], 1) if n == spec["src"] else
               sp.Eq(X[n], sum(sp.Rational(str(g)) * X[s]
                               for s, d, g in spec["branches"] if d == n))
               for n in nodes]
        sol = sp.solve(eqs, list(X.values()), dict=True)[0]
        self.assertEqual(sp.nsimplify(sol[X[spec["snk"]]]),
                         sp.Rational(self.exp["T"].numerator, self.exp["T"].denominator))


class TestSFG_A(SFGCase, unittest.TestCase):
    """Case A: 6 forward paths, 7 loops, 3 pairs + 1 triple of non-touching loops."""
    key = "A"


class TestSFG_B(SFGCase, unittest.TestCase):
    """Case B: 4 paths, 6 loops (2 self-loops, fractions), pairs, triples and a 4-group."""
    key = "B"


class TestSFGEdgeCases(unittest.TestCase):
    def test_delta_zero_is_handled(self):
        g = SignalFlowGraph()
        for s, d, w in [("x1", "x2", 1), ("x2", "x3", 1), ("x2", "x2", 1)]:
            g.add_branch(s, d, w)
        r = mason.analyze(g, "x1", "x3")
        self.assertEqual(r["delta"]["value"], 0)
        self.assertFalse(r["valid"])
        self.assertIn("undefined", r["message"])

    def test_duplicate_branch_rejected(self):
        g = SignalFlowGraph()
        g.add_branch("a", "b", 1)
        with self.assertRaises(ValueError):
            g.add_branch("a", "b", 2)

    def test_non_numeric_gain_rejected(self):
        g = SignalFlowGraph()
        for bad in ("abc", "", None, "nan", "inf"):
            with self.assertRaises(ValueError, msg=repr(bad)):
                g.add_branch("a", "b", bad)

    def test_empty_node_name_rejected(self):
        with self.assertRaises(ValueError):
            SignalFlowGraph().add_branch("", "b", 1)

    def test_input_equals_output_rejected(self):
        g = build(GRAPH_A)
        with self.assertRaises(ValueError):
            get_forward_paths(g, "R", "R")

    def test_no_forward_path(self):
        g = SignalFlowGraph()
        g.add_branch("a", "b", 1)
        g.add_branch("c", "d", 1)
        r = mason.analyze(g, "a", "d")
        self.assertEqual(r["paths"], [])
        self.assertEqual(r["T"], 0)

    def test_no_loops_graph(self):
        g = SignalFlowGraph()
        for s, d, w in [("R", "a", 2), ("a", "C", 3)]:
            g.add_branch(s, d, w)
        r = mason.analyze(g, "R", "C")
        self.assertEqual(get_loops(g), [])
        self.assertEqual(r["delta"]["value"], 1)
        self.assertEqual(r["T_fraction"], 6)


# =========================================================================== #
#  PART 2 tests
# =========================================================================== #
class TestRouthStable(unittest.TestCase):
    """Case C: s^6 + 12s^5 + 77s^4 + 272s^3 + 571s^2 + 740s + 375 = 0"""
    @classmethod
    def setUpClass(cls):
        cls.res = routh.analyze(POLY_C)

    def test_first_column(self):
        self.assertEqual(self.res["first_column"], FIRST_COL_C)

    def test_table_shape(self):
        self.assertEqual(len(self.res["rows"]), 7)       # s^6 ... s^0

    def test_specific_entries(self):
        steps = {s["name"]: s["value"] for s in self.res["steps"]}
        self.assertEqual(steps["b1"], Fraction(163, 3))          # (12*77-1*272)/12
        self.assertEqual(steps["b2"], Fraction(1528, 3))         # (12*571-1*740)/12
        self.assertEqual(steps["b3"], Fraction(375))             # (12*375-1*0)/12

    def test_verdict(self):
        self.assertTrue(self.res["valid"])
        self.assertTrue(self.res["stable"])
        self.assertEqual(self.res["sign_changes"], 0)

    def test_numeric_poles_agree(self):
        self.assertEqual(self.res["rhp_numeric"], 0)
        self.assertTrue(self.res["agree"])
        self.assertTrue(same_poles([p["value"] for p in self.res["poles"]], POLES_C))


class TestRouthUnstable(unittest.TestCase):
    """Case D: s^7 + 6s^6 + 29s^5 + 78s^4 + 182s^3 + 272s^2 + 464s + 320 = 0"""
    @classmethod
    def setUpClass(cls):
        cls.res = routh.analyze(POLY_D)

    def test_first_column(self):
        self.assertEqual(self.res["first_column"], FIRST_COL_D)

    def test_sign_changes_location(self):
        # + (s^2) -> - (s^1) -> + (s^0): rows 6 and 7 of the table
        self.assertEqual(self.res["changes_at"], [6, 7])
        self.assertEqual(self.res["sign_changes"], 2)

    def test_verdict(self):
        self.assertTrue(self.res["valid"])
        self.assertFalse(self.res["stable"])

    def test_numeric_poles_agree(self):
        self.assertEqual(self.res["rhp_numeric"], 2)
        self.assertEqual(self.res["lhp_numeric"], 5)
        self.assertTrue(self.res["agree"])
        self.assertTrue(same_poles([p["value"] for p in self.res["poles"]], POLES_D))

    def test_text_input_gives_same_result(self):
        text = "s^7 + 6s^6 + 29s^5 + 78s^4 + 182s^3 + 272s^2 + 464s + 320 = 0"
        self.assertEqual(routh.analyze(text)["first_column"], FIRST_COL_D)

    def test_explanation_mentions_everything(self):
        text = routh.explain(self.res)
        for part in ("CHARACTERISTIC EQUATION", "BUILDING THE ROUTH TABLE",
                     "ROUTH TABLE", "ROUTH CRITERION", "NUMERICAL POLES",
                     "UNSTABLE", "VERIFIED"):
            self.assertIn(part, text)


class TestRouthEdgeCases(unittest.TestCase):
    def test_missing_power(self):
        with self.assertRaises(ValueError):
            routh.analyze("s^4 + 10s^3 + 50s + 24 = 0")

    def test_mixed_signs(self):
        with self.assertRaises(ValueError):
            routh.analyze("s^3 + 2s^2 - 3s + 4 = 0")

    def test_garbage_text(self):
        for bad in ("hello", "", "s^2 + + 3", "s^2 + 3s + 2 = 5"):
            with self.assertRaises(ValueError, msg=repr(bad)):
                routh.analyze(bad)

    def test_coefficient_list_format(self):
        a = routh.analyze([1, 10, 35, 50, 24])
        b = routh.analyze("1 10 35 50 24")
        c = routh.analyze("s^4 + 10s^3 + 35s^2 + 50s + 24 = 0")
        self.assertEqual(a["first_column"], b["first_column"])
        self.assertEqual(a["first_column"], c["first_column"])

    def test_unicode_superscripts(self):
        r = routh.analyze("s\u00b3 + 6s\u00b2 + 11s + 6 = 0")
        self.assertTrue(r["stable"])


# =========================================================================== #
#  Show mode:  python tests/test_cases.py show A
# =========================================================================== #
def _show(key):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if key in ("A", "B"):
        print(mason.explain(solve(key)))
    elif key in ("C", "D"):
        print(routh.explain(routh.analyze(POLY_C if key == "C" else POLY_D)))
    else:
        print("Use one of: A B C D")


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "show":
        _show(sys.argv[2].upper())
    else:
        unittest.main(verbosity=2)
