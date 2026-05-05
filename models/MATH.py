import re
import io
import os
import time
import cmath
import math
import builtins
import itertools
import fractions
import collections
import statistics as stats_stdlib
import threading
import contextlib
import numpy as np
from datetime import datetime
from typing import Optional
from transformers import GenerationConfig
from models.LLM import LLMModel, SYSTEM_PROMPTS

SYSTEM_PROMPTS["math"] = (
    "You are a math expert. Take a deep breath and reason step by step, "
    "then reply with ONLY the number of the correct option."
)

SYSTEM_PROMPTS["mathstral"] = (
    "You are a math competition expert. For each question:\n"
    "1. Identify the mathematical domain and the relevant framework or theorem.\n"
    "2. Show your reasoning step by step, being precise about definitions.\n"
    "3. Compute or derive the answer carefully.\n"
    "4. Reply with ONLY the option number on the last line."
)

SYSTEM_PROMPTS["mathstral_reason"] = (
    "You are a math competition expert. Reason step by step through the problem.\n"
    "DO NOT compute any numerical values yourself.\n"
    "Identify the correct mathematical framework, set up the equations, and simplify as far as possible analytically.\n"
    "End your response with EXACTLY one of:\n"
    "  COMPUTE: [the precise expression, equation, or system to evaluate numerically]\n"
    "  ANSWER: [option number]  (only when the answer follows from pure reasoning, no numbers needed)"
)

SYSTEM_PROMPTS["code"] = (
    "You are a math expert solving multiple choice questions.\n"
    "Default to writing code. Use a ```python``` block that prints the final result for ANY "
    "question with numbers: algebra, calculus, probability, statistics, combinatorics, "
    "geometry, complex numbers, or any theorem applied to a concrete value.\n"
    "IMPORTANT: do NOT write any import statements. The following are already available:\n"
    "  math, cmath, np (numpy), sp (sympy), stats (scipy.stats),\n"
    "  quad, dblquad, nquad (scipy.integrate), erf, erfinv, gamma (scipy.special),\n"
    "  norm, chi2, binom, poisson, t_dist, f_dist (scipy stats distributions),\n"
    "  solve, symbols, sqrt, Rational, integrate, diff, Matrix, pi, I, oo,\n"
    "  ln, log, log2, log10, exp, sin, cos, tan, asin, acos, atan, atan2,\n"
    "  sinh, cosh, tanh, hypot, degrees, radians, floor, ceil, factorial,\n"
    "  sqrt, prod, reduce, comb, gcd, lcm, perm, inf, pi, e,\n"
    "  factorint, isprime, nextprime, totient, mod_inverse, binomial (sympy),\n"
    "  N, Sum, Product, Abs, Piecewise,\n"
    "  itertools, Fraction, Counter, defaultdict, deque.\n"
    "Only reason in plain text for purely abstract questions with no numbers at all.\n"
    "Always end your response with ONLY the option number on the last line."
)


_CODE_QUESTION_KEYWORDS = [
    # generic computation triggers
    "compute", "calculate", "evaluate", "determine the value", "find the value",
    "what is the value", "what is the sum", "what is the product", "what is the ratio", "what is the probability",
    "find all", "find the", "how many", "total number of",
    # algebra
    "simplify", "expand", "factor", "solve for", "solve the equation",
    "system of equations", "quadratic", "polynomial", "roots of", "zeros of",
    "remainder", "rearranging", "inverse of", "determinant", "matrix",
    "eigenvalue", "eigenvector",
    # calculus
    "derivative", "differentiate", "integral", "integrate", "antiderivative",
    "find the limit", "limit of", "limit as", "converges", "diverges",
    "rate of change", "tangent line", "critical point", "inflection point",
    "maximum value", "minimum value", "optimize",
    # geometry
    "area", "volume", "surface area", "perimeter", "circumference",
    "distance", "length", "diagonal", "hypotenuse", "midpoint",
    "slope", "coordinate", "what is the area", "what is the volume",
    "what is the length",
    # trigonometry
    "sin(", "cos(", "tan(", "arcsin", "arccos", "arctan",
    "angle between", "find the angle",
    # statistics & probability
    "probability that", "probability of", "what is the probability",
    "expected value", "expected number", "expected",
    "standard deviation", "variance", "mean of", "median",
    "percentile", "confidence interval", "normal distribution",
    "z-score", "correlation coefficient", "regression",
    "binomial", "poisson distribution",
    # number theory & combinatorics
    "divisible", "prime factorization", "least common multiple",
    "greatest common divisor", "modulo", "congruent mod",
    "combinations", "permutations", "arrangements", "how many ways",
    "sum of", "rate of",
    # sequences & series
    "arithmetic sequence", "geometric sequence", "common difference",
    "common ratio", "nth term",
]

_TEXT_QUESTION_KEYWORDS = [
    "which of the following is true", "which statement", "must be true",
    "most likely", "best describes", "best interpretation", "interpretation of",
    "necessary for the validity", "assumption", "observational study",
    "placebo", "response variable", "explain why", "conclusion concerning",
    "irreducible over", "isomorphic to", "the external direct product",
    "homomorphism", "equivalence relation",
]


_STATEMENT_FEW_SHOT = """Example of how to answer a Statement 1 / Statement 2 question:

Question: Statement 1 | The identity element of a group is unique. Statement 2 | Every group of order 4 is abelian.
Options:
0: True, False
1: False, False
2: True, True
3: False, True

Reasoning:
- Statement 1: True. If e and e' are both identities then e = e·e' = e'.
- Statement 2: True. Groups of order 4 are either Z_4 or Z_2×Z_2, both abelian.
Answer: 2

Now answer the following question the same way:
"""


_NUMBER_WORDS = {
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
    "seventeen", "eighteen", "nineteen", "twenty", "thirty", "forty", "fifty",
    "sixty", "seventy", "eighty", "ninety", "hundred", "thousand", "million",
}

def _has_numeric_content(question: str) -> bool:
    if any(c.isdigit() for c in question):
        return True
    words = set(re.findall(r'\b\w+\b', question.lower()))
    return bool(words & _NUMBER_WORDS)


def _should_use_code(question: str) -> bool:
    """Keyword heuristic: True → use code executor, False → use plain text path."""
    if not _has_numeric_content(question):
        return False   # no numeric content → purely abstract, code cannot help
    q = question.lower()
    for kw in _TEXT_QUESTION_KEYWORDS:
        if kw in q:
            return False
    for kw in _CODE_QUESTION_KEYWORDS:
        if kw in q:
            return True
    return False   # default: text is safer


class MathLLMModel(LLMModel):
    """
    LLMModel subclass for math competitions.
    Optionally generates and executes Python code to solve computational problems
    (Agentic AI), logging each question and generated script to AgenticAI_scripts/.
    Set use_router=True to dynamically choose code vs text path per question.
    """

    def __init__(
        self,
        use_code_executor: bool = False,
        use_router: bool = False,
        log_dir: str = "AgenticAI_scripts",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._use_code_executor = use_code_executor
        self._use_router        = use_router

        # Replace the generic Retriever with MathRetriever for Wolfram MathWorld priority
        if self._retriever is not None:
            from utils.retrieval import MathRetriever
            self._retriever = MathRetriever()

        self._log_correct = 0
        self._log_total   = 0

        if use_code_executor or use_router:
            os.makedirs(log_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._log_path = os.path.join(log_dir, f"session_{timestamp}.txt")

    def get_info(self) -> dict:
        info = super().get_info()
        info["code_executor"] = self._use_code_executor
        return info

    # ── generation (overridden to adapt user message for code mode) ──────────

    def _generate(self, question_text: str, options: dict, context: str = "") -> str:
        options_str = "\n".join(f"{id}: {text}" for id, text in options.items())
        user_suffix = (
            "You MUST write a ```python``` code block that prints the answer, even when context is provided. "
            "Read constraints carefully — only enforce what the problem explicitly states. "
            "End with only the option number."
            if self._use_code_executor else
            "Reply with only the option number."
        )

        if re.search(r"statement\s*1", question_text, re.IGNORECASE):
            user_content = _STATEMENT_FEW_SHOT + f"Question: {question_text}\n\nOptions:\n{options_str}\n\n{user_suffix}"
        else:
            user_content = f"Question: {question_text}\n\nOptions:\n{options_str}\n\n{user_suffix}"
        if context:
            user_content = f"Context from web search:\n{context}\n\n{user_content}"
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user",   "content": user_content},
        ]
        gen_config = GenerationConfig(
            max_new_tokens=self._max_new_tokens,
            do_sample=self._do_sample,
            temperature=self._temperature if self._do_sample else None,
        )
        output = self.pipe(messages, generation_config=gen_config, return_full_text=False)
        return output[0]["generated_text"].strip()

    def answer(self, question_text: str, options: dict) -> int:
        # Router: decide code vs text path per question
        use_code = self._use_code_executor
        if self._use_router:
            use_code = _should_use_code(question_text)
            # Snapshot all settings that change between paths
            original_prompt    = self._system_prompt
            original_tokens    = self._max_new_tokens
            original_use_code  = self._use_code_executor
            original_retriever = self._retriever
            if use_code:
                self._system_prompt      = SYSTEM_PROMPTS.get("code", self._system_prompt)
                self._max_new_tokens     = 256
                self._use_code_executor  = True
                _MATHWORLD_TRIGGERS = {
                    # named results
                    "theorem", "lemma", "corollary", "proposition", "conjecture",
                    "formula", "identity", "inequality",
                    "law", "rule",
                    # named constructs
                    "series", "transform", "distribution",
                    # combinatorics specializations
                    "derangement", "stirling", "partition",
                    "surjection", "injection", "bijection",
                    "generating function", "inclusion-exclusion",
                    "pigeonhole", "catalan", "bell number",
                    # advanced topics
                    "eigenvalue", "eigenvector", "diagonaliz",
                    "convergence", "divergence", "radius of convergence",
                    "lagrange", "fourier", "laplace", "euler",
                    "markov", "bayes", "poisson",
                }
                if any(kw in question_text.lower() for kw in _MATHWORLD_TRIGGERS) and original_retriever is not None:
                    self._retriever = original_retriever  # MathWorld-only lookup for theorem questions
                    print(f"  [ROUTER] → code path (+MathWorld)")
                else:
                    self._retriever = None
                    print(f"  [ROUTER] → code path")
            else:
                self._system_prompt      = SYSTEM_PROMPTS.get("default", self._system_prompt)
                self._max_new_tokens     = 10
                self._use_code_executor  = False
                self._retriever          = None   # disable wiki for conceptual questions
                print(f"  [ROUTER] → text path")

        try:
            retrieval_options = None if use_code else options
            context  = self._retriever.get_context(question_text, retrieval_options) if self._retriever else ""
            if context:
                preview = context[:300] + ("..." if len(context) > 300 else "")
                print(f"  [CONTEXT] {preview}")
            t0       = time.time()
            response = self._generate(question_text, options, context)
            elapsed  = time.time() - t0
        finally:
            if self._use_router:
                # Always restore settings, even if an exception was raised
                self._system_prompt     = original_prompt
                self._max_new_tokens    = original_tokens
                self._use_code_executor = original_use_code
                self._retriever         = original_retriever

        if use_code:
            return self._answer_with_code(question_text, options, response, elapsed, context)
        if len(response) > 20:
            preview = response[:500] + ("..." if len(response) > 500 else "")
            print(f"  [RESPONSE] {preview}")
        return self._parse_token(response, options)

    # ── code extraction & preprocessing ──────────────────────────────────────

    def _preprocess_code(self, code: str) -> str:
        """Strip import statements — all needed modules are pre-loaded in the namespace."""
        cleaned = []
        for line in code.split("\n"):
            if re.match(r"\s*(import |from \S+ import )", line):
                continue  # drop the import; module is already in namespace
            cleaned.append(line)
        return "\n".join(cleaned)

    def _extract_code(self, response: str) -> Optional[str]:
        match = re.search(r"```python\s*\n(.*?)```", response, re.DOTALL)
        if not match:
            match = re.search(r"```\s*\n(.*?)```", response, re.DOTALL)
        return match.group(1).strip() if match else None

    # ── safe execution ────────────────────────────────────────────────────────

    def _execute_code(self, code: str) -> Optional[str]:
        code = self._preprocess_code(code)

        # ── safe builtins ─────────────────────────────────────────────────────
        safe_builtins = {
            name: getattr(builtins, name)
            for name in [
                # I/O & repr
                "print", "repr",
                # numeric types
                "int", "float", "complex", "bool",
                # string / char
                "str", "chr", "ord",
                # base conversion
                "bin", "oct", "hex",
                # containers
                "list", "dict", "tuple", "set", "frozenset",
                # iteration
                "range", "enumerate", "zip", "map", "filter",
                "sorted", "reversed", "iter", "next",
                # math builtins
                "abs", "min", "max", "sum", "pow", "round", "divmod",
                # logic / inspection
                "len", "any", "all", "callable", "isinstance",
                "type", "hasattr", "getattr", "hash",
            ]
            if hasattr(builtins, name)
        }

        # ── base namespace ────────────────────────────────────────────────────
        namespace = {
            "__builtins__": safe_builtins,
            "math":        math,
            "cmath":       cmath,
            "np":          np,
            "itertools":   itertools,
            "fractions":   fractions,
            "Fraction":    fractions.Fraction,
            "Counter":     collections.Counter,
            "defaultdict": collections.defaultdict,
            "deque":       collections.deque,
            "OrderedDict": collections.OrderedDict,
            "statistics":  stats_stdlib,
            "collections": collections,
        }

        # ── functools ─────────────────────────────────────────────────────────
        from functools import reduce
        namespace["reduce"]     = reduce
        namespace["accumulate"] = itertools.accumulate

        # ── math functions (no prefix) ────────────────────────────────────────
        for _fn in [
            "factorial", "floor", "ceil", "trunc", "fabs",
            "log", "log2", "log10", "exp",
            "sin", "cos", "tan", "asin", "acos", "atan", "atan2", "hypot",
            "degrees", "radians",
            "sinh", "cosh", "tanh", "asinh", "acosh", "atanh",
            "gcd", "comb", "perm",
            "sqrt", "isnan", "isinf", "isfinite", "isclose", "fmod", "remainder",
        ]:
            if hasattr(math, _fn):
                namespace[_fn] = getattr(math, _fn)

        # constants & aliases
        namespace["pi"]  = math.pi
        namespace["e"]   = math.e
        namespace["tau"] = getattr(math, "tau", 2 * math.pi)
        namespace["inf"] = math.inf
        namespace["nan"] = math.nan
        namespace["ln"]  = math.log
        namespace["lcm"] = math.lcm if hasattr(math, "lcm") else (
            lambda a, b: abs(a * b) // math.gcd(a, b))
        namespace["prod"] = math.prod if hasattr(math, "prod") else (
            lambda it: reduce(lambda a, b: a * b, it, 1))

        # ── numpy shortcuts ───────────────────────────────────────────────────
        namespace.update({
            "array":    np.array,
            "zeros":    np.zeros,
            "ones":     np.ones,
            "eye":      np.eye,
            "linspace": np.linspace,
            "arange":   np.arange,
            "dot":      np.dot,
            "cross":    np.cross,
            "linalg":   np.linalg,
        })

        # ── sympy ─────────────────────────────────────────────────────────────
        try:
            import sympy as sp
            namespace["sp"] = sp
            for _name in [
                # core algebra
                "symbols", "Symbol", "Function", "Wild", "Dummy",
                "solve", "solveset", "linsolve", "nonlinsolve", "nsolve",
                "dsolve",
                "Eq", "Ne", "Lt", "Le", "Gt", "Ge",
                "And", "Or", "Not", "Implies", "Equivalent",
                # simplification
                "simplify", "expand", "factor", "cancel", "together", "apart",
                "collect", "radsimp", "trigsimp", "powsimp", "combsimp",
                "nsimplify", "refine",
                # polynomials
                "Poly", "div", "roots", "real_roots", "nroots", "discriminant",
                # calculus
                "diff", "integrate", "Derivative", "Integral",
                "limit", "series", "summation",
                # linear algebra
                "Matrix", "det", "trace",
                # types & constants
                "sqrt", "root", "Pow", "Rational", "Integer", "Float",
                "pi", "E", "I", "oo", "zoo",
                "Sum", "Product",
                # numeric eval
                "N",
                # complex
                "Abs", "sign", "re", "im", "conjugate", "arg",
                # rounding
                "floor", "ceiling", "frac",
                # special
                "Piecewise", "Heaviside", "DiracDelta",
                # combinatorics
                "factorial", "factorial2", "subfactorial", "binomial",
                "rf", "ff", "RisingFactorial", "FallingFactorial",
                "necklaces", "lyndon",
                # sequences
                "fibonacci", "lucas", "catalan", "bell", "bernoulli",
                "euler", "harmonic", "genocchi",
                # number theory
                "factorint", "isprime", "nextprime", "prevprime", "primerange",
                "primepi", "totient", "mod_inverse", "primitive_root",
                "divisor_count", "divisor_sigma", "mobius",
                "jacobi_symbol", "legendre_symbol",
                # sets
                "Interval", "FiniteSet", "Union", "Intersection",
                "Complement", "EmptySet", "UniversalSet",
                # geometry
                "Point", "Line", "Segment", "Circle", "Triangle", "Ellipse",
                # trig
                "sin", "cos", "tan", "asin", "acos", "atan", "atan2",
                "sinh", "cosh", "tanh", "asinh", "acosh", "atanh",
                "cot", "csc", "sec", "sinc",
                # exp & log
                "exp", "log", "ln",
                # special functions
                "gamma", "loggamma", "beta", "polygamma", "digamma",
                "zeta", "erf", "erfc", "erfinv",
                "Ei", "Si", "Ci",
                # output
                "latex", "pretty",
            ]:
                if hasattr(sp, _name) and _name not in namespace:
                    namespace[_name] = getattr(sp, _name)
        except ImportError:
            pass

        # ── sympy deep-path shortcuts (not accessible via sp. prefix) ─────────
        try:
            from sympy.functions.combinatorial.numbers import (
                stirling as _stirling,
                nC as _nC,   # combinations C(n,k)
                nP as _nP,   # permutations P(n,k)
                nT as _nT,   # multisets (combinations with repetition)
            )
            namespace["stirling"] = _stirling
            namespace["nC"] = _nC
            namespace["nP"] = _nP
            namespace["nT"] = _nT
        except ImportError:
            pass
        try:
            from sympy.utilities.iterables import partitions as _partitions
            namespace["partitions"] = _partitions
        except ImportError:
            pass

        # ── scipy ─────────────────────────────────────────────────────────────
        try:
            import scipy.stats    as scipy_stats
            import scipy.integrate as scipy_integrate
            import scipy.special   as scipy_special
            import scipy.optimize  as scipy_optimize
            import scipy.linalg    as scipy_linalg

            namespace["stats"]  = scipy_stats
            namespace["linalg"] = scipy_linalg

            # distributions
            namespace.update({
                "norm":               scipy_stats.norm,
                "t_dist":             scipy_stats.t,
                "f_dist":             scipy_stats.f,
                "chi2":               scipy_stats.chi2,
                "binom":              scipy_stats.binom,
                "poisson":            scipy_stats.poisson,
                "geom":               scipy_stats.geom,
                "hypergeom":          scipy_stats.hypergeom,
                "nbinom":             scipy_stats.nbinom,
                "beta_dist":          scipy_stats.beta,
                "gamma_dist":         scipy_stats.gamma,
                "expon":              scipy_stats.expon,
                "uniform_dist":       scipy_stats.uniform,
                "lognorm":            scipy_stats.lognorm,
                "multinomial":        scipy_stats.multinomial,
                "multivariate_normal": scipy_stats.multivariate_normal,
            })

            # integration
            namespace.update({
                "quad":      scipy_integrate.quad,
                "dblquad":   scipy_integrate.dblquad,
                "nquad":     scipy_integrate.nquad,
                "odeint":    scipy_integrate.odeint,
                "solve_ivp": scipy_integrate.solve_ivp,
            })

            # special functions (use _func suffix to avoid clash with sympy)
            namespace.update({
                "erf":        scipy_special.erf,
                "erfc":       scipy_special.erfc,
                "erfinv":     scipy_special.erfinv,
                "gamma_func": scipy_special.gamma,
                "gammaln":    scipy_special.gammaln,
                "beta_func":  scipy_special.beta,
                "betaln":     scipy_special.betaln,
                "digamma":    scipy_special.digamma,
                "polygamma":  scipy_special.polygamma,
                "zeta_func":  scipy_special.zeta,
                "comb_f":     scipy_special.comb,
            })

            # optimization
            namespace.update({
                "brentq":          scipy_optimize.brentq,
                "bisect":          scipy_optimize.bisect,
                "fsolve":          scipy_optimize.fsolve,
                "minimize_scalar": scipy_optimize.minimize_scalar,
                "minimize":        scipy_optimize.minimize,
                "linprog":         scipy_optimize.linprog,
            })

            # linalg extras
            namespace.update({
                "linalg_solve": scipy_linalg.solve,
                "expm":         scipy_linalg.expm,
                "logm":         scipy_linalg.logm,
            })

        except ImportError:
            pass

        # ── statistics stdlib shortcuts ───────────────────────────────────────
        for _fn in ["mean", "median", "mode", "stdev", "variance",
                    "geometric_mean", "harmonic_mean", "quantiles",
                    "pstdev", "pvariance", "median_low", "median_high"]:
            if hasattr(stats_stdlib, _fn):
                namespace.setdefault(f"stat_{_fn}", getattr(stats_stdlib, _fn))
        # ── SymPy-aware wrappers ─────────────────────────────────────────────
        def _evalf_if_sympy(val):
            if hasattr(val, "evalf"):
                try:
                    val = val.evalf()
                except Exception:
                    pass
            return val

        def _smart_print(*args, **kwargs):
            builtins.print(*[_evalf_if_sympy(a) for a in args], **kwargs)

        def _safe_float(x):
            try:
                return builtins.float(x)
            except TypeError:
                return builtins.float(_evalf_if_sympy(x))

        def _safe_int(x):
            try:
                return builtins.int(x)
            except TypeError:
                return builtins.int(_evalf_if_sympy(x))

        namespace["print"] = _smart_print
        namespace["float"] = _safe_float
        namespace["int"]   = _safe_int

        import sys
        namespace["sys"] = sys  # allow models to call sys.setrecursionlimit() if needed

        captured  = io.StringIO()
        error: list = []

        def _run():
            old_limit = sys.getrecursionlimit()
            sys.setrecursionlimit(3000)
            try:
                with contextlib.redirect_stdout(captured):
                    exec(code, namespace)  # noqa: S102
                # If nothing was printed, try to print the last expression
                if not captured.getvalue().strip():
                    lines = [l for l in code.strip().splitlines() if l.strip()
                             and not l.strip().startswith("#")]
                    if lines:
                        last = lines[-1].strip()
                        # Only auto-print if last line looks like an expression, not an assignment
                        if last and "=" not in last or last.startswith("print"):
                            try:
                                val = eval(last, namespace)  # noqa: S307
                                if val is not None:
                                    val = _evalf_if_sympy(val)
                                    with contextlib.redirect_stdout(captured):
                                        print(val)
                            except Exception:
                                pass
            except Exception as e:
                error.append(e)
            finally:
                sys.setrecursionlimit(old_limit)

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        thread.join(timeout=5)
        if thread.is_alive():
            self._last_exec_error = "TIMEOUT (>5s)"
            return None
        if error:
            self._last_exec_error = f"{type(error[0]).__name__}: {error[0]}"
            return None
        result = captured.getvalue().strip()
        if not result:
            self._last_exec_error = "NO OUTPUT (code ran but nothing was printed)"
            return None
        self._last_exec_error = None
        return result

    # ── result → option mapping ───────────────────────────────────────────────

    @staticmethod
    def _normalize(text: str) -> str:
        """Normalize dashes and special chars before number extraction."""
        return text.replace('–', '-').replace('—', '-').replace('−', '-')

    def _match_to_option(self, result: str, options: dict) -> Optional[int]:
        result = self._normalize(result)

        # 1. Direct key match: if the output IS a valid option key, trust it immediately
        result_stripped = result.strip().strip(".,!?")
        if result_stripped in options:
            return int(result_stripped)

        # 2. Prefer standalone numbers (not embedded in identifiers like Z_3, S_10)
        #    A standalone number is NOT preceded/followed by a letter, underscore, or digit
        _NUM = r"-?\d+\.?\d*(?:[eE][+-]?\d+)?"
        standalone = re.findall(r"(?<![a-zA-Z_\d])" + _NUM + r"(?![a-zA-Z_\d])", result)
        all_nums   = re.findall(_NUM, result)
        result_nums = standalone if standalone else all_nums
        if not result_nums:
            return None
        result_val = float(result_nums[-1])

        # 3. Match against option values (also normalize option texts)
        best_id, best_diff = None, float("inf")
        for opt_id, opt_text in options.items():
            opt_text_n = self._normalize(opt_text)
            opt_nums   = re.findall(_NUM, opt_text_n)
            if opt_nums:
                diff = abs(float(opt_nums[-1]) - result_val)
                if diff < best_diff:
                    best_diff, best_id = diff, opt_id
        return int(best_id) if best_id is not None else None

    # ── logging ───────────────────────────────────────────────────────────────

    def _log(self, question_text: str, options: dict, code: Optional[str],
             code_result: Optional[str], final_answer: int, elapsed: float,
             context: str = ""):
        lines = ["=" * 80, f"QUESTION: {question_text}", "OPTIONS:"]
        for opt_id, opt_text in options.items():
            lines.append(f"  {opt_id}: {opt_text}")
        if context:
            lines += ["-" * 80, f"RETRIEVED CONTEXT:\n{context}"]
        lines.append("-" * 80)
        if code:
            lines += ["PYTHON CODE:", code, "-" * 40]
            if code_result is not None:
                lines.append(f"CODE OUTPUT: {code_result}")
            else:
                exec_err = getattr(self, "_last_exec_error", "unknown error")
                lines.append(f"CODE OUTPUT: FAILED — {exec_err}")
        else:
            lines.append("NO CODE GENERATED (conceptual question)")
        lines += [f"FINAL ANSWER: {final_answer}", f"TIME: {elapsed:.1f}s"]
        # entry left open — log_result() will append RESULT and closing line
        with open(self._log_path, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    def log_result(self, correct: bool):
        """Append the server verdict and running accuracy to the last open log entry."""
        if not hasattr(self, "_log_path"):
            return
        self._log_total   += 1
        self._log_correct += int(correct)
        verdict  = "CORRECT ✓" if correct else "WRONG ✗"
        accuracy = f"{self._log_correct}/{self._log_total} ({self._log_correct/self._log_total*100:.1f}%)"
        with open(self._log_path, "a", encoding="utf-8") as f:
            f.write(f"RESULT: {verdict}  |  RUNNING SCORE: {accuracy}\n{'=' * 80}\n\n")

    def finalize_log(self, correct: int, total: int):
        """
        Prepend the final score and rename the log file with a LOCAL_ prefix.
        Called from play_offline() after all questions are answered.
        """
        if not hasattr(self, "_log_path") or not os.path.exists(self._log_path):
            return

        with open(self._log_path, "r", encoding="utf-8") as f:
            content = f.read()

        header = (
            f"SCORE: {correct}/{total} ({correct/total*100:.1f}%)\n"
            f"{'=' * 80}\n\n"
        )

        timestamp = os.path.basename(self._log_path).replace("session_", "").replace(".txt", "")
        new_filename = f"LOCAL_{correct}of{total}_{timestamp}.txt"
        new_path = os.path.join(os.path.dirname(self._log_path), new_filename)

        with open(new_path, "w", encoding="utf-8") as f:
            f.write(header + content)

        os.remove(self._log_path)
        self._log_path = new_path
        print(f"Log saved to {new_path}")

    # ── agentic answer ────────────────────────────────────────────────────────

    def _answer_with_code(self, question_text: str, options: dict, response: str, elapsed: float, context: str = "") -> int:
        code         = self._extract_code(response)
        code_result  = None
        final_answer = None

        if code:
            code_result = self._execute_code(code)
            if code_result is not None:
                matched = self._match_to_option(code_result, options)
                if matched is not None:
                    final_answer = matched
                    print(f"  [AGENTIC OK  ] code ran → output: {code_result!r} → option {final_answer}")
            if final_answer is None:
                exec_err = getattr(self, "_last_exec_error", "unknown error")
                print(f"  [AGENTIC FAIL] {exec_err} → text fallback")
        else:
            print(f"  [STANDARD    ] no code generated → text parsing")

        if final_answer is None:
            final_answer = self._parse_token(response, options)

        self._log(question_text, options, code, code_result, final_answer, elapsed, context)
        return final_answer
