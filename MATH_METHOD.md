# Math Questions — Method Documentation

Everything tried for the Math category of the NLP challenge competition,
including models, retrieval strategies, code execution, prompt engineering,
and experimental approaches. Results are from the 63Q offline benchmark
(locally curated set of 63 questions) and the online competition platform.

---

## Online Competition Results

| Config | Games | Avg Correct / 15 | Accuracy % | Notes |
|---|---|---|---|---|
| qwen-1.5b | 10 | 0.30 | 2.0% | Baseline — tiny model, pure text |
| qwen-7b | 10 | 0.10 | 0.7% | 4-bit quant hurts heavily |
| llama-8b-wiki | 10 | 0.60 | 4.0% | Wikipedia retrieval, text-only |
| qwen-7b-code | 10 | 0.20 | 1.3% | Code executor + 4-bit Qwen |
| llama-8b-wiki-code | 10 | 0.50 | 3.3% | LLaMA + wiki + code |
| phi-4-mini-wiki-code | 20 | 0.40 | 2.7% | Phi-4-mini, Wikipedia + DDG, code |
| phi-4-dense | 15 | 0.20 | 1.3% | Phi-4 14B 4-bit + dense retrieval |
| **phi-4-mini-dense** | **15** | **1.47** | **9.8%** | **Best online — Phi-4-mini + FAISS** |
| phi-4-mini-hybrid | 20 | 1.30 | 8.7% | Hybrid Dense+BM25+RRF+cross-encoder |
| phi-4-mini-hybrid + fixes | 20 | 0.65 | 4.3% | Retrieval fixes (reverted — regression) |

---

## Local 63Q Benchmark Results

The 63Q set is a manually curated offline dataset used to validate configs
before submitting online. Questions span calculus, probability, combinatorics,
algebra, statistics, number theory, and abstract math.

| Config | Model | Score | Accuracy % | Notes |
|---|---|---|---|---|
| phi-4-mini-router | Phi-4-mini-instruct 3.8B | 27/63 | 42.9% | MathWorld+Wiki router, no code executor |
| llama-8b-router | LLaMA-3.1-8B-Instruct 4-bit | 30/63 | 47.6% | Router, no code executor |
| mathstral-7b-router | Mathstral-7B-v0.1 4-bit | 30/63 | 47.6% | Router, mathstral prompt |
| phi-4-mini-wiki-code | Phi-4-mini-instruct 3.8B | 32/63 | 50.8% | Wikipedia+DDG + code executor |
| qwen25-math-7b-dense | Qwen2.5-Math-7B-Instruct 4-bit | 33/63 | 52.4% | Dense retrieval, qwen_math prompt |
| phi-4-mini-dense-router | Phi-4-mini-instruct 3.8B | 34/63 | 54.0% | Dense + router (code vs text) |
| mathstral-7b-dense | Mathstral-7B-v0.1 4-bit | 31/63 | 49.2% | Dense retrieval |
| math-dual | Mathstral-7B + Phi-4-mini | 31/63 | 49.2% | Dual-model: reason→compute |
| qwen25-math-7b-dense-code (run 1) | Qwen2.5-Math-7B-Instruct 4-bit | 17/63 | 27.0% | SymPy crash bug |
| qwen25-math-7b-dense-code (run 2) | Qwen2.5-Math-7B-Instruct 4-bit | 29/63 | 46.0% | After SymPy float fix |
| phi-4-mini-dense (best) | Phi-4-mini-instruct 3.8B | 37/63 | 58.7% | Dense FAISS, code executor, code prompt |
| **phi-4-mini-hybrid** | **Phi-4-mini-instruct 3.8B** | **38/63** | **60.3%** | **Best local — Hybrid retrieval** |
| phi-4-mini-hybrid + fixes | Phi-4-mini-instruct 3.8B | 36/63 | 57.1% | Dense-veto + BM25 gate (reverted) |
| phi-4-mini-verify | Phi-4-mini-instruct 3.8B | 22/63 | 34.9% | Option verification prompt (failed) |

---

## Model Selection

### Why Phi-4-mini-instruct (3.8B) at full precision

Early runs tested LLaMA-3.1-8B-Instruct (4-bit) and Qwen2.5-Math-7B-Instruct (4-bit),
which are significantly larger on paper. However, 4-bit quantization substantially
degrades reasoning quality, especially for multi-step math. Phi-4-mini runs at full
bfloat16 precision within GPU memory, producing more reliable symbolic reasoning
despite the smaller parameter count.

Phi-4 (14B) was also tested with 4-bit quantization and performed worse than its
smaller sibling Phi-4-mini at full precision: 1.3% vs 9.8% online.

Mathstral-7B-v0.1 (Mistral fine-tuned on math competition data) was tested with
the expectation that domain-specific fine-tuning would help. It plateaued at 30/63
locally and did not improve over Phi-4-mini despite being purpose-built for math.

### Models tested

| Model | Params | Precision | Peak online |
|---|---|---|---|
| Qwen2.5-Math-1.5B-Instruct | 1.5B | bfloat16 | 2.0% |
| Qwen2.5-Math-7B-Instruct | 7.6B | 4-bit | 1.3% |
| LLaMA-3.1-8B-Instruct | 8.0B | 4-bit | 4.0% |
| Mathstral-7B-v0.1 | 7.2B | 4-bit | not submitted |
| Phi-4 | 14.7B | 4-bit | 1.3% |
| **Phi-4-mini-instruct** | **3.8B** | **bfloat16** | **9.8%** |

---

## Retrieval Strategy

### Phase 1 — No retrieval (baseline)

First configs used no retrieval at all. The model answered purely from its
parametric knowledge. This established a floor: pure-text reasoning on math
competition questions gives roughly 40-47% locally.

### Phase 2 — Wikipedia + DuckDuckGo (wiki retrieval)

`Retriever` class: searches Wikipedia for math-relevant terms extracted from
the question. Falls back to DuckDuckGo if Wikipedia returns nothing useful.
A `_build_query()` function strips LaTeX and common preamble phrases to produce
a cleaner search string. Multi-concept questions (e.g. "Statement 1 | ... Statement 2 | ...")
are decomposed into separate sub-queries via `_decompose_question()`.

Results were mixed: Wikipedia articles are often too broad, and for competition
questions the relevant theorem excerpt is hard to surface. Still, wiki retrieval
helped LLaMA reach 4.0% online (vs ~2% without it).

### Phase 3 — MathWorld router

`MathRetriever` (subclass of `Retriever`) queries MathWorld first — a curated
encyclopedia of mathematical definitions, theorems, and formulas. MathWorld is
much more likely to return a precise, concise definition than Wikipedia for
math-specific queries. Falls back to Wikipedia, then DuckDuckGo.

A routing heuristic (`_should_use_code`) classifies each question as either
computational (→ code executor, no retrieval) or conceptual (→ text path +
MathWorld retrieval). This prevented MathWorld lookups for questions that just
needed a numerical computation, saving time and avoiding confusion.

### Phase 4 — Dense retrieval with FAISS

`DenseRetriever`: encodes a local corpus of 12,629 MathWorld passages and
2,314 Wikipedia math passages using `intfloat/e5-base-v2` (a sentence embedding
model fine-tuned for retrieval tasks). At inference time, the question is encoded
and the top-K nearest passages are retrieved by cosine similarity via FAISS ANN
(approximate nearest neighbor) indexing.

This was the key breakthrough: moving from web-search retrieval (slow, noisy,
rate-limited) to local dense retrieval (fast, deterministic, always available)
jumped Phi-4-mini from ~2.7% to **9.8% online** (phi-4-mini-dense).

### Phase 5 — Hybrid Dense + BM25 + RRF + cross-encoder reranking

`HybridDenseRetriever` combines:
- Dense retrieval (e5-base-v2 + FAISS) for semantic matching
- BM25 sparse retrieval for exact keyword/symbol matching
- Reciprocal Rank Fusion (RRF) to merge both ranked lists
- Cross-encoder reranking (`cross-encoder/ms-marco-MiniLM-L-6-v2`) to rerank
  the top fused candidates by relevance

The intuition: dense retrieval handles paraphrases and conceptual similarity
well, but struggles with rare terminology, LaTeX symbols, and exact formula
matches. BM25 catches these. Fusion via RRF prevents either source from
dominating. Cross-encoder reranking applies a more expensive but accurate
relevance model to the shortlist.

Local improvement: 37/63 → 38/63. The online improvement was within noise
(9.8% → 8.7%), partly because the online game format (ends at first wrong
answer) is high-variance.

### Retrieval fix experiments (both reverted)

**Fix 1 — Dense-veto:** When dense retrieval found no passage above a similarity
threshold of 0.82, return an empty context string instead of passing noisy
low-confidence passages to the model. The idea was that a confused model with
irrelevant context might be worse than one answering from its parametric knowledge.

Result: **catastrophic regression**. Online went from 1.30 to 0.65 avg (13/20
games scored zero). The low-confidence passages were providing enough scaffolding
to orient the model. Removing them left the model fully unsupported. Reverted.

**Fix 2 — BM25 dense-gate:** Only admit BM25 candidates that also appeared in
the dense results above threshold 0.78, ensuring BM25 wasn't adding completely
unrelated passages. Also reverted along with Fix 1.

---

## Answer Generation Methods

### 1. Plain text (greedy)

The simplest path: the model produces a short response (max 20 tokens) and
the answer is extracted by scanning for a valid option key (0, 1, 2, 3) or
a `\boxed{N}` pattern. Fast (~0.1s), but limited to what the model's parametric
knowledge can surface.

### 2. Code executor (agentic AI)

`MathLLMModel` with `use_code_executor=True`: the model is prompted to write
a `python` code block. The code is extracted, pre-processed (import statements
stripped — all needed modules are pre-loaded in the sandbox namespace), and
executed in a sandboxed thread (5-second timeout, safe builtins only).

The sandbox namespace includes: `math`, `cmath`, `numpy`, `sympy`, `scipy`
(stats, integrate, special, optimize), `itertools`, `fractions`, `networkx`,
and all common functions available without a prefix (`sqrt`, `sin`, `cos`,
`solve`, `integrate`, `diff`, `N`, `factorial`, `comb`, `norm`, etc.).

The code output (stdout) is matched against the option values numerically:
fractions, scientific notation, and approximate floating-point values are all
handled. A "trivial code" guard rejects `print(N)` — a single digit — as a
degenerate case.

This is the biggest single improvement: it converts reasoning about numbers
into reliable numerical computation.

### 3. Question routing (code vs text)

`use_router=True`: before each question, `_should_use_code(question)` classifies
it using two keyword lists:

- `_CODE_QUESTION_KEYWORDS`: ~60 terms covering computation triggers (calculate,
  integrate, probability, derivative, combinations, etc.)
- `_TEXT_QUESTION_KEYWORDS`: ~15 terms for abstract/conceptual questions
  (which is true, must be true, assumption, observational study, isomorphic, etc.)

If the question has no numeric content at all (`_has_numeric_content` returns
False), the text path is always taken. A small override set
(`_ALWAYS_CODE_KEYWORDS`) forces code for geometry-formula questions even
without explicit digits.

On the code path: `system_prompt="code"`, `max_new_tokens=256`, retrieval
disabled (code is self-sufficient for numeric questions).
On the text path: `system_prompt="default"`, `max_new_tokens=20`, retrieval
enabled for MathWorld + Wikipedia lookup.

### 4. Dual-model architecture (math-dual)

A pipeline where two models collaborate:
- **Stage 1** (Mathstral-7B): receives `mathstral_reason` prompt — reason step
  by step, identify the framework, set up equations analytically, then output
  either `COMPUTE: [expression]` or `ANSWER: [option]`
- **Stage 2** (Phi-4-mini): if Stage 1 produced `COMPUTE:`, evaluate the
  expression numerically

The idea: Mathstral is a math competition specialist; Phi-4-mini is better at
running clean Python. Separating reasoning from computation could improve both.

Result: 31/63 — no improvement over either model alone. The handoff between
models introduced new failure modes: imprecise `COMPUTE:` expressions, wrong
symbolic setups that produced correct-looking code but wrong answers.

### 5. Option verification (experimental, failed)

Reframing answer generation as constraint-checking rather than forward computation.
Instead of computing the answer and matching it to an option, the model writes
code that *tests each option* and identifies which one satisfies the problem's
constraints.

Example intent: for a polynomial-roots question, instead of computing the roots
and matching to an option, the model would substitute each option back into the
polynomial and check which gives zero.

Implemented via a new `verify` system prompt and a corresponding `ANSWER: X`
output format detected in `_answer_with_code`.

Result: **22/63 locally (catastrophic)**. Three failure modes:

1. **String formatting bug in model-generated code**: model output `ANSWER: _3`
   or `ANSWER:3` (no space, or underscore prefix) — not matched by the regex
   `\bANSWER:\s*([0-3])\b`.
2. **Wrong mathematical reasoning**: the model-generated verifier computed the
   wrong constraint (e.g. checking `f(x) == option` instead of `f(option) == 0`).
3. **Conceptual questions**: for questions with no numerical constraints to test
   (abstract algebra, logic), the model produced nonsensical code or garbled
   plain-text responses.

Root cause: Phi-4-mini (3.8B) is too small to reliably write correct
constraint-checking code. Verification is a harder task than forward computation.
Reverted immediately.

### 6. Statement few-shot injection

For questions following the pattern "Statement 1 | X. Statement 2 | Y.",
a few-shot example is automatically prepended to the user message showing how
to evaluate both statements independently and output the correct combination.
This handles a common failure mode where the model conflated the two statements.

---

## System Prompt Engineering

Eight distinct system prompts were written and tested across different models
and configurations. All prompts are defined in `models/LLM.py` (base prompts)
and `models/MATH.py` (math-specific prompts).

### `default`

```
You are a quiz contestant. Given a multiple choice question,
reply with ONLY the number of the correct option. No explanation.
```

**Purpose:** minimal baseline. No role, no reasoning, just output the option.
Used for all early non-math runs and as the text-path target in the router
(conceptual questions where reasoning is unhelpful noise).

**Design rationale:** short prompts leave maximum token budget for the actual
question. On a 3.8B model, a long system prompt competes with the question
for the effective context window.

**Used by:** all LLMModel-based configs, router text path.

---

### `retrieval`

```
You are a quiz contestant. Given a multiple choice question and optional web context,
reply with ONLY the number of the correct option. No explanation.
```

**Purpose:** same as `default` but explicitly acknowledges the retrieval context
that follows. Without this, models sometimes ignored the context block entirely.

**Used by:** any config with `use_retrieval`, `use_dense_retrieval`, or
`use_hybrid_retrieval` enabled (when `system_prompt` is left at default).

---

### `math`

```
You are a math expert. Take a deep breath and reason step by step,
then reply with ONLY the number of the correct option.
```

**Purpose:** early math-specific prompt. The phrase "take a deep breath" comes
from a published finding (Yang et al., 2023) that emotional / deliberative
priming phrases improve LLM reasoning on benchmarks. "Reason step by step" is
a standard chain-of-thought trigger.

**Design rationale:** tried to get the model to show working before committing
to an answer, without specifying a rigid output format.

**Limitation:** on Phi-4-mini, this prompt produced verbose responses that
sometimes filled `max_new_tokens=256` without reaching a numeric conclusion,
causing token truncation and fallback to the first token (usually wrong).
Replaced by `code` for computational questions and `default` for conceptual ones.

**Used by:** early phi-4-mini experiments.

---

### `mathstral`

```
You are a math competition expert. For each question:
1. Identify the mathematical domain and the relevant framework or theorem.
2. Show your reasoning step by step, being precise about definitions.
3. Compute or derive the answer carefully.
4. Reply with ONLY the option number on the last line.
```

**Purpose:** structured reasoning prompt tailored to Mathstral-7B, a model
fine-tuned on math olympiad and competition data. Explicitly asks the model to
identify the domain first — a meta-cognitive step that helped Mathstral avoid
applying the wrong theorem.

**Design rationale:** Mathstral tends to output long, structured proofs. This
prompt aligns with its training distribution. The final line "ONLY the option
number" was necessary to prevent Mathstral from ending with a sentence like
"therefore the answer is option 2" which sometimes failed `_parse_token`.

**Observed behaviour:** on the 63Q set, Mathstral correctly identified the
domain and theorem for most questions but made arithmetic errors in the
computation step. Router configs helped by delegating computation to code.

**Used by:** mathstral-7b-router, mathstral-7b-dense, math-dual (Stage 1).

---

### `mathstral_reason`

```
You are a math competition expert. Reason step by step through the problem.
DO NOT compute any numerical values yourself.
Identify the correct mathematical framework, set up the equations, and simplify
as far as possible analytically.
End your response with EXACTLY one of:
  COMPUTE: [the precise expression, equation, or system to evaluate numerically]
  ANSWER: [option number]  (only when the answer follows from pure reasoning, no numbers needed)
```

**Purpose:** designed for the dual-model architecture. Forces Mathstral to
produce only symbolic/analytical work and explicitly output a `COMPUTE:` handoff
for numerical evaluation by a second model (Phi-4-mini).

**Design rationale:** separating symbolic reasoning from numerical computation
addresses the main failure mode of math LLMs: setting up the right equation but
making arithmetic mistakes. Mathstral handles the harder "which theorem applies"
step; a code executor handles the easier "evaluate this integral" step.

**Key detail:** "DO NOT compute any numerical values yourself" was necessary —
without it Mathstral would compute intermediate results and pass those forward,
losing the exact expression needed for reliable numerical evaluation.

**Used by:** math-dual config (Stage 1 of the dual-model pipeline).

---

### `qwen_math`

```
Please integrate natural language reasoning with programs to solve the problem above,
and put your final answer within \boxed{}.
For multiple choice questions, put the correct option number (0, 1, 2, or 3) inside \boxed{}.
Be brief: short reasoning, minimal code.
```

**Purpose:** matches Qwen2.5-Math's training format. Qwen math models are
trained with Tool-Integrated Reasoning (TIR): they interleave natural language
and code blocks, then produce a LaTeX `\boxed{}` answer.

**Design rationale:** using a prompt close to the training distribution avoids
the "format shock" that degrades outputs. The `\boxed{}` format is also easy
to parse reliably. A custom `_BoxedStoppingCriteria` stops generation as soon
as `\boxed{N}` is produced for a valid option key, saving significant tokens.

**Additional implementation:** `_generate()` appends a user suffix
"Answer concisely. End with \\boxed{option_number}." to reinforce the format.

**Limitation:** Qwen2.5-Math-7B in 4-bit quantization still underperformed
Phi-4-mini at full precision. The TIR-style code it generated was often more
complex than needed and prone to errors under quantization.

**Used by:** qwen25-math-7b-dense, qwen25-math-7b-dense-code.

---

### `code` *(best performing)*

```
You are a math expert solving multiple choice questions.
Default to writing code. Use a ```python``` block that prints the final result
for ANY question with numbers: algebra, calculus, probability, statistics,
combinatorics, geometry, complex numbers, or any theorem applied to a concrete value.
IMPORTANT: do NOT write any import statements. The following are already available:
  math, cmath, np (numpy), sp (sympy), stats (scipy.stats),
  quad, dblquad, nquad (scipy.integrate), erf, erfinv, gamma (scipy.special),
  norm, chi2, binom, poisson, t_dist, f_dist (scipy stats distributions),
  solve, symbols, sqrt, Rational, integrate, diff, Matrix, pi, I, oo,
  ln, log, log2, log10, exp, sin, cos, tan, asin, acos, atan, atan2,
  sinh, cosh, tanh, hypot, degrees, radians, floor, ceil, factorial,
  sqrt, prod, reduce, comb, gcd, lcm, perm, inf, pi, e,
  factorint, isprime, nextprime, totient, mod_inverse, binomial (sympy),
  N, Sum, Product, Abs, Piecewise,
  itertools, Fraction, Counter, defaultdict, deque,
  nx (networkx) for graph theory problems.
CRITICAL SymPy rule: SymPy expressions cannot be used directly in comparisons or round().
Always convert first: use float(N(expr)) before any comparison or rounding.
Examples: 'if float(N(f2)) < 0' not 'if f2 < 0'; 'round(float(N(expr)))' not 'round(expr)'.
After solve(), do: sols = [float(N(s)) for s in solve(...)]
Only reason in plain text for purely abstract questions with no numbers at all.
Always end your response with ONLY the option number on the last line.
```

**Purpose:** the main production prompt. Instructs the model to default to code
for any computational question, with a comprehensive list of available libraries
so the model never needs to write import statements (which are stripped anyway,
but writing them wastes tokens and sometimes produces "ImportError" in the model's
reasoning).

**Key design decisions:**

1. **"Default to writing code"** — not "write code if needed." The word "default"
   pushes the model harder toward code generation without making it strictly
   mandatory (which caused issues with purely abstract questions).

2. **Exhaustive library list** — listing every available function by name
   dramatically reduced `NameError` exceptions. Early versions had the model
   writing `import sympy as sp` then using `sp.solve()`, only for the import
   to be stripped, causing `NameError: name 'sp' is not defined`. The list
   clarifies what is pre-loaded.

3. **CRITICAL SymPy rule** — this single addition fixed approximately +2
   questions on the 63Q set. SymPy's symbolic expressions are not Python floats.
   Expressions like `if sp.solve(f, x)[0] < 0` crash with
   `TypeError: cannot determine truth value of Relational`, and
   `round(sp.sqrt(2))` returns a SymPy `Integer` not a Python int. The explicit
   rule and examples train the model to always call `float(N(expr))` first.

4. **User suffix reinforcement** — `_generate()` appends:
   "You MUST write a python code block that prints the answer, even when context
   is provided. Read constraints carefully — only enforce what the problem
   explicitly states. End with only the option number."
   The "even when context is provided" clause was added after observing the
   model sometimes deciding to answer from retrieval context without code,
   losing numerical precision.

5. **Fallback chain** — if code extraction fails, code execution fails, or
   no output is produced, the system falls back to parsing the model's text
   response directly. This means a code failure is not a total loss.

**Used by:** phi-4-mini-dense, phi-4-mini-hybrid, phi-4-mini-dense-router
(code path), phi-4-mini-verify config (reverted to `code` after verify failed).

---

### `verify` *(experimental — reverted after 22/63)*

```
You are a math expert. For each multiple-choice question, write a ```python``` block that:
1. Extracts the numerical value of each option (0, 1, 2, 3) from the problem.
2. Tests which option satisfies the problem constraints (substitute back, compute and compare, etc.).
3. Ends with: print('ANSWER:', correct_option_number)
Do NOT import anything. Available: math, np, sp (sympy), stats (scipy.stats), solve, symbols,
sqrt, integrate, diff, pi, e, N, norm, comb, factorial, comb, gcd, lcm, Rational, Fraction, etc.
SymPy rule: always use float(N(expr)) before comparing or rounding SymPy expressions.
For purely abstract questions (no numbers), reason in plain text and end with only the option number.
```

**Purpose:** option verification — instead of computing the answer and finding
the closest option, write code that *tests each option against the problem's
constraints* and identifies which satisfies them. The expected output is
`ANSWER: X` where X is the option key.

**Design rationale:** verification is often easier than discovery. Substituting
a candidate answer back into an equation to check it holds is a simpler
computation than solving the equation from scratch. This approach also makes
the model's reasoning more transparent (each test is explicit code) and should
reduce format errors (instead of matching a float to an option, the model
directly states `ANSWER: 2`).

**Implementation additions:**
- `_answer_with_code()` checks for `ANSWER:\s*([0-3])` in code output before
  falling through to the standard numerical matching path.
- User message suffix: "Write a python block that tests each option and
  prints: ANSWER: X"

**Why it failed (22/63):**

1. **String formatting bugs** — the model frequently produced `ANSWER: _3`,
   `ANSWER:3` (no space), or `ANSWER: option3` from f-string misuse. The
   regex `\bANSWER:\s*([0-3])\b` did not match these variants, causing fallback
   to numerical matching which was now completely wrong (no numeric output).

2. **Incorrect constraint extraction** — the model often misidentified what
   "satisfies the constraints" means. For a question about polynomial roots, it
   checked `f(option_value) == 0` when the option was a coefficient, not a root.

3. **Conceptual questions** — for abstract math (group theory, topology,
   statistics interpretation), there are no testable numerical constraints.
   The model produced syntactically correct code that printed nonsense or
   crashed, and the fallback text response also failed because the verify prompt
   had conditioned the model away from clean text reasoning.

**Conclusion:** 3.8B model is too small to reliably meta-reason about which
constraints to test per question. Forward computation (compute the answer,
match to option) is simpler and more reliable.

---

## Prompt Engineering Iterations Summary

The progression from worst to best prompt performance:

1. `default` → 42.9% (27/63) — baseline, no domain knowledge
2. `math` → ~45% — "take a deep breath" helps slightly
3. `mathstral` on Mathstral-7B → 47.6% — better structure, same ceiling
4. `code` without SymPy rule → ~50% — code helps but crashes on SymPy types
5. `code` with SymPy rule → 58.7% (37/63) — single rule change, +2 questions
6. `code` + hybrid retrieval → 60.3% (38/63) — better context
7. `verify` → 34.9% (22/63) — worse than baseline (reverted)

---

## Key Engineering Details

### Code sandbox

The code executor (`_execute_code`) runs in a daemon thread with a 5-second
timeout. The execution namespace is carefully constructed:

- `__builtins__` is replaced with a safe subset (no `open`, `__import__`, `eval`,
  `exec`, `compile`, `os`, `sys.exit`, etc.)
- All needed math libraries are pre-loaded: `math`, `cmath`, `numpy`, `sympy`,
  `scipy`, `itertools`, `fractions`, `collections`, `networkx`
- `print` is replaced by a SymPy-aware wrapper that calls `.evalf()` on any
  SymPy expression before printing (catches cases where the model prints a
  SymPy object directly)
- `float()` and `int()` are replaced by wrappers that unwrap single-element
  lists (common from `sympy.solve()`) and call `.evalf()` if needed
- Import statements in model-generated code are silently stripped by
  `_preprocess_code()` (prevents NameError when the model writes `import sympy`)

### Output matching

After code execution, `_match_to_option()` finds the best matching option:
- First checks if the output is directly a valid option key ("2", "0", etc.)
- Extracts all standalone numbers (not embedded in identifiers like `Z_3`)
- Handles fractions in option text: `{3}/{4}`, `3/4`, and `\frac{3}{4}` all
  parse to 0.75 for comparison
- Matches by minimum absolute difference (tolerance: exact match preferred)

### Logging

Every code-executor run is logged to `AgenticAI_scripts/` with:
- Full question text and options
- Retrieved context (if any)
- Generated Python code
- Code output / error message
- Final answer and time taken
- After-the-fact verdict and running accuracy (from `log_result()`)

Log files are renamed to `LOCAL_Xof63_timestamp.txt` after offline evaluation.

---

## Failure Taxonomy

From analysing the 63Q logs:

| Failure type | Approx frequency | Example |
|---|---|---|
| Wrong theorem / domain | ~25% | Applied normal dist to a non-normal problem |
| SymPy type error (pre-fix) | ~10% | `round(sp.sqrt(2))` crashed |
| Code produced no output | ~8% | Loop with no `print` |
| Code logic error | ~15% | Wrong formula or off-by-one |
| Conceptual question guessed | ~20% | Text-path answers for abstract algebra |
| Retrieval mismatch | ~10% | Retrieved unrelated passage, confused model |
| Trivial code guard triggered | ~5% | Model wrote `print(2)` as a guess |
| Verify format mismatch | ~100% verify runs | `ANSWER: _3` not matched |

---

## Config Summary

All configs are YAML files under `config/`. Key fields:

| Config file | Model | Prompt | Retrieval | Code |
|---|---|---|---|---|
| phi-4-mini-router.yaml | Phi-4-mini | code/default (router) | MathWorld+Wiki | router |
| phi-4-mini-wiki-code.yaml | Phi-4-mini | code | Wikipedia+DDG | yes |
| phi-4-mini-dense.yaml | Phi-4-mini | code | FAISS dense | yes |
| phi-4-mini-dense-router.yaml | Phi-4-mini | code/default | FAISS dense | router |
| phi-4-mini-hybrid.yaml | Phi-4-mini | code | Hybrid (Dense+BM25+RRF) | yes |
| phi-4-mini-verify.yaml | Phi-4-mini | code (was verify) | Hybrid | yes |
| mathstral-7b-router.yaml | Mathstral-7B | mathstral | MathWorld+Wiki | no |
| mathstral-7b-dense.yaml | Mathstral-7B | mathstral | FAISS dense | no |
| math-dual.yaml | Mathstral-7B+Phi-4-mini | mathstral_reason + code | FAISS dense | stage 2 |
| qwen25-math-7b-dense.yaml | Qwen2.5-Math-7B | qwen_math | FAISS dense | no |
| qwen25-math-7b-dense-code.yaml | Qwen2.5-Math-7B | qwen_math | FAISS dense | yes |

The final submitted config is `phi-4-mini-hybrid.yaml` (38/63 local, 8.7% online
best). The `phi-4-mini-verify.yaml` config was reverted to `system_prompt: "code"`
after the 22/63 disaster and is effectively equivalent to `phi-4-mini-hybrid.yaml`.

---

## What Worked

- **Full-precision small model** beats quantized large model (Phi-4-mini > Phi-4 4-bit)
- **Code executor** is the single biggest accuracy driver for computational questions
- **FAISS local dense retrieval** over Wikipedia/DDG web search — faster, more relevant
- **SymPy float rule** in the prompt — single change, +2 questions fixed
- **Keyword-based routing** (code vs text path) prevents code noise on conceptual questions
- **Statement few-shot** helps on Statement 1 / Statement 2 multi-part questions
- **Hybrid BM25+Dense+RRF** — marginal improvement, worth keeping

## What Did Not Work

- **Quantized large models** — 4-bit degrades reasoning too much
- **Domain-specific models** (Mathstral, Qwen-Math) — didn't beat full-precision generalist
- **Dense-veto fix** — removing low-confidence context was net negative
- **Option verification prompt** — model too small for meta-level constraint reasoning
- **Dual-model pipeline** — handoff introduces more errors than it saves
- **Long reasoning prompts** — Phi-4-mini is sensitive to prompt length; every extra line
  can destabilize output format and cause token truncation
