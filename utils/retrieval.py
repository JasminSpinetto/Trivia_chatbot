import html
import re
import requests
from concurrent.futures import ThreadPoolExecutor, wait
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


_PARALLEL_TIMEOUT = 5.0   # per search-phase budget; extract batch is one extra call
_MAX_CONTEXT_LEN  = 5000
_MAX_ARTICLES     = 2     # keep only the top-scoring articles to avoid "lost in the middle"
_WIKI_SENTENCES   = 25    # sentences per article summary
_MAX_WORKERS      = 4     # concurrent search calls (I/O-bound, safe for Colab)

_WIKI_API  = "https://en.wikipedia.org/w/api.php"
_WIKT_API  = "https://en.wiktionary.org/w/api.php"
_WIKI_HDRS = {"User-Agent": "PoliMillionaire-quiz/1.0 (educational NLP project)"}



_STOP_WORDS = frozenset({
    "what", "which", "how", "where", "when", "why", "who", "whom", "whose",
    "the", "a", "an",
    "of", "for", "to", "from", "by", "at", "in", "on", "into", "with",
    "about", "between", "among", "during", "after", "before",
    "and", "or", "but",
    "is", "are", "was", "were", "did", "does", "do", "have", "has", "had",
    "be", "been", "being", "would", "could", "should", "will", "can", "may",
    "might", "must", "used",
    "it", "its", "this", "that", "these", "those", "they", "them", "their",
    "he", "she", "his", "her", "we", "our", "you", "your",
    "also", "then", "than", "not", "no", "more", "most", "such", "very",
    "term", "refers", "refer", "using", "based", "given", "found", "known",
    "called", "named", "describes", "describe", "connection", "according",
    "following", "primary", "secondary", "fundamental", "best", "main",
    "whole", "general", "specific", "certain", "common",
    "times", "time", "year", "years", "century", "period",
    "article", "question", "option", "answer", "passage",
})


_QUOTED_RE = re.compile(
    r"(?:the\s+)?(?:term|word|concept|phrase|name)\s+['‘’“”]([^'‘’“”\"]+)['‘’“”]"
    r"|['‘’“”]([^'‘’“”\"]{1,40})['‘’“”]",
    re.IGNORECASE,
)

def _quoted_terms(question: str) -> list:
    """Extract terms highlighted in quotes — e.g. the term 'bꜣk', 'Weltanschauung'.
    These are usually the most important concept in a terminology question and
    may contain non-ASCII characters that our keyword regex misses entirely.
    """
    seen, out = set(), []
    for m in _QUOTED_RE.finditer(question):
        term = (m.group(1) or m.group(2) or "").strip()
        if term and term.lower() not in seen:
            seen.add(term.lower())
            out.append(term)
    return out


def _keywords(question: str) -> list:
    words = re.findall(r"[a-zA-Z]+", question.lower())
    return sorted({w for w in words if len(w) > 4 and w not in _STOP_WORDS})


def _proper_nouns(question: str) -> list:
    words = re.findall(r"\b[a-zA-Z]+\b", question)
    return [w.lower() for i, w in enumerate(words)
            if i > 0 and w[0].isupper() and len(w) >= 4]


def _focused_query(question: str) -> str:
    """Proper nouns first, then content words — up to 6 tokens."""
    words = re.findall(r"\b[a-zA-Z']+\b", question)
    proper  = [re.sub(r"'s?$", "", w) for i, w in enumerate(words)
               if i > 0 and w[0].isupper() and len(w) > 2]
    content = [w.lower() for w in words if len(w) > 4 and w.lower() not in _STOP_WORDS]
    seen    = {p.lower() for p in proper}
    extra   = [w for w in content if w not in seen]
    return " ".join((proper + extra)[:6])


def _options_query(question: str, options: dict) -> str:
    """Proper nouns from question + distinctive terms from answer options."""
    words = re.findall(r"\b[a-zA-Z']+\b", question)
    proper = [re.sub(r"'s?$", "", w) for i, w in enumerate(words)
              if i > 0 and w[0].isupper() and len(w) > 2]
    q_words = set(re.findall(r"[a-zA-Z]+", question.lower()))
    opt_terms: list = []
    for opt_text in options.values():
        for w in re.findall(r"[a-zA-Z]+", opt_text.lower()):
            if len(w) > 4 and w not in _STOP_WORDS and w not in q_words and w not in opt_terms:
                opt_terms.append(w)
    return " ".join((proper + opt_terms)[:8])


def _proper_noun_phrases(question: str) -> list:
    """Extract sequences of consecutive capitalised words as distinct phrases.

    'In Ancient Greek, the loss … in Koine Greek' → ['Ancient Greek', 'Koine Greek']
    """
    words = re.findall(r"\b[a-zA-Z']+\b", question)
    phrases, current = [], []
    for i, w in enumerate(words):
        if i > 0 and w[0].isupper() and len(w) > 2:
            current.append(re.sub(r"'s?$", "", w))
        else:
            if current:
                phrases.append(" ".join(current))
                current = []
    if current:
        phrases.append(" ".join(current))
    return phrases


def _is_relevant(question: str, context: str) -> bool:
    """Accept if ≥50% of question proper nouns appear in context, OR if any
    content keyword overlaps (exact or 4-char prefix stem).

    The stem fallback catches morphological variants like "roofing"/"roof",
    "material"/"materials", "ancient"/"ancients" that exact matching misses.
    """
    c_lower = context.lower()
    proper  = _proper_nouns(question)
    if proper:
        matches = sum(1 for p in proper if p in c_lower)
        if matches / len(proper) >= 0.5:
            return True
    q_kw = set(_keywords(question))
    c_kw = set(_keywords(context))
    if q_kw & c_kw:
        return True
    # Stem fallback: share a 4-char prefix (e.g. "roof" matches "roofing")
    q_stems = {w[:4] for w in q_kw if len(w) >= 5}
    c_stems = {w[:4] for w in c_kw if len(w) >= 5}
    return bool(q_stems & c_stems)


def _strip_html(raw: str) -> str:
    text = re.sub(r"<[^>]+>", " ", raw)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


class Retriever:
    """Wikipedia retrieval via the MediaWiki REST API.

    API call pattern (minimises total requests to avoid 429s):
      Phase 1 — search: N queries run in parallel, each returns up to 5
                candidate page titles  →  N search API calls
      Phase 2 — extract: all unique titles fetched in ONE batched call
                (MediaWiki accepts up to 50 titles per request)
      Total: N + 1 calls instead of 2N.
    """

    def __init__(self, log_fn=None):
        self._log = log_fn or (lambda msg: None)
        # Retry only on HTTP status errors (429/5xx), never on connection errors.
        # connect=0 makes connection failures fast-fail so we fall through to
        # the next retrieval stage immediately instead of wasting seconds.
        self._session = requests.Session()
        _retry = Retry(
            total=1,
            connect=0,               # fail fast on connection errors
            read=1,
            backoff_factor=2.0,      # one retry after 2s — gives Wikipedia time to clear burst rate limits
            status_forcelist=[429, 500, 502, 503, 504],
            respect_retry_after_header=False,  # ignore Retry-After (could be 60s+, busts budget)
        )
        self._session.mount("https://", HTTPAdapter(max_retries=_retry))
        self._session.headers.update(_WIKI_HDRS)

    def _search_titles(self, query: str, srlimit: int = 5):
        """Return up to srlimit Wikipedia page titles, or None on error.

        Returning None (vs empty list) lets callers distinguish a network/
        rate-limit failure from a query that simply matched no articles.
        """
        try:
            r = self._session.get(
                _WIKI_API,
                params={"action": "query", "list": "search",
                        "srsearch": query, "srlimit": srlimit, "format": "json"},
                timeout=3,
            )
            r.raise_for_status()
            return [hit["title"] for hit in r.json()["query"]["search"]]
        except Exception as e:
            self._log(f"WIKI SEARCH ERR  : {query!r} → {e}")
            return None  # signals error, not "no results"

    def _fetch_extracts(self, titles: list) -> dict:
        """Fetch intro extracts for all titles in ONE batched API call.

        Returns {title: text} for every title that has content.
        MediaWiki follows redirects automatically via redirects=1.
        """
        if not titles:
            return {}
        try:
            r = self._session.get(
                _WIKI_API,
                params={"action": "query", "prop": "extracts",
                        "exintro": True, "exsentences": _WIKI_SENTENCES,
                        "titles": "|".join(titles[:50]),
                        "redirects": 1, "format": "json"},
                timeout=6,
            )
            r.raise_for_status()
            out = {}
            for pid, page in r.json()["query"]["pages"].items():
                if pid == "-1":
                    continue
                text = _strip_html(page.get("extract", ""))
                if text:
                    out[page.get("title", titles[0])] = text
            return out
        except Exception as e:
            self._log(f"WIKI EXTRACT ERR : {e}")
            return {}

    def _fetch_wiktionary(self, terms: list) -> dict:
        """Fetch Wiktionary entries for the given terms in one batched call.

        Returns {term: text}.  No relevance check needed — terms come directly
        from the quoted-term extractor, so they are the exact concept being asked
        about.  We skip exintro because Wiktionary pages start with POS sections,
        not a traditional lead paragraph.
        """
        if not terms:
            return {}
        try:
            r = self._session.get(
                _WIKT_API,
                params={"action": "query", "prop": "extracts",
                        "exsentences": 8,
                        "titles": "|".join(terms[:10]),
                        "redirects": 1, "format": "json"},
                timeout=4,
            )
            r.raise_for_status()
            out = {}
            for pid, page in r.json()["query"]["pages"].items():
                if pid == "-1":
                    continue
                text = _strip_html(page.get("extract", ""))
                if text:
                    out[page.get("title", terms[0])] = text
            return out
        except Exception as e:
            self._log(f"WIKT EXTRACT ERR : {e}")
            return {}

    def _run_queries(self, queries: list, srlimit: int = 5) -> tuple:
        """Run search queries in parallel, then batch-fetch all extracts.

        Returns (extracts: dict, had_error: bool).
        had_error=True means at least one search hit a network/rate-limit
        error — callers use this to skip further Wikipedia attempts and
        fall through to DuckDuckGo immediately.
        Total API calls = len(queries) searches + 1 extract batch.
        """
        if not queries:
            return {}, False

        # Phase 1: parallel searches → candidate title lists
        with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as ex:
            future_map = {ex.submit(self._search_titles, q, srlimit): q for q in queries}
            done, _    = wait(future_map.keys(), timeout=_PARALLEL_TIMEOUT)

        had_error = False
        seen_titles, ordered_titles = set(), []
        for fut in done:
            q = future_map[fut]
            try:
                result = fut.result()
                if result is None:       # _search_titles signals error with None
                    had_error = True
                else:
                    for title in result:
                        if title not in seen_titles:
                            seen_titles.add(title)
                            ordered_titles.append(title)
            except Exception as e:
                had_error = True
                self._log(f"SEARCH ERROR     : {e}")

        if not ordered_titles:
            return {}, had_error

        # Phase 2: one batch extract call for all unique titles
        self._log(f"WIKI TITLES      : {ordered_titles}")
        return self._fetch_extracts(ordered_titles), had_error

    def get_context(self, question: str, options: dict = None) -> str:
        if options is None:
            options = {}

        proper_only  = " ".join(
            re.sub(r"'s?$", "", w)
            for i, w in enumerate(re.findall(r"\b[a-zA-Z']+\b", question))
            if i > 0 and w[0].isupper() and len(w) > 2
        )
        focused      = _focused_query(question)
        opts_query   = _options_query(question, options)
        keywords     = _keywords(question)

        quoted   = _quoted_terms(question)

        self._log(f"SEARCH QUOTED    : {quoted}")
        self._log(f"SEARCH PROPER    : {proper_only!r}")
        self._log(f"SEARCH FOCUSED   : {focused!r}")
        self._log(f"SEARCH OPTIONS   : {opts_query!r}")
        self._log(f"SEARCH KEYWORDS  : {keywords}")

        # Quoted terms go first — they name the exact concept being asked about.
        # For questions like "the term 'bꜣk'" the quoted string contains
        # non-ASCII characters invisible to the ASCII keyword/noun extractors.
        quoted_queries = [f"{t} {' '.join(keywords[:2])}" for t in quoted] if quoted else []

        # Deduplicated primary query list
        seen_q, queries = set(), []
        for q in quoted_queries + [proper_only, focused, opts_query]:
            if q and q.lower() not in seen_q:
                seen_q.add(q.lower())
                queries.append(q)

        if not queries:
            self._log("SEARCH SOURCES   : no queries generated")
            return ""

        self._log(f"SEARCH SOURCES   : {[f'wikipedia({q!r})' for q in queries]}")

        # Launch Wiktionary in the background while Wikipedia searches run.
        # Only for quoted terms — we know the exact page title, no search needed.
        wikt_executor = None
        wikt_future   = None
        if quoted:
            wikt_executor = ThreadPoolExecutor(max_workers=1)
            wikt_future   = wikt_executor.submit(self._fetch_wiktionary, quoted)
            self._log(f"WIKT LOOKUP      : {quoted}")

        extracts, wiki_error = self._run_queries(queries)

        # Relevance filter and deduplication
        seen_text, unique = set(), []
        for title, ctx in extracts.items():
            if ctx[:80] not in seen_text:
                if _is_relevant(question, ctx):
                    seen_text.add(ctx[:80])
                    unique.append(ctx)
                    self._log(f"WIKI ACCEPTED    : [{title}]")
                else:
                    self._log(f"WIKI REJECTED    : [{title}] (relevance too low)")

        # Collect Wiktionary results — bypass relevance filter (exact term match)
        if wikt_future is not None:
            try:
                wikt_extracts = wikt_future.result(timeout=4.0)
                for title, ctx in wikt_extracts.items():
                    if ctx and ctx[:80] not in seen_text:
                        seen_text.add(ctx[:80])
                        unique.append(ctx)
                        self._log(f"WIKT ACCEPTED    : [{title}]")
            except Exception as e:
                self._log(f"WIKT TIMEOUT     : {e}")
            finally:
                wikt_executor.shutdown(wait=False)

        # Phrase/combo fallback — skipped when Wikipedia is erroring (429, connection
        # refused, etc.) since those queries would also fail.
        if not unique and not wiki_error:
            phrase_list = [
                p for p in _proper_noun_phrases(question)
                if p.lower() not in seen_q and len(p) > 2
            ]
            kw_exclude    = {w for p in phrase_list for w in p.lower().split()}
            extra_kw      = [k for k in keywords if k not in kw_exclude][:3]
            combo_queries = []
            for p in phrase_list:
                for k in extra_kw:
                    q = f"{p} {k}"
                    if q.lower() not in seen_q:
                        seen_q.add(q.lower())
                        combo_queries.append(q)

            fallback_queries = phrase_list + combo_queries
            if fallback_queries:
                self._log(f"FALLBACK PHRASES : {phrase_list}")
                if combo_queries:
                    self._log(f"FALLBACK COMBOS  : {combo_queries}")
                fb_extracts, _ = self._run_queries(fallback_queries)
                for title, ctx in fb_extracts.items():
                    if ctx[:80] not in seen_text:
                        if _is_relevant(question, ctx):
                            seen_text.add(ctx[:80])
                            unique.append(ctx)
                            self._log(f"FALLBACK ACCEPT  : [{title}]")
                        else:
                            self._log(f"FALLBACK REJECT  : [{title}] (relevance too low)")

        if not unique:
            self._log("SEARCH COMBINED  : (none)")
            return ""

        # Sort so the most on-topic article leads the context window.
        # Proper nouns weighted 2× over generic keywords.
        q_kw     = set(_keywords(question))
        q_proper = set(_proper_nouns(question))
        def _score(ctx: str) -> int:
            lower = ctx.lower()
            return (sum(1 for kw in q_kw if kw in lower)
                    + sum(2 for p in q_proper if p in lower))
        unique.sort(key=_score, reverse=True)
        unique = unique[:_MAX_ARTICLES]

        combined = "\n\n---\n\n".join(unique)[:_MAX_CONTEXT_LEN]
        self._log(f"SEARCH COMBINED  : {len(unique)} source(s), {len(combined)} chars")
        return combined

    def _search_options(self, options: dict, question: str,
                        seen_text: set, unique: list) -> None:
        """Search each answer option as its own Wikipedia query (srlimit=1).

        Results are appended in-place to `unique` and tracked in `seen_text`.
        This gives the model grounding for every candidate answer, not just
        the one the LLM happened to query about.
        """
        if not options:
            return
        opt_queries = list(dict.fromkeys(v.strip() for v in options.values() if v.strip()))
        if not opt_queries:
            return
        self._log(f"WIKI OPT SEARCH  : {opt_queries}")
        opt_extracts, _ = self._run_queries(opt_queries, srlimit=1)
        for title, ctx in opt_extracts.items():
            if ctx[:80] not in seen_text:
                if _is_relevant(question, ctx):
                    seen_text.add(ctx[:80])
                    unique.append(ctx)
                    self._log(f"WIKI OPT ACCEPT  : [{title}]")
                else:
                    self._log(f"WIKI OPT REJECT  : [{title}] (relevance too low)")

    # ── helpers shared by individual-tool methods ──────────────────────────────

    def _score_and_combine(self, unique: list, question: str) -> str:
        """Score articles by relevance, keep top _MAX_ARTICLES, return combined string."""
        if not unique:
            return ""
        q_kw     = set(_keywords(question))
        q_proper = set(_proper_nouns(question))
        def _score(ctx: str) -> int:
            lower = ctx.lower()
            return (sum(1 for kw in q_kw if kw in lower)
                    + sum(2 for p in q_proper if p in lower))
        unique.sort(key=_score, reverse=True)
        unique   = unique[:_MAX_ARTICLES]
        combined = "\n\n---\n\n".join(unique)[:_MAX_CONTEXT_LEN]
        self._log(f"CONTEXT READY    : {len(unique)} source(s), {len(combined)} chars")
        return combined

    # ── individual tool retrieval methods ──────────────────────────────────────

    def _opensearch_title(self, title: str):
        """Resolve a guessed article title via Wikipedia's opensearch endpoint.

        opensearch does title-prefix matching (what the Wikipedia search box uses),
        much more precise than list=search full-text for well-known article names.
        Returns the best non-disambiguation title string, or None on miss/error.
        """
        if not title:
            return None
        try:
            r = self._session.get(
                _WIKI_API,
                params={"action": "opensearch", "search": title,
                        "limit": 5, "namespace": 0, "format": "json"},
                timeout=3,
            )
            r.raise_for_status()
            data   = r.json()
            titles = data[1] if len(data) > 1 else []
            # Exact case-insensitive match wins; otherwise first non-disambiguation
            for t in titles:
                if t.lower() == title.lower():
                    return t
            for t in titles:
                if "(disambiguation)" not in t.lower():
                    return t
            return titles[0] if titles else None
        except Exception as e:
            self._log(f"OPENSEARCH ERR   : {title!r} → {e}")
            return None

    def _get_wikipedia_context(self, question: str, options: dict, query=None) -> str:
        """Wikipedia-only retrieval.

        When query is a dict {"title": ..., "alternatives": [...]} (from the
        structured LLM JSON output):
          Phase 1 — opensearch with the LLM-guessed title (precise title match).
          Phase 2 — list=search with the alternative terms if Phase 1 misses.

        When query is a plain string or None, falls back to the single best
        auto-focused query via list=search (original behaviour).
        """
        if isinstance(query, dict):
            title        = query.get("title", "").strip()
            alternatives = [a for a in query.get("alternatives", []) if a.strip()]

            # Phase 1: opensearch with LLM title guess
            matched = self._opensearch_title(title) if title else None
            if matched:
                self._log(f"OPENSEARCH HIT   : {matched!r}")
                extracts  = self._fetch_extracts([matched])
                seen_text, unique = set(), []
                for t, ctx in extracts.items():
                    if ctx[:80] not in seen_text and _is_relevant(question, ctx):
                        seen_text.add(ctx[:80])
                        unique.append(ctx)
                        self._log(f"WIKI ACCEPTED    : [{t}]")
                    elif ctx[:80] not in seen_text:
                        self._log(f"WIKI REJECTED    : [{t}] (relevance too low)")
                if unique:
                    return self._score_and_combine(unique, question)
                self._log("OPENSEARCH MISS  : not relevant → trying alternatives")
            else:
                self._log(f"OPENSEARCH MISS  : no title match for {title!r}")

            # Phase 2: list=search with alternatives (or title as last resort)
            fallback = alternatives or ([title] if title else [])
            if not fallback:
                return ""
            self._log(f"WIKI FALLBACK    : {fallback}")
            extracts, _ = self._run_queries(fallback, srlimit=3)
            seen_text, unique = set(), []
            for t, ctx in extracts.items():
                if ctx[:80] not in seen_text:
                    if _is_relevant(question, ctx):
                        seen_text.add(ctx[:80])
                        unique.append(ctx)
                        self._log(f"WIKI ACCEPTED    : [{t}]")
                    else:
                        self._log(f"WIKI REJECTED    : [{t}] (relevance too low)")
            return self._score_and_combine(unique, question)

        # Plain string or None — original single-query behaviour
        if query:
            search_q = query
            self._log(f"WIKI QUERY       : {search_q!r}  (llm-generated)")
        else:
            quoted = _quoted_terms(question)
            if quoted:
                keywords = _keywords(question)
                search_q = f"{quoted[0]} {' '.join(keywords[:2])}".strip()
            else:
                search_q = _focused_query(question)
            if not search_q:
                self._log("WIKI QUERY       : (no query generated)")
                return ""
            self._log(f"WIKI QUERY       : {search_q!r}  (auto)")

        extracts, _ = self._run_queries([search_q], srlimit=3)
        seen_text, unique = set(), []
        for title, ctx in extracts.items():
            if ctx[:80] not in seen_text:
                if _is_relevant(question, ctx):
                    seen_text.add(ctx[:80])
                    unique.append(ctx)
                    self._log(f"WIKI ACCEPTED    : [{title}]")
                else:
                    self._log(f"WIKI REJECTED    : [{title}] (relevance too low)")

        return self._score_and_combine(unique, question)

    def _get_wiktionary_context(self, question: str, options: dict, query=None) -> str:
        """Wiktionary-only lookup — best for word definitions and etymology."""
        if isinstance(query, dict):
            # Use title as the primary term, alternatives as extras
            terms = [query.get("title", "")] + query.get("alternatives", [])
            terms = [t.strip() for t in terms if t.strip()]
            self._log(f"WIKT LOOKUP      : {terms}  (llm-generated)")
        elif query:
            terms = [query]
            self._log(f"WIKT LOOKUP      : {terms}  (llm-generated)")
        else:
            terms = _quoted_terms(question)
        if not terms:
            # Fall back: capitalised words from question that might be terms
            words = re.findall(r"\b[a-zA-Z]+\b", question)
            terms = [
                w for i, w in enumerate(words)
                if i > 0 and w[0].isupper() and 2 < len(w) <= 25
                and w.lower() not in _STOP_WORDS
            ][:6]
        if not terms:
            self._log("WIKT LOOKUP      : (no terms found)")
            return ""

        self._log(f"WIKT LOOKUP      : {terms}")
        wikt_extracts = self._fetch_wiktionary(terms)

        unique, seen_text = [], set()
        for title, ctx in wikt_extracts.items():
            if ctx[:80] not in seen_text:
                seen_text.add(ctx[:80])
                unique.append(ctx)
                self._log(f"WIKT ACCEPTED    : [{title}]")

        if not unique:
            self._log("WIKT             : no entries found")
        return self._score_and_combine(unique, question)

    def get_context_for_tool(self, tool: str, question: str,
                             options: dict = None, query: str = None) -> str:
        """Route retrieval to a named tool, optionally using an LLM-generated query.

        tool  : 'none' | 'wikipedia' | 'wiktionary'
        query : LLM-written search string; when provided it replaces the automatic
                multi-query generation inside each tool method.
        Falls back to the full waterfall for unrecognised tool names.
        """
        options = options or {}
        if tool == "none":
            self._log("TOOL ROUTE       : memory (no retrieval)")
            return ""
        if tool == "wikipedia":
            self._log("TOOL ROUTE       : wikipedia")
            return self._get_wikipedia_context(question, options, query=query)
        if tool == "wiktionary":
            self._log("TOOL ROUTE       : wiktionary")
            return self._get_wiktionary_context(question, options, query=query)
        self._log(f"TOOL ROUTE       : unknown '{tool}' → waterfall fallback")
        return self.get_context(question, options)
