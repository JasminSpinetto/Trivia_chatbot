import html
import re
import requests
from concurrent.futures import ThreadPoolExecutor, wait
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    from duckduckgo_search import DDGS
    _DDGS_AVAILABLE = True
except ImportError:
    _DDGS_AVAILABLE = False

# Extract Wikipedia article title from a Wikipedia URL
_WIKI_URL_RE = re.compile(r"en\.wikipedia\.org/wiki/([^#?&]+)")

_PARALLEL_TIMEOUT = 5.0   # per search-phase budget; extract batch is one extra call
_MAX_CONTEXT_LEN  = 3000
_WIKI_SENTENCES   = 15    # sentences per article summary
_MAX_WORKERS      = 4     # concurrent search calls (I/O-bound, safe for Colab)

_WIKI_API  = "https://en.wikipedia.org/w/api.php"
_WIKI_HDRS = {"User-Agent": "PoliMillionaire-quiz/1.0 (educational NLP project)"}

# Questions that refer to a specific competition passage — Wikipedia can't help
_ARTICLE_PATTERN = re.compile(
    r"\b(according to (the |this )?(article|passage|text|excerpt)|"
    r"(the|this) (article|passage|text) (states?|says?|mentions?|describes?|claims?)|"
    r"(as|as described|as stated) in (the|this) (article|passage|text)|"
    r"based on (the|this) (article|passage|text)|"
    r"(from|in) (the|this) (following |above )?(article|passage|text|excerpt))\b",
    re.IGNORECASE,
)

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
    content keyword overlaps.
    """
    c_lower = context.lower()
    proper  = _proper_nouns(question)
    if proper:
        matches = sum(1 for p in proper if p in c_lower)
        if matches / len(proper) >= 0.5:
            return True
    q_kw = set(_keywords(question))
    c_kw = set(_keywords(context))
    return bool(q_kw & c_kw)


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
            backoff_factor=0.2,      # one retry after 0.2s
            status_forcelist=[429, 500, 502, 503, 504],
            respect_retry_after_header=False,
        )
        self._session.mount("https://", HTTPAdapter(max_retries=_retry))
        self._session.headers.update(_WIKI_HDRS)

    def _search_titles(self, query: str):
        """Return up to 5 Wikipedia page titles, or None on error.

        Returning None (vs empty list) lets callers distinguish a network/
        rate-limit failure from a query that simply matched no articles.
        """
        try:
            r = self._session.get(
                _WIKI_API,
                params={"action": "query", "list": "search",
                        "srsearch": query, "srlimit": 5, "format": "json"},
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

    def _run_queries(self, queries: list) -> tuple:
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
            future_map = {ex.submit(self._search_titles, q): q for q in queries}
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

    def _search_ddgs(self, query: str, max_results: int = 6) -> list:
        """DuckDuckGo text search — no API key, no hard rate limit.

        Returns raw result dicts: [{"title": ..., "href": ..., "body": ...}]
        """
        if not _DDGS_AVAILABLE:
            return []
        try:
            return list(DDGS().text(query, max_results=max_results)) or []
        except Exception as e:
            self._log(f"DDGS ERROR       : {e}")
            return []

    def get_context(self, question: str, options: dict = None) -> str:
        if options is None:
            options = {}

        # Questions about a specific competition passage — Wikipedia can't help
        if _ARTICLE_PATTERN.search(question):
            self._log("PASSAGE QUESTION : skipping retrieval (passage not publicly available)")
            return ""

        proper_only  = " ".join(
            re.sub(r"'s?$", "", w)
            for i, w in enumerate(re.findall(r"\b[a-zA-Z']+\b", question))
            if i > 0 and w[0].isupper() and len(w) > 2
        )
        focused      = _focused_query(question)
        opts_query   = _options_query(question, options)
        keywords     = _keywords(question)

        self._log(f"SEARCH PROPER    : {proper_only!r}")
        self._log(f"SEARCH FOCUSED   : {focused!r}")
        self._log(f"SEARCH OPTIONS   : {opts_query!r}")
        self._log(f"SEARCH KEYWORDS  : {keywords}")

        # Deduplicated primary query list
        seen_q, queries = set(), []
        for q in [proper_only, focused, opts_query]:
            if q and q.lower() not in seen_q:
                seen_q.add(q.lower())
                queries.append(q)

        if not queries:
            self._log("SEARCH SOURCES   : no queries generated")
            return ""

        self._log(f"SEARCH SOURCES   : {[f'wikipedia({q!r})' for q in queries]}")
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

        # Phrase/combo fallback — skipped when Wikipedia is erroring (429, connection
        # refused, etc.) since those queries would also fail. Go straight to DDGS.
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
                            self._log(f"FALLBACK ACCEPTED: [{title}]")
                        else:
                            self._log(f"FALLBACK REJECTED: [{title}] (relevance too low)")

        # Last resort: DuckDuckGo.
        # Strategy: DDGS finds Wikipedia article URLs (no search-API rate limit) →
        # extract titles → fetch full Wikipedia extracts in one batch call.
        # If no Wikipedia URLs appear, use raw DDGS snippets instead.
        if not unique and _DDGS_AVAILABLE:
            ddgs_query = focused or proper_only or " ".join(keywords[:4])
            self._log(f"DDGS SEARCH      : {ddgs_query!r}")
            ddgs_results = self._search_ddgs(ddgs_query)

            wiki_titles = []
            seen_wiki = set()
            for r in ddgs_results:
                m = _WIKI_URL_RE.search(r.get("href", ""))
                if m:
                    title = m.group(1).replace("_", " ")
                    if title not in seen_wiki:
                        seen_wiki.add(title)
                        wiki_titles.append(title)

            if wiki_titles:
                self._log(f"DDGS WIKI TITLES : {wiki_titles}")
                ddgs_extracts = self._fetch_extracts(wiki_titles)
                for title, ctx in ddgs_extracts.items():
                    if ctx[:80] not in seen_text and _is_relevant(question, ctx):
                        seen_text.add(ctx[:80])
                        unique.append(ctx)
                        self._log(f"DDGS WIKI ACCEPT : [{title}]")
                    elif ctx:
                        self._log(f"DDGS WIKI REJECT : [{title}] (relevance too low)")

            # If Wikipedia URLs gave nothing, use raw DDGS snippets as context
            if not unique:
                for r in ddgs_results:
                    snippet = r.get("body", "").strip()
                    if snippet and snippet[:80] not in seen_text and _is_relevant(question, snippet):
                        seen_text.add(snippet[:80])
                        unique.append(snippet)
                        self._log(f"DDGS SNIP ACCEPT : [{r.get('title', '')}]")

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

        combined = "\n\n---\n\n".join(unique)[:_MAX_CONTEXT_LEN]
        self._log(f"SEARCH COMBINED  : {len(unique)} source(s), {len(combined)} chars")
        return combined
