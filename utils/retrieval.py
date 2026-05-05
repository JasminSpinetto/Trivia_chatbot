import html
import re
import requests
from concurrent.futures import ThreadPoolExecutor, wait

_PARALLEL_TIMEOUT = 6.0
_MAX_CONTEXT_LEN  = 3000
_WIKI_SENTENCES   = 15  # sentences per article summary

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
    content keyword overlaps.  The OR fallback prevents false rejections when
    a proper noun happens to be absent from the article's opening sentences.
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
    """Wikipedia retrieval via the MediaWiki REST API (no third-party library).

    Uses requests directly to avoid the silent failures and rate-limit
    fragility of the wikipedia Python library.  Each call:
      1. Searches for up to 5 candidate page titles.
      2. Fetches the intro extract (up to _WIKI_SENTENCES sentences) for
         the first title that returns content, following redirects automatically.
    """

    def __init__(self, log_fn=None):
        self._log = log_fn or (lambda msg: None)

    def _search_wikipedia(self, query: str) -> tuple[str, str]:
        """Return (title, intro_text) for the best matching Wikipedia article."""
        try:
            r = requests.get(
                _WIKI_API,
                params={"action": "query", "list": "search",
                        "srsearch": query, "srlimit": 5, "format": "json"},
                headers=_WIKI_HDRS,
                timeout=5,
            )
            r.raise_for_status()
            hits = r.json()["query"]["search"]
        except Exception as e:
            self._log(f"WIKI SEARCH ERR  : {query!r} → {e}")
            return "", ""

        for hit in hits:
            title = hit["title"]
            try:
                r2 = requests.get(
                    _WIKI_API,
                    params={"action": "query", "prop": "extracts",
                            "exintro": True, "exsentences": _WIKI_SENTENCES,
                            "titles": title, "redirects": 1, "format": "json"},
                    headers=_WIKI_HDRS,
                    timeout=5,
                )
                r2.raise_for_status()
                pages = r2.json()["query"]["pages"]
                for pid, page in pages.items():
                    if pid == "-1":
                        continue
                    raw = page.get("extract", "")
                    text = _strip_html(raw)
                    if text:
                        return page.get("title", title), text
            except Exception:
                continue

        return "", ""

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

        # Deduplicated query list
        seen_q, queries = set(), []
        for q in [proper_only, focused, opts_query]:
            if q and q.lower() not in seen_q:
                seen_q.add(q.lower())
                queries.append(q)

        if not queries:
            self._log("SEARCH SOURCES   : no queries generated")
            return ""

        tasks = [(q, lambda q=q: self._search_wikipedia(q)) for q in queries]
        self._log(f"SEARCH SOURCES   : {[f'wikipedia({q!r})' for q, _ in tasks]}")

        with ThreadPoolExecutor(max_workers=len(tasks)) as ex:
            future_map = {ex.submit(fn): q for q, fn in tasks}
            done, _    = wait(future_map.keys(), timeout=_PARALLEL_TIMEOUT)

        seen, unique = set(), []
        for fut in done:
            q = future_map[fut]
            try:
                title, ctx = fut.result()
                if ctx:
                    self._log(f"WIKI RESULT      : [{title}] {ctx[:120]}{'...' if len(ctx) > 120 else ''}")
                    if _is_relevant(question, ctx) and ctx[:80] not in seen:
                        seen.add(ctx[:80])
                        unique.append(ctx)
                        self._log(f"WIKI ACCEPTED    : [{title}]")
                    elif ctx:
                        self._log(f"WIKI REJECTED    : [{title}] (relevance too low)")
                else:
                    self._log(f"WIKI RESULT      : (no article found for {q!r})")
            except Exception as e:
                self._log(f"SEARCH ERROR     : {e}")

        # Fallback: individual proper-noun phrases + phrase×keyword combos in one
        # parallel batch.  Bare phrases handle "Ancient Greek / Koine Greek" splits;
        # combos handle generic nouns like "Greek" that need a content word to
        # disambiguate — e.g. "Greek" + "architecture" → "Ancient Greek architecture".
        if not unique:
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
                fb_tasks = [(q, lambda q=q: self._search_wikipedia(q)) for q in fallback_queries]
                with ThreadPoolExecutor(max_workers=max(1, len(fb_tasks))) as ex:
                    fb_future_map = {ex.submit(fn): q for q, fn in fb_tasks}
                    fb_done, _    = wait(fb_future_map.keys(), timeout=_PARALLEL_TIMEOUT)
                for fut in fb_done:
                    q = fb_future_map[fut]
                    try:
                        title, ctx = fut.result()
                        if ctx:
                            self._log(f"FALLBACK RESULT  : [{title}] {ctx[:120]}{'...' if len(ctx) > 120 else ''}")
                            if _is_relevant(question, ctx) and ctx[:80] not in seen:
                                seen.add(ctx[:80])
                                unique.append(ctx)
                                self._log(f"FALLBACK ACCEPTED: [{title}]")
                            else:
                                self._log(f"FALLBACK REJECTED: [{title}] (relevance too low)")
                        else:
                            self._log(f"FALLBACK RESULT  : (no article found for {q!r})")
                    except Exception as e:
                        self._log(f"FALLBACK ERROR   : {e}")

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
