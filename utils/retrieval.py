import re
import warnings
from concurrent.futures import ThreadPoolExecutor, wait

warnings.filterwarnings("ignore", category=UserWarning, module="wikipedia")

_PARALLEL_TIMEOUT = 6.0
_MAX_CONTEXT_LEN  = 1500
_WIKI_SENTENCES   = 8   # richer summaries than the default 4

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

try:
    import wikipedia
    wikipedia.set_lang("en")
    wikipedia.set_rate_limiting(False)
    _WIKIPEDIA_AVAILABLE = True
except ImportError:
    _WIKIPEDIA_AVAILABLE = False


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
    """Proper nouns from question + distinctive terms from answer options.
    Helps surface articles specific to the answer domain, e.g. 'aspirated
    phoneme loss' for a Greek phonology question.
    """
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
    Used as fallback queries when the combined proper-noun search returns nothing.
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
    """At least 50% of proper nouns from the question must appear in the
    context.  Filters obviously off-topic articles (e.g. a Greece geography
    article returned for a Greek phonology question) while keeping anything
    with a genuine topical overlap.  Falls back to keyword overlap when the
    question has no proper nouns.
    """
    c_lower = context.lower()
    proper  = _proper_nouns(question)
    if proper:
        matches = sum(1 for p in proper if p in c_lower)
        return matches / len(proper) >= 0.5
    # No proper nouns — accept if any content keyword matches
    q_kw = set(_keywords(question))
    c_kw = set(_keywords(context))
    return bool(q_kw & c_kw)


class Retriever:
    """Wikipedia-only multi-query retrieval with relevance filtering.

    Fires three Wikipedia searches simultaneously:
      1. proper-nouns-only   e.g. 'Attic Greek Hellenistic'
      2. focused (nouns + content words)
      3. options-based (nouns + distinctive terms from the answer choices)

    Each result is relevance-checked (≥50% of question proper nouns must
    appear).  Results are deduplicated and joined with '---' separators.
    """

    def __init__(self, log_fn=None):
        self._log = log_fn or (lambda msg: None)

    def _search_wikipedia(self, query: str) -> tuple[str, str]:
        """Try each candidate title until one returns a summary."""
        try:
            titles = wikipedia.search(query, results=3)
        except Exception:
            return "", ""
        for title in titles:
            try:
                summary = wikipedia.summary(title, sentences=_WIKI_SENTENCES,
                                            auto_suggest=False)
                if summary:
                    return title, summary
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

        if not _WIKIPEDIA_AVAILABLE or not queries:
            self._log("SEARCH SOURCES   : Wikipedia not available")
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

        # Fallback: search each proper-noun phrase individually.
        # Handles cases like "Ancient Greek Koine Greek" where the combined
        # query matches no Wikipedia title but each phrase has its own article.
        if not unique:
            fallback_phrases = [
                p for p in _proper_noun_phrases(question)
                if p.lower() not in seen_q and len(p) > 2
            ]
            if fallback_phrases:
                self._log(f"FALLBACK PHRASES : {fallback_phrases}")
                fb_tasks = [(p, lambda p=p: self._search_wikipedia(p)) for p in fallback_phrases]
                with ThreadPoolExecutor(max_workers=max(1, len(fb_tasks))) as ex:
                    fb_future_map = {ex.submit(fn): p for p, fn in fb_tasks}
                    fb_done, _   = wait(fb_future_map.keys(), timeout=_PARALLEL_TIMEOUT)
                for fut in fb_done:
                    p = fb_future_map[fut]
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
                            self._log(f"FALLBACK RESULT  : (no article found for {p!r})")
                    except Exception as e:
                        self._log(f"FALLBACK ERROR   : {e}")

        if not unique:
            self._log("SEARCH COMBINED  : (none)")
            return ""

        # Sort so the most on-topic article leads — proper nouns weighted 2×
        # over generic keywords so the most specific match gets the first (and
        # largest) slice of the context window before the char limit truncates.
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
