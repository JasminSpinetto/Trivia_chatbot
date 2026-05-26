import sys, re
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, ".")

from utils.retrieval import HybridDenseRetriever

r = HybridDenseRetriever()

QUESTIONS = [
    ("Salt tank ODE",
     "A tank initially contains a salt solution of 3 grams of salt dissolved in 100 liters of water. "
     "Salt water containing 0.02 grams of salt per liter of water is pumped in at 4 liters per minute. "
     "The drain is adjusted so that the amount of solution in the tank remains constant.",
     None),
    ("Practice quiz motivation",
     "A teacher believes that giving her students a practice quiz every week will motivate the students "
     "to study harder, thus improving their grades. She decides to do a study on this.",
     None),
    ("Stratified sampling",
     "A company has 1000 employees evenly distributed throughout five assembly plants. "
     "A sample of 80 employees is to be drawn for a survey.",
     None),
    ("Circle surface of revolution",
     "Let C be the circle in the yz-plane whose equation is (y - 3)^2 + z^2 = 1. "
     "If C is revolved around the z-axis, find the surface area of the resulting torus.",
     None),
]

for label, question, options in QUESTIONS:
    print(f"\n{'='*70}")
    print(f"QUESTION: {label}")
    print(f"  {question[:100]}")

    # --- Dense scores (before threshold) ---
    dense_query = f"query: {question}"
    embedding = r._encoder.encode(
        [dense_query], normalize_embeddings=True, convert_to_numpy=True
    ).astype("float32")

    print(f"\n  Dense top-5 per source (threshold={r._MIN_SCORE}):")
    for idx, passages in r._indexes:
        D, I = idx.search(embedding, k=5)
        for i, s in zip(I[0], D[0]):
            if 0 <= i < len(passages):
                snippet = passages[i][:80].replace('\n', ' ')
                flag = "PASS" if s >= r._MIN_SCORE else "fail"
                print(f"    [{flag}] score={s:.4f}  {snippet}")

    # --- BM25 top-5 ---
    bm25_text   = re.sub(r'\$\$?[^$]+\$\$?', ' ', question)
    bm25_tokens = re.findall(r'\b[a-z0-9]{3,}\b', bm25_text.lower())
    bm25_scores = r._bm25.get_scores(bm25_tokens)
    top5        = bm25_scores.argsort()[::-1][:5]
    print(f"\n  BM25 top-5 (tokens: {bm25_tokens[:8]}):")
    for i in top5:
        snippet = r._bm25_passages[i][:80].replace('\n', ' ')
        print(f"    score={bm25_scores[i]:.3f}  {snippet}")

    # --- Final context returned ---
    ctx = r.get_context(question, options)
    print(f"\n  Final context ({len(ctx)} chars):")
    print(f"    {ctx[:200].replace(chr(10), ' ')}")

print("\nDone.")
