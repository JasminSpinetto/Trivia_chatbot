import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from duckduckgo_search import DDGS
except ImportError:
    DDGS = None

class TfIdfBaseModel:
    def get_info(self):
        return {
            "Name": "TF-IDF Base", 
            "Description": "Matches answers to the question using Cosine Similarity on TF-IDF representation."
        }

    def answer(self, question: str, options: dict) -> int:
        option_ids = list(options.keys())
        option_texts = list(options.values())
        
        # The corpus consists of the question followed by the options
        corpus = [question] + option_texts
        vectorizer = TfidfVectorizer(stop_words='english')
        
        try:
            tfidf_matrix = vectorizer.fit_transform(corpus)
        except ValueError:
            # Fallback if text is empty or contains only stop words
            return int(random.choice(option_ids))
            
        question_vec = tfidf_matrix[0:1]
        options_vec = tfidf_matrix[1:]
        
        similarities = cosine_similarity(question_vec, options_vec)[0]
        max_sim = max(similarities)
        
        # Find all options that share the maximum similarity (handles ties)
        best_indices = [i for i, sim in enumerate(similarities) if sim == max_sim]
        
        # Select random if tied
        chosen_index = random.choice(best_indices)
        return int(option_ids[chosen_index])


class TfIdfWebModel(TfIdfBaseModel):
    def get_info(self):
        return {
            "Name": "TF-IDF Web", 
            "Description": "Retrieves top web search document for the question and matches answers to it using TF-IDF."
        }

    def get_web_document(self, query: str) -> str:
        if DDGS is None:
            print("duckduckgo_search not found. Falling back to the question text.")
            return query
            
        try:
            # Perform a lightweight web search using duckduckgo (requires no API Key)
            results = list(DDGS().text(query, max_results=1))
            if results and len(results) > 0:
                return results[0]['body']
            return query
        except Exception as e:
            print(f"Web search error: {e}")
            return query

    def answer(self, question: str, options: dict) -> int:
        # Search the web to retrieve a relevant document
        web_doc = self.get_web_document(question)
        
        # Create corpus with web doc as the baseline reference, instead of just the question
        return super().answer(web_doc, options)
        
        # NOTE: We can recycle the TfIdfBaseModel logic by passing the web document 
        # as the "question" since the comparison metric logic remains entirely the same.