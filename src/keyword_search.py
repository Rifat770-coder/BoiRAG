import re
from collections import Counter, defaultdict
from math import log


def search_terms(text):
    # Keep Bengali vowel signs and virama attached to their words.
    for word in re.findall(r"[a-z0-9\u0980-\u09ff]+", text.casefold()):
        yield word
        for offset in range(len(word) - 3):
            yield f"~{word[offset:offset + 4]}"


class KeywordIndex:
    def __init__(self, documents):
        self.documents = documents
        self.postings = defaultdict(list)
        for position, document in enumerate(documents):
            for term, count in Counter(search_terms(document.page_content)).items():
                self.postings[term].append((position, count))

    def search(self, question, *, k):
        scores = Counter()
        for term in sorted(set(search_terms(question))):
            matches = self.postings.get(term, ())
            weight = log((len(self.documents) + 1) / (len(matches) + 1))
            if term.startswith("~"):
                weight *= 0.25
            for position, count in matches:
                scores[position] += weight * count / (count + 1)
        ranked = sorted(scores, key=lambda position: (-scores[position], position))
        return [self.documents[position] for position in ranked[:k] if scores[position] > 0]
