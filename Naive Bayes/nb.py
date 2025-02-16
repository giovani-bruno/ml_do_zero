from typing import Set, Tuple, Dict, Iterable, NamedTuple
from collections import defaultdict
import re, math


class Message(NamedTuple):
    text: str
    is_spam: bool

def tokenize(text: str) -> Set[str]:
    text = text.lower()                         # Converta para minúsculas,
    all_words = re.findall("[a-z0-9']+", text)  # extraia as palavras
    return set(all_words)                       # e remova as duplicatas.

assert tokenize("Data Science is science") == {"data", "science", "is"}

class NaiveBayesClassifier:
    def __init__(self, k: float = 0.5) -> None:
        self.k = k  # fator de suavização

        self.tokens: Set[str] = set()
        self.token_spam_counts: Dict[str, int] = defaultdict(int)
        self.token_ham_counts: Dict[str, int] = defaultdict(int)
        self.spam_messages = self.ham_messages = 0

    def train(self, messages: Iterable[Message]) -> None:
        for message in messages:
            # Incremente as contagens de mensagens
            if message.is_spam:
                self.spam_messages += 1
            else:
                self.ham_messages += 1

            # Incremente as contagens de palavras
            for token in tokenize(message.text):
                self.tokens.add(token)
                if message.is_spam:
                    self.token_spam_counts[token] += 1
                else:
                    self.token_ham_counts[token] += 1

    def _probabilities(self, token: str) -> Tuple[float, float]:
        """Retorna P(token | spam) e P(token | not spam)"""
        spam = self.token_spam_counts[token]
        ham = self.token_ham_counts[token]

        p_token_spam = (spam + self.k) / (self.spam_messages + 2 * self.k)
        p_token_ham = (ham + self.k) / (self.ham_messages + 2 * self.k)

        return p_token_spam, p_token_ham

    def predict(self, text: str) -> float:
        text_tokens = tokenize(text)
        log_prob_if_spam = log_prob_if_ham = 0.0

        # Itere em cada palavra do vocubulário
        for token in self.tokens:
            prob_if_spam, prob_if_ham = self._probabilities(token)

            # Se o token aparecer na mensagem,
            # adicione o log da probabildiade de vê-lo
            if token in text_tokens:
                log_prob_if_spam += math.log(prob_if_spam)
                log_prob_if_ham += math.log(prob_if_ham)

            # se não, adicione o log da probabilidade de não vê-lo
            # que é log(1 - probabilidade de vê-lo).
            else:
                log_prob_if_spam += math.log(1.0 - prob_if_spam)
                log_prob_if_ham += math.log(1.0 - prob_if_ham)

        prob_if_spam = math.exp(log_prob_if_spam)
        prob_if_ham = math.exp(log_prob_if_ham)
        return prob_if_spam / (prob_if_spam + prob_if_ham)