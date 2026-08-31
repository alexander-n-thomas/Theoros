import logging
import threading
from llm_guard import scan_prompt, scan_output
from llm_guard import input_scanners
from llm_guard import output_scanners
from llm_guard.input_scanners.base import Scanner as InputScanner
from llm_guard.output_scanners.base import Scanner as OutputScanner
import nltk

BAN_CODE_THRESHOLD = 0.5
GIBBERISH_THRESHOLD = 0.5
MALICIOUS_URLS_THRESHOLD = 0.5
PROMPT_INJECTION_THRESHOLD = 0.5
SENTIMENT_THRESHOLD = -0.1
TOKEN_LIMIT = 4096
TOXICITY_THRESHOLD = 0.5
RELEVANCY_THRESHOLD = 0.5

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')


class RelevancyInput(InputScanner):

    def __init__(self, threshold: float = 0.75):
        import importlib
        import joblib
        model_path = importlib.resources.files("theoros.models.relevancy").joinpath("relevancy.joblib")
        self.model = joblib.load(model_path)
        self.threshold = threshold

    def scan(self, prompt: str) -> tuple[str, bool, float]:
        from nltk.tokenize import sent_tokenize
        sentences = sent_tokenize(prompt)
        pred_proba = self.model.predict_proba(sentences)[:, 1]
        return prompt, pred_proba.mean() > self.threshold, 1-pred_proba.mean()


class RelevancyOutput(OutputScanner):

    def __init__(self, threshold: float = 0.75):
        import importlib
        import joblib
        model_path = importlib.resources.files("theoros.models.relevancy").joinpath("relevancy.joblib")
        self.model = joblib.load(model_path)
        self.threshold = threshold

    def scan(self, prompt: str, output: str) -> tuple[str, bool, float]:
        from nltk.tokenize import sent_tokenize
        sentences = sent_tokenize(output)
        pred_proba = self.model.predict_proba(sentences)[:, 1]
        return prompt, pred_proba.mean() > self.threshold, 1-pred_proba.mean()


class PromptScanner():
    def __init__(self):
        self.scanners = [
            input_scanners.BanCode(threshold=BAN_CODE_THRESHOLD),
            # input_scanners.Gibberish(threshold=GIBBERISH_THRESHOLD),
            input_scanners.InvisibleText(),
            input_scanners.PromptInjection(threshold=PROMPT_INJECTION_THRESHOLD),
            input_scanners.TokenLimit(limit=TOKEN_LIMIT),
            input_scanners.Toxicity(threshold=TOXICITY_THRESHOLD),
            # RelevancyInput(threshold=RELEVANCY_THRESHOLD),
        ]

    def scan_input(self, user_input: str) -> tuple[str, dict[str, bool], dict[str, float]]:
        sanitized_prompt, results_valid, results_score = scan_prompt(scanners=self.scanners, prompt=user_input, fail_fast=True)
        return sanitized_prompt, results_valid, results_score


class OutputScanner():
    def __init__(self):
        self.scanners = [
            output_scanners.BanCode(threshold=BAN_CODE_THRESHOLD),
            # output_scanners.Gibberish(threshold=GIBBERISH_THRESHOLD),
            output_scanners.NoRefusalLight(),
            output_scanners.Sentiment(threshold=SENTIMENT_THRESHOLD),
            output_scanners.Toxicity(threshold=TOXICITY_THRESHOLD),
            # RelevancyOutput(threshold=RELEVANCY_THRESHOLD),
        ]

    def scan_output(self, prompt: str, model_output: str) -> \
            tuple[str, dict[str, bool], dict[str, float]]:
        sanitized_output, results_valid, results_score = scan_output(scanners=self.scanners, prompt=prompt, output=model_output, fail_fast=True)
        return sanitized_output, results_valid, results_score