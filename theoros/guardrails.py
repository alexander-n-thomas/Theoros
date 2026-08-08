import logging
import threading
from llm_guard import scan_prompt, scan_output
from llm_guard import input_scanners
from llm_guard import output_scanners


BAN_CODE_THRESHOLD = 0.5
GIBBERISH_THRESHOLD = 0.5
MALICIOUS_URLS_THRESHOLD = 0.5
PROMPT_INJECTION_THRESHOLD = 0.5
SENTIMENT_THRESHOLD = -0.1
TOKEN_LIMIT = 4096
TOXICITY_THRESHOLD = 0.5


logging.getLogger("llm_guard").setLevel(logging.INFO)


class SingletonMeta(type):
    """
    A thread-safe implementation of Singleton.
    """
    _instances = {}
    _lock: threading.Lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        with cls._lock:
            if cls not in cls._instances:
                instance = super().__call__(*args, **kwargs)
                cls._instances[cls] = instance
        return cls._instances[cls]


class PromptScanner(metaclass=SingletonMeta):
    def __init__(self):
        self.scanners = [
            input_scanners.BanCode(threshold=BAN_CODE_THRESHOLD),
            input_scanners.Gibberish(threshold=GIBBERISH_THRESHOLD),
            input_scanners.InvisibleText(),
            input_scanners.PromptInjection(threshold=PROMPT_INJECTION_THRESHOLD),
            input_scanners.TokenLimit(limit=TOKEN_LIMIT),
            input_scanners.Toxicity(threshold=TOXICITY_THRESHOLD),
        ]

    def scan_input(self, user_input: str) -> tuple[str, dict[str, bool], dict[str, float]]:
        sanitized_prompt, results_valid, results_score = scan_prompt(scanners=self.scanners, prompt=user_input, fail_fast=True)
        return sanitized_prompt, results_valid, results_score


class OutputScanner(metaclass=SingletonMeta):
    def __init__(self):
        self.scanners = [
            output_scanners.BanCode(threshold=BAN_CODE_THRESHOLD),
            output_scanners.Gibberish(threshold=GIBBERISH_THRESHOLD),
            output_scanners.MaliciousURLs(threshold=MALICIOUS_URLS_THRESHOLD),
            output_scanners.NoRefusalLight(),
            output_scanners.Sentiment(threshold=SENTIMENT_THRESHOLD),
            output_scanners.Toxicity(threshold=TOXICITY_THRESHOLD),
        ]

    def scan_output(self, prompt: str, model_output: str) -> tuple[str, dict[str, bool], dict[str, float]]:
        sanitized_output, results_valid, results_score = scan_output(scanners=self.scanners, prompt=prompt, output=model_output, fail_fast=True)
        return sanitized_output, results_valid, results_score