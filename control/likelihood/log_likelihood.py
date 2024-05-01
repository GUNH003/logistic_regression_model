from abc import ABC, abstractmethod


# Log-Likelihood class
class LogLikelihood(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def calculate_log_likelihood(self):
        pass
