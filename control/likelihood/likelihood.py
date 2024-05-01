from abc import ABC, abstractmethod


# Likelihood class
class Likelihood(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def calculate_likelihood(self):
        pass
