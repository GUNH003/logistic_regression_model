from abc import ABC, abstractmethod


class EstimationStrategy(ABC):
    def __init__(self, s_max_itr, s_tolerance, s_alpha) -> None:
        self.__s_max_itr = s_max_itr
        self.__s_tolerance = s_tolerance
        self.__s_alpha = s_alpha

    @abstractmethod
    def estimate():
        pass
