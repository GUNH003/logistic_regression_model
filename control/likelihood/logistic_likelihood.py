import numpy as np
from likelihood import Likelihood
from link_function.sigmoid import Sigmoid


class LogisticLikelihood(Likelihood):
    def __init__(self, v_y, m_x, v_beta) -> None:
        super().__init__()
        self.__v_y = v_y
        self.__m_x = m_x
        self.__v_beta = v_beta
        self.__link_function = Sigmoid(self.__m_x, self.__v_beta)

    def calculate_likelihood(self):
        """
        Calculate the likelihood for all observed values of dependent variable
        in a logistic regression model, assuming that all observed values of
        dependent variable follow independent bernoulli distributions.

        Args:
            v_y (np.array): Vector containing observed values of dependent
                            variable.
            m_x (np.array): Design matrix containing observed values of
                            independent variable.
            v_beta (np.array): Vector containing parameters to estimate.

        Returns:
            np.array: Likelihood vector.
        """
        # Calulate predictions
        v_p = self.__link_function.calculate_predicted_propability()
        # Calculate likelihood of each observation
        v_likelihood = np.multiply(
            np.power(v_p, self.__v_y),
            np.power((np.ones(self.__m_x.shape[0]) - v_p),
                     (np.ones(self.__m_x.shape[0]) - self.__v_y)))
        return v_likelihood
