import numpy as np
from log_likelihood import LogLikelihood
from link_function.sigmoid import Sigmoid


class LogisticLogLikelihood(LogLikelihood):
    def __init__(self, v_y, m_x, v_beta) -> None:
        super().__init__()
        self.__v_y = v_y
        self.__m_x = m_x
        self.__v_beta = v_beta
        self.__link_function = Sigmoid(self.__m_x, self.__v_beta)

    def calculate_log_likelihood(self):
        """
        Calculate the log-likelihood for all observed values of dependent
        variable in a logistic regression model, assuming that all observed
        values of dependent variable follow independent bernoulli
        distributions.

        Args:
            v_y (np.array): Vector containing observed values of dependent
                            variable.
            m_x (np.array): Design matrix containing observed values of
                            independent variable.
            v_beta (np.array): Vector containing parameters to estimate.

        Returns:
            float: Log-likelihood for the model.
        """
        # Calulate predictions
        v_p = self.__link_function.calculate_predicted_propability()
        # Create a vector of ones
        v_one = np.ones(self.__m_x.shape[0])
        # Calculate log-likelihood for each observation
        v_log_likelihood = np.multiply(self.__v_y, np.log(v_p)) + np.multiply(
            (v_one - self.__v_y), np.log(v_one - v_p))
        # Sum log-likelihood for all observations
        s_log_likelihood = np.sum(v_log_likelihood)
        return s_log_likelihood
