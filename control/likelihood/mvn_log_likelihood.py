import numpy as np
from log_likelihood import LogLikelihood


class MVNLogLikelihood(LogLikelihood):
    def __init__(self, v_x, v_mu, m_sigma) -> None:
        super().__init__()
        self.__v_x = v_x
        self.__v_mu = v_mu
        self.__m_sigma = m_sigma

    def calculate_log_likelihood(self):
        """
        Calculate the log-likelihood for a single instance of multiple
        variables with given mean vector v_mu and covariance matrix m_sigma,
        assuming the variables follow multivariate normal distribution.

        Args:
            v_x (np.array): A set values for variables that follow the MVN.
            v_mu (np.array): Mean vector of the MVN.
            m_sigma (np.array): Covariance matrix of the MVN.

        Returns:
            float: Log-likelihood for a single instance of multiple variables.
        """
        # Calculate number of variables
        s_k = self.__v_x.shape[0]
        # Calculate log-likelihood
        s_const = (- s_k / 2) * np.log(2 * np.pi)
        s_part_1 = (-0.5) * np.log(np.linalg.det(self.__m_sigma))
        s_part_2 = -0.5 * (self.__v_x - self.__v_mu).T @ np.linalg.inv(
            self.__m_sigma) @ (self.__v_x - self.__v_mu)
        s_log_likelihood = s_const + s_part_1 + s_part_2
        return s_log_likelihood
