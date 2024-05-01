import numpy as np
from control.estimation.estimation_strategy import EstimationStrategy
from control.link_function.sigmoid import Sigmoid
from control.likelihood.logistic_log_likelihood import LogisticLogLikelihood
from control.likelihood.mvn_log_likelihood import MVNLogLikelihood


# Opimization class
class LogisticMAP(EstimationStrategy):
    def __init__(self, v_y, m_x, v_mu, m_sigma, s_max_itr, s_tolerance,
                 s_alpha):
        """
        Use gradient ascent method to obtain Maximum A Posteriori
        estimates for logistic regression coefficients. The prior
        distribution is multivariate normal distribution that has
        mean vector v_mu and covariance matrix m_sigma. Mean vector
        and covariance matrix need to be specified in the arguments.

        Args:
            v_y (np.array): Vector containing observed values of dependent
                            variable.
            m_x (np.array): Design matrix containing observed values of
                            independent variable.
            v_mu (np.array): Mean vector for prior MVN distribution.
            m_sigma (np.array): Covariance matrix for prior MVN distribution.
            s_max_itr (int): Maximum number of iteration for numerical
                             method.
            s_tolerance (float): Tolerance for log-likelihood change.
            s_alpha (float): Change rate for gradient ascent.
        """
        self.__v_y = v_y
        self.__m_x = m_x
        self.__v_mu = v_mu
        self.__m_sigma = m_sigma
        self.__s_max_itr = s_max_itr
        self.__s_tolerance = s_tolerance
        self.__s_alpha = s_alpha

    def estimate(self):
        """
        Use gradient ascent method to obtain Maximum A Posteriori
        estimates for logistic regression coefficients. The prior
        distribution is multivariate normal distribution that has
        mean vector v_mu and covariance matrix m_sigma. Mean vector
        and covariance matrix need to be specified in the arguments.

        Returns:
            list: List of Maximum A Posteriori estimates for logistic
                  regression coefficients, predictied probabilities and
                  log-likelihood.
        """
        output = []
        v_beta = np.zeros(self.__m_x.shape[1])
        v_p = Sigmoid(
            self.__m_x, v_beta).calculate_predicted_propability()
        s_ll_data_current = LogisticLogLikelihood(
            self.__v_y, self.__m_x, v_beta).calculate_log_likelihood()
        s_ll_prior_current = MVNLogLikelihood(
            v_beta, self.__v_mu, self.__m_sigma).calculate_log_likelihood()
        s_total_ll_current = s_ll_data_current + s_ll_prior_current

        for i in range(self.__s_max_itr):
            v_gradient = self.__calculate_gradient(v_p, v_beta)
            v_beta = self.__update_coefficients(v_beta, v_gradient)
            v_p = Sigmoid(self.__m_x, v_beta).calculate_predicted_propability()
            s_ll_data_next = LogisticLogLikelihood(
                self.__v_y, self.__m_x, v_beta).calculate_log_likelihood()
            s_ll_prior_next = MVNLogLikelihood(
                v_beta, self.__v_mu, self.__m_sigma).calculate_log_likelihood()
            s_total_ll_next = s_ll_data_next + s_ll_prior_next
            s_diff = np.abs(s_total_ll_next - s_total_ll_current)
            if s_diff < self.__s_tolerance:
                output.append(v_beta)
                output.append(v_p)
                output.append(s_total_ll_current)
                output.append(i)
                break
            s_total_ll_current = s_total_ll_next
        return output

    def __calculate_gradient(self, v_p, v_beta):
        v_gradient_data = self.__m_x.T @ (self.__v_y - v_p)
        v_gradient_prior = - np.linalg.inv(
            self.__m_sigma) @ (v_beta - self.__v_mu)
        return v_gradient_data + v_gradient_prior

    def __update_coefficients(self, v_beta, v_gradient):
        return v_beta + self.__s_alpha * v_gradient
