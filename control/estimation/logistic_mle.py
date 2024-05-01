import numpy as np
from control.estimation.estimation_strategy import EstimationStrategy
from control.link_function.sigmoid import Sigmoid
from control.likelihood.logistic_log_likelihood import LogisticLogLikelihood


# Opimization class
class LogisticMLE(EstimationStrategy):
    def __init__(self, v_y, m_x, s_max_itr, s_tolerance, s_alpha):
        """
        Use gradient ascent method to obtain Maximum Likelihood Estimates for
        logistic regression coefficients.

        Args:
            v_y (np.array): Vector containing observed values of dependent
                            variable.
            m_x (np.array): Design matrix containing observed values of
                            independent variable.
            s_max_itr (int): Maximum number of iteration for numerical
                            method.
            s_tolerance (float): Tolerance for log-likelihood change.
            s_alpha (float): Change rate for gradient ascent.
        """
        self.__v_y = v_y
        self.__m_x = m_x
        self.__s_max_itr = s_max_itr
        self.__s_tolerance = s_tolerance
        self.__s_alpha = s_alpha

    def estimate(self):
        """
        Use gradient ascent method to obtain Maximum Likelihood Estimates for
        logistic regression coefficients.

        Returns:
            list: List of Maximum Likelihood Estimates for logistic regression
                coefficients, predictied probabilities, log-likelihood and
                number of iterations.
        """
        output = []
        v_beta = np.zeros(self.__m_x.shape[1])
        v_p = Sigmoid(self.__m_x, v_beta).calculate_predicted_propability()
        s_ll_current = LogisticLogLikelihood(
            self.__v_y, self.__m_x, v_beta).calculate_log_likelihood()
        for i in range(self.__s_max_itr):
            v_gradient = self.__calculate_gradient(v_p)
            v_beta = self.__update_coefficients(v_beta, v_gradient)
            v_p = Sigmoid(
                self.__m_x, v_beta).calculate_predicted_propability()
            s_ll_next = LogisticLogLikelihood(
                self.__v_y, self.__m_x, v_beta).calculate_log_likelihood()
            if np.abs(s_ll_next - s_ll_current) < self.__s_tolerance:
                output.append(v_beta)
                output.append(v_p)
                output.append(s_ll_current)
                output.append(i)
                break
            s_ll_current = s_ll_next
        return output

    def __calculate_gradient(self, v_p):
        """Helper method. Calculates gradient.

        Args:
            v_p (np.array): vector of predicted probability

        Returns:
            np.array: gradient vector
        """
        return self.__m_x.T @ (self.__v_y - v_p)

    def __update_coefficients(self, v_beta, v_gradient):
        """Helper method. Updates model parameters.

        Args:
            v_beta (np.array): model parameter vector
            v_gradient (np.array): gradient vector

        Returns:
            np.array: updated model parameter vector
        """
        return v_beta + self.__s_alpha * v_gradient
