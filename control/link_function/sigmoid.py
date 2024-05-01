import numpy as np


# Sigmoid function class
class Sigmoid:
    def __init__(self, m_x, v_beta) -> None:
        self.__m_x = m_x
        self.__v_beta = v_beta

    def __sigmoid(self, m_x, v_beta):
        """
        Sigmoid function used in logistic regression. Calculate the
        predicted probability vector by performing element-wise
        operation using broadcasting.

        Returns:
            np.array: Vector containing predicted probability for each
                      observation.
        """
        # Linear combination of pairwise product
        # between IVs obsered value and coefficient
        v_z = m_x @ v_beta
        # Apply sigmoid function using broadcasting
        v_predicted_probability = 1 / (1 + np.exp(-v_z))
        return v_predicted_probability

    def calculate_predicted_propability(self):
        return self.__sigmoid(self.__m_x, self.__v_beta)
