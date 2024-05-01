import numpy as np
from model.regressionModel.regression_model import Regression


# Logistic regression model class
class LogisticRegression(Regression):
    def __init__(self, DV, IV, dataframe) -> None:
        """
        Create an instance of LogisticRegression model. The dataframe has
        size n x p, where n is the total number of observations and p is
        the total number of variables.
        The dependent variable is initialized as a vector of size n x 1.
        The independent variables are initialized as the design matrix of size
        n x (k + 1), where k is the number of independent variables and k + 1
        accounts for the intercept by inserting one column of ones at the 0th
        column.

        Args:
            DV (str): Column name of the dependent variable.
            IV (list): List of column names of the independent variables.
            dataframe (pd.dataframe): Dataframe containing DV and IVs.
        """
        super().__init__(DV, IV, dataframe)
        # Initialize change in odds
        self.v_change_in_odds = None
        # Initialize predicted probabilities
        self.v_p = None
        # Likelihood for binomial distribution
        self.s_log_likelihood = None
        # Hessian matrix
        self.m_hessian = np.zeros((len(IV), len(IV)))
        # SE for coefficients
        self.v_se = np.zeros(len(IV))
        # Wald statistics
        self.wald_stats = None
        # Akaike information criterion (AIC)
        self.s_aic = None
        # Number of iterations
        self.s_num_itr = None
        # Log-likelihood for each iterations
        self.v_log_likelihood = None
