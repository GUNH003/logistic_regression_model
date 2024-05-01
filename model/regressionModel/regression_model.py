import numpy as np
from abc import ABC


class Regression(ABC):
    def __init__(self, DV, IV, dataframe) -> None:
        super().__init__()
        """
        Create an instance of Regression model. The dataframe has size n x p,
        where n is the total number of observations and p is the total number
        of variables.
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
        # Dataframe
        self.df = dataframe
        # Initialize dependent variable as a vector of n x 1
        self.v_y = self.df[DV]
        # Account for intercept
        self.v_ones = np.ones(self.df.shape[0])
        # Initialize independent variable as a matrix of n x k
        self.m_x = self.df[IV].values
        # Design matrix of size n x (k + 1)
        self.m_x = np.insert(self.m_x, 0, self.v_ones, axis=1)
        # Initialize coefficients
        self.v_beta = np.zeros(self.m_x.shape[1])
