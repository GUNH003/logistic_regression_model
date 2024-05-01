import numpy as np
from control.likelihood.logistic_log_likelihood import LogisticLogLikelihood
from control.likelihood.mvn_log_likelihood import MVNLogLikelihood


class Metropolis:
    def __init__(self, v_y, m_x, starting_mean, prior_mean, prior_sigma,
                 proposal_sigma, num_itr, num_burn_in):
        """
        User interface that carry out Metropolis algorithm to estimate the
        prior distribution of the logsitic regression coefficients.

        Args:
            v_y (np.array): Dependent variable, used to calcualte likelihood.
            m_x (np.array): Independent variable(s), used to calcualte
                            likelihood.
            starting_mean (np.array): Initial state of the markov chain.
            prior_mean (np.array): Mean for prior distribution.
            prior_sigma (np.array): Covariance matrix for prior distribution.
            proposal_sigma (np.array): Covariance matrix for proposal
                                       distribution.
            num_itr (int): Number of total iterations of sampling.
            num_burn_in (int): Number of burn-in iterations of sampling.
            discard (bool, optional): True=return all samples. False=return
                                      valid
            samples. Defaults to True.
        """
        self.__v_y = v_y
        self.__m_x = m_x
        self.__starting_mean = starting_mean
        self.__prior_mean = prior_mean
        self.__prior_sigma = prior_sigma
        self.__proposal_sigma = proposal_sigma
        self.__num_itr = num_itr
        self.__num_burn_in = num_burn_in

    def optimize(self):
        """
        User interface that carry out Metropolis algorithm to estimate the
        prior distribution of the logsitic regression coefficients.

        Returns:
            list: List of sample points from the stationary distribution of the
            markov chain.
        """
        samples = [self.__starting_mean]
        for i in range(self.__num_itr):
            # Calcualte the inital log-likelihood, which is the sum of the
            # log-likelihood of the data distribution and the log-likelihood
            # of the prior distribution based on the current data and the
            # assumption about the prior distribution
            current_log_likelihood = self.__calculate_log_likelihood(
                samples[-1])
            # Randomly choose the next candidate for the Markov Chain
            candidate = np.random.multivariate_normal(samples[-1],
                                                      self.__proposal_sigma)
            # Calculate the candidate's log-likelihood
            candidate_log_likelihood = self.__calculate_log_likelihood(
                candidate)
            # Calculate the acceptance probability based on the
            # "detailed balance" condition, with the assumption that the
            # proposal distribution is a symmetric multi-variate normal
            # distribution
            acceptance_probability = self.__calculate_acceptance_probability(
                current_log_likelihood, candidate_log_likelihood)
            # Update the current parameter to the candidate, if "detailed
            # balance" condition is reached
            samples.append(self.__update_samples(
                acceptance_probability, candidate, samples[-1]))
        # Discard the burn-in samples
        valid_samples = samples[self.__num_burn_in:]
        # Calculate the mean vector
        mean = np.average(valid_samples, axis=0)
        # Calculate the covariance matrix of the samples
        cov = np.cov(valid_samples, rowvar=False)
        return [samples, valid_samples, mean, cov]

    def __calculate_log_likelihood(self, sample):
        logisticLogLikelihood = LogisticLogLikelihood(
            self.__v_y,
            self.__m_x,
            sample).calculate_log_likelihood()
        mvnLogLikelihood = MVNLogLikelihood(
            sample,
            self.__prior_mean,
            self.__prior_sigma).calculate_log_likelihood()
        return logisticLogLikelihood + mvnLogLikelihood

    def __calculate_acceptance_probability(
            self, current_log_likelihood, candidate_log_likelihood):
        return min(1,
                   np.exp(candidate_log_likelihood - current_log_likelihood))

    def __update_samples(
            self, acceptance_probability, candidate, current_sample):
        u = np.random.uniform(0.0, 1.0)
        return candidate if u <= acceptance_probability else current_sample
