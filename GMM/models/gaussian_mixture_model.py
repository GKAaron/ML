"""Implements the Gaussian Mixture model, and trains using EM algorithm."""
import numpy as np
import scipy
from scipy.stats import multivariate_normal


class GaussianMixtureModel(object):
    """Gaussian Mixture Model"""
    def __init__(self, n_dims, n_components=1,
                 max_iter=10,
                 reg_covar=1e-6):
        """
        Args:
            n_dims: The dimension of the feature.
            n_components: Number of Gaussians in the GMM.
            max_iter: Number of steps to run EM.
            reg_covar: Amount to regularize the covariance matrix, (i.e. add
                to the diagonal of covariance matrices).
        """
        self._n_dims = n_dims
        self._n_components = n_components
        self._max_iter = max_iter
        self._reg_covar = reg_covar

        # Randomly Initialize model parameters
        self._mu = np.random.uniform(0,1,
        (self._n_components,self._n_dims)) # np.array of size (n_components, n_dims)
        # Initialized with uniform distribution.
        self._pi = np.random.uniform(0,1,  # np.array of size (n_components, 1)
        (self._n_components,1)) # np.array of size (n_components, n_dims)

        # Initialized with identity.
        self._sigma = 100*np.array([np.identity(self._n_dims,dtype='float64') 
        for i in range(self._n_components)])  # np.array of size (n_components, n_dims, n_dims)

    def fit(self, x):
        """Runs EM steps.

        Runs EM steps for max_iter number of steps.

        Args:
            x(numpy.ndarray): Feature array of dimension (N, ndims).
        """
        ran = np.random.randint(0,x.shape[0],self._n_components)
        self._mu = x[ran]
        for i in range(self._max_iter):
            z_ik = self._e_step(x)
            self._m_step(x,z_ik)


    def _e_step(self, x):
        """E step.

        Wraps around get_posterior.

        Args:
            x(numpy.ndarray): Feature array of dimension (N, ndims).
        Returns:
            z_ik(numpy.ndarray): Array containing the posterior probability
                of each example, dimension (N, n_components).
        """
        return self.get_posterior(x)

    def _m_step(self, x, z_ik):
        """M step, update the parameters.

        Args:
            x(numpy.ndarray): Feature array of dimension (N, ndims).
            z_ik(numpy.ndarray): Array containing the posterior probability
                of each example, dimension (N, n_components).
                (Alternate way of representing categorical distribution of z_i)
        """
        # Update the parameters.
        z_k = np.sum(z_ik,axis=0)
        N = x.shape[0]
        pi = z_k/N
        mu = []
        cov_m = []
        for i in range(self._n_components):
            mu.append(np.sum(z_ik[:,i:i+1]*x,axis=0)/(N*pi[i]))
            X =  x-mu[i].reshape((1,-1))
            cov = np.dot(X.T,X*z_ik[:,i:i+1])/(N*pi[i])
            cov= cov+np.identity(cov.shape[0])*self._reg_covar
            cov_m.append(cov)
        self._pi = pi.reshape((-1,1))
        self._mu = np.array(mu)
        self._sigma = np.array(cov_m)

    def get_conditional(self, x):
        """Computes the conditional probability.

        p(x^(i)|z_ik=1)

        Args:
            x(numpy.ndarray): Feature array of dimension (N, ndims).
        Returns:
            ret(numpy.ndarray): The conditional probability for each example,
                dimension (N,, n_components).
        """
        ret = []
        for i in range(self._n_components):
            ret.append(self._multivariate_gaussian(x,self._mu[i],
            self._sigma[i]))
        return np.array(ret).T

    def get_marginals(self, x):
        """Computes the marginal probability.

        p(x^(i)|pi, mu, sigma)

        Args:
             x(numpy.ndarray): Feature array of dimension (N, ndims).
        Returns:
            (1) The marginal probability for each example, dimension (N,).
        """
        ret = self.get_conditional(x)
        ret = ret*self._pi.T
        return np.sum(ret,axis=1)

    def get_posterior(self, x):
        """Computes the posterior probability.

        p(z_{ik}=1|x^(i))

        Args:
            x(numpy.ndarray): Feature array of dimension (N, ndims).
        Returns:
            z_ik(numpy.ndarray): Array containing the posterior probability
                of each example, dimension (N, n_components).
        """
        ret = self.get_conditional(x)
        ret = ret*self._pi.T
        p = self.get_marginals(x).reshape((-1,1))
        z_ik = ret/p
        return z_ik

    def _multivariate_gaussian(self, x, mu_k, sigma_k):
        """Multivariate Gaussian, implemented for you.
        Args:
            x(numpy.ndarray): Array containing the features of dimension (N,
                ndims)
            mu_k(numpy.ndarray): Array containing one single mean (ndims,1)
            sigma_k(numpy.ndarray): Array containing one signle covariance matrix
                (ndims, ndims)
        """
        return multivariate_normal.pdf(x, mu_k, sigma_k)

    def supervised_fit(self, x, y):
        """Assign each cluster with a label through counting.
        For each cluster, find the most common digit using the provided (x,y)
        and store it in self.cluster_label_map.
        self.cluster_label_map should be a list of length n_components,
        where each element maps to the most common digit in that cluster.
        (e.g. If self.cluster_label_map[0] = 9. Then the most common digit
        in cluster 0 is 9.
        Args:
            x(numpy.ndarray): Array containing the feature of dimension (N,
                ndims).
            y(numpy.ndarray): Array containing the label of dimension (N,)
        """
        self.fit(x)
        z_ik = self.get_posterior(x)
        self.cluster_label_map = []
        p = z_ik*self._pi.reshape((1,-1))
        l = np.zeros_like(z_ik)
        p = np.argmax(p,axis=1)
        l[range(0,x.shape[0]),p]=1
        for i in range(x.shape[0]):
            l[i][l[i]==1]=y[i]
        for i in range(self._n_components):
            la,po = np.unique(l[:,i],return_counts=True)
            if len(la)==1 and la[0]==0:
                self.cluster_label_map.append(np.random.randint(0,
                np.max(np.unique(y))))
            else:
                self.cluster_label_map.append(la[np.argmax(po)])
        

    def supervised_predict(self, x):
        """Predict a label for each example in x.
        Find the get the cluster assignment for each x, then use
        self.cluster_label_map to map to the corresponding digit.
        Args:
            x(numpy.ndarray): Array containing the feature of dimension (N,
                ndims).
        Returns:
            y_hat(numpy.ndarray): Array containing the predicted label for each
            x, dimension (N,)
        """

        z_ik = self.get_posterior(x)
        p = z_ik*self._pi.reshape((1,-1))
        p = np.argmax(p,axis=1)
        y_hat = []
        for i in range(x.shape[0]):
            y_hat.append(self.cluster_label_map[p[i]])
        return np.array(y_hat)
