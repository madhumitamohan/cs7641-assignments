import numpy as np
from numpy.linalg import LinAlgError
from tqdm import tqdm

SIGMA_CONST = 1e-06
LOG_CONST = 1e-20


class GMM(object):
    def __init__(self, X, K, max_iters=100, seed=5):
        """
        Args:
            X: the observations/datapoints, N x D numpy array
            K: number of clusters/components
            max_iters: maximum number of iterations (used in EM implementation)
            seed: value for seeding numpy.random
        """
        self.points = X
        self.max_iters = max_iters
        self.N = self.points.shape[0]
        self.D = self.points.shape[1]
        self.K = K
        self.num_iters = 0
        self.seed = seed

    @staticmethod
    def softmax(logit):
        """
        Performs softmax on the logits. The formula is in the notebook.

        Args:
            logit: np.ndarray(N,D), unbounded model outputs
        Return:
            prob: np.ndarray(N,D), transformed logits that can be interpreted as probabilities
        Hints:
            keepdims=True means that instead of collapsing the axis dimension, it will leave that dimension as size 1.
            E.g., np.prod(logit, axis=1).shape will result in (N,)
            while np.prod(logit, axis=1, keepdims=True).shape will result in (N,1)
            This will save you a line if you need to broadcast over the dimension you just collapsed.
        """

        stable_values = logit - np.max(logit, axis=1, keepdims= True)
        prob = np.divide(np.exp(stable_values), np.sum(np.exp(stable_values), axis = 1, keepdims=True))
        return prob

    @staticmethod
    def logsumexp(logit):
        """
        Performs logsumexp on the logits. The formula is in the notebook.

        Args:
            logit: np.ndarray(N,D), unbounded model outputs
        Return:
            s: np.ndarray(N,), a smoother version of max on the logits
        Hints:
            Refer to the hint for softmax.
        """
        max_values =  np.max(logit, axis= 1, keepdims=True)
        stable_values = logit - max_values
        s_inter = np.log(np.sum(np.exp(stable_values), axis = 1, keepdims= True))
        s = s_inter + max_values
        return s.squeeze() #To convert dimension from N,1 to N,

    @staticmethod
    def multinormalPDF(points, mu, sigma):
        """
        This function models the probability density function of a Gaussian.
        This function should output the density for each input datapoint.
        Note that this is the multivariate version of Gaussian.

        Args:
            points: np.ndarray(N,D), the points for which you will calculate the probability density
            mu: np.ndarray(D,), the mean of the Gaussian being modelled
            sigma: np.ndarray(D,D), the covariance matrix of the Gaussian being modelled
        Return:
            pdf: (N,) numpy array, the probability densities of the given points under the given Gaussian
        Note:
            sigma may not be invertible. Before doing anything, you should try to invert it.
            If that throws a LinAlgError, you should add SIGMA_CONST to the matrix.
        Hints:
            1. Use np.linalg.det() and np.linalg.inv(). Don't code your own linalg package.
            2. This is a static method; you should compute over the points given, not those in self.points.
        """
        covariance = sigma
        try:
            inv = np.linalg.inv(covariance)
        except LinAlgError:
            covariance = sigma + SIGMA_CONST
            inv = np.linalg.inv(covariance)

        D = points.shape[1]
        denom = np.sqrt(np.power(2 * np.pi, D) * np.linalg.det(covariance))
        constant = 1/denom

        x_minus_mu = points-mu
        exp_part_1 = x_minus_mu @ inv

        exp_part_2 = - (np.sum(exp_part_1 * x_minus_mu, axis = 1) /2)

        pdf = constant * np.exp(exp_part_2)
        return pdf


    def create_pi(self):
        """
        Initialize the mixing coefficients for each component.
        A default implementation could just be uniform probabilities.
        You can use whatever tricks you'd like, we will only test the shape.

        Return:
            pi: np.ndarray(K), the mixing coefficients
        """
        pi = np.ones(self.K) / self.K
        return pi

    def create_mu(self):
        """
        Initialize the random centers for each component.
        A default implementation could just be randomly choosing K points, similar to our naive choice in kmeans.
        You can use whatever tricks you'd like, we will only test the shape.

        Return:
            mu: np.ndarray(K,D), the Gaussians's means
        """
        indices = np.random.choice(self.N, self.K, replace=False)
        return self.points[indices]

    def create_sigma(self):
        """
        Initialize the covariance matrices for each component.
        A default implementation could just be an identity matrix.
        You can use whatever tricks you'd like, we will only test the shape.

        Return:
            sigma: np.ndarray(K,D,D), the Gaussians's covariance matrices
        Hints:
            np.eye() makes an identity matrix :)
            You'll need K of them though.
        """
        sigma = np.tile(np.identity(self.D), (self.K, 1, 1))
        return sigma

    def _init_components(self):
        """
        Calls your create functions that you just completed above.
        """
        if self.seed is not None:
            np.random.seed(self.seed)
        pi = self.create_pi()
        mu = self.create_mu()
        sigma = self.create_sigma()
        return pi, mu, sigma

    def _ll_joint(self, pi, mu, sigma):
        """
        Used to compute the log-likelihood of each datapoint (in self.points) under the model.
        To be precise, this should generate a 2D array, computing the likelihood for each point under each component separately.
        The formula is in the notebook.

        Looping over N may take too much time, while looping over K should be fine.
        We won't check for looping, but there will be a loose time constraint.

        Args:
            pi: np.ndarray(K,), mixing coefficients
            mu: np.ndarray(K,D), means
            sigma: np.ndarray(K,D,D), covariances
        Return:
            ll: np.ndarray(N,K), the log-likelihood ELBO terms
        Note:
            Pay careful attention to the usage of LOG_CONST.
            You should unconditionally include it, unlike SIGMA_CONST.
        """
        log_likelihood = np.empty((self.N, self.K))
        for k in range(self.K):

            likelihood =  pi[k] * self.multinormalPDF(self.points, mu[k], sigma[k]) + LOG_CONST
            log_likelihood[:, k] = np.log(likelihood)
        
        return log_likelihood

    def _E_step(self, pi, mu, sigma):
        """
        In the E-Step, we calculate the expected values of the hidden variables under the current parameters.
        For GMM, this means calculating the responsibilities (tau) of each point to each component.

        Args:
            pi: np.ndarray(K,), mixing coefficients
            mu: np.ndarray(K,D), means
            sigma: np.ndarray(K,D,D), covariances
        Return:
            tau: np.ndarray(N,K), the responsibilities of each point to each component
        Hint:
            Re-read the background material for the formulation of responsibility.
            The E-Step can be done solely using two of the helper functions you've already made.
        """
        # log_likelihood_values = self._ll_joint(pi, mu, sigma) #n*k size
        # for k in range(self.K):
        #     num = np.exp(log_likelihood_values[:, k])
        #     denom = np.sum(log_likelihood_values, axis=1)
        #     tau = num/denom
        return self.softmax(self._ll_joint(pi,mu,sigma))

    def _M_step(self, tau):
        """
        In the M-Step, we calculate the maximum likelihood estimators for the parameters given the E-Step's results.
        For GMM, this means using responsibilities to recompute the means, covariances, and mixing ratios.

        Args:
            tau: np.ndarray(N,K), the responsibilities matrix
        Return:
            pi: np.ndarray(K,), new mixing coefficients
            mu: np.ndarray(K,D), new means
            sigma: np.ndarray(K,D,D), new covariances
        Hint:
            There are formulas in the slides and in the Jupyter Notebook.
        """
        N_k = np.sum(tau, axis= 0)
        pi = (N_k / self.N)

        mu = np.empty((self.K, self.D))
        sigma = np.empty((self.K, self.D, self.D))
        for k in range(self.K):
            mu[k] = np.sum(tau[:, k, np.newaxis] * self.points, axis=0)/N_k[k]
            centered = self.points - mu[k]
            sigma[k] = ((tau[:, k, np.newaxis] * centered).T @ centered) /N_k[k]

        return pi, mu, sigma

    def __call__(self, rel_tol=1e-10, disable_tqdm=False):
        """
        __call__ is a python dunder method that lets you just call an instance! If you have:
        myGMM = GMM(some_data, K=7)
        Then, `myGMM.__call__()` is the same as `myGMM()`

        This function runs the code you wrote to fit the GMM.
        It starts at your _init_components, then alternates _E_step and _M_step until convergence of the likelihood.

        Set disable_tqdm to True to get rid of progress bars.

        Return:
            tau, (pi, mu, sigma)

            tau: np.ndarray(N,K), the responsibilities from the most recent E-step
            pi: np.ndarray(K,), fitted mixing coefficients from the most recent M-step
            mu: np.ndarray(K,D), fitted means from the most recent M-step
            sigma: np.ndarray(K,D,D), fitted covariances from the most recent M-step
        """
        pi, mu, sigma = self._init_components()
        pbar = tqdm(range(self.max_iters), disable=disable_tqdm)
        prev_loss = None
        for it in pbar:
            tau = self._E_step(pi, mu, sigma)
            pi, mu, sigma = self._M_step(tau)
            joint_ll = self._ll_joint(pi, mu, sigma)
            loss = -np.sum(self.logsumexp(joint_ll))
            if prev_loss is not None:
                relative_diff = (prev_loss - loss) / prev_loss
                if relative_diff < rel_tol:
                    break
            prev_loss = loss
            if not disable_tqdm:
                pbar.set_description("iter %d, loss: %.4f" % (it, loss))
            self.num_iters += 1
        return tau, (pi, mu, sigma)


def cluster_pixels_gmm(image, K, max_iters=10):
    """
    Clusters pixels in the input image.

    Flatten the image down to a list of colors, (H*W, 3), then cluster those colors into K components.
    For each pixel in the list, set it to the mean of the component to which it most likely belongs.
    Return the image reconstructed.

    Args:
        image: np.ndarray(H,W,3)[int], an input image
          - this will be ints in the range [0,255]
          - if your work assumes the values are floats, you will need to upcast
        K: int, number of colors to compress to
        max_iters: int, number of iterations at which to stop regardless of convergence
    Return:
        clustered_image: np.ndarray(H,W,3)[int], image after compression and reconstruction
          - this must contain only K unique colors
          - this must be ints in the range [0, 255]
    Hints:
        To force a type conversion on a numpy array, you can use .asType(...).
        You can use numpy types or native types.
        np.clip() can be used to clip values in an array at a minimum and maximum.
        Also, remember you can call your GMM with the disable_tqdm=True kwarg to make it not spawn a tqdm bar if you'd like.
    """
    flattened_image = image.reshape(-1, 3).astype(float)
    gmm = GMM(flattened_image, K, max_iters)
    tau, (pi, mu, sigma) = gmm(disable_tqdm=True)
    labels = np.argmax(tau, axis=1)

    reconstructed_image = np.rint(np.clip(mu[labels], 0, 255)).astype(np.uint8)

    clustered_image = reconstructed_image.reshape(image.shape)
    return clustered_image


def image_frob_dist(image1, image2):
    """
    Computes the distance between two images (using the Frobenius tensor norm).
    To make it scale invariant, we also divide by the number of pixels being summed.
    This is sensitive to whether you represent images as [0,255] or [0.0,1.0], but we only use the former.

    We can use this to determine the distance between our image and our reconstruction.

    Args:
        image1: np.ndarray(...), an input image
        image2: np.ndarray(...), an input image
          - must be broadcastable shapes
    Return:
        distance: float, the distance between the images
    """
    frobenius_dist = float(np.sqrt(np.sum(np.power(image1 - image2, 2))))
    num_pixels = image1.flatten().shape[0]
    distance = frobenius_dist / num_pixels
    return distance


def density(points, pi, mu, sigma):
    """
    Evaluate the probability density at each point under the supplied GMM parameters.

    Args:
        points: np.ndarray(N, D), a set of D-dimensional points to be evaluated
        pi: np.ndarray(K,), mixture coefficients
        mu: np.ndarray(K, D), component means
        sigma: np.ndarray(K, D, D), component covariances
    Return:
        densities: np.ndarray(N,), the pdf value of each supplied point
    Hint:
        You already did the normal distribution for a single Gaussian in the GMM class.
        You can just loop over the components K to compute the mixture.
    """
    K = np.shape(pi)[0]

    densities = np.zeros(np.shape(points)[0])

    for k in range(K):
        densities += pi[k] * GMM.multinormalPDF(points, mu[k], sigma[k])

    return densities



def rejection_sample(xmin, xmax, ymin, ymax, pi, mu, sigma, dmax=1.0):
    """
    Performs rejection sampling.

    1. Pick a candidate c=(x,y) uniformly at random inside a bounded domain, x from [xmin, xmax] and y from [ymin, ymax]
    2. Compute the probability density of that point under the GMM, p(c).
    3. Pick a number uniformly at random from the range [0.0, dmax]
    4. If u <= p(c), return c.
    5. Else, reject c, and try again. (this may happen an arbitrary number of times)

    Args:
        xmin: float, a lower bound on generated x values
        xmax: float, an upper bound on generated x values
        ymin: float, a lower bound on generated y values
        ymax: float, an upper bound on generated y values
        pi: np.ndarray(K,), mixture coefficients (to be passed into density)
        mu: np.ndarray(K, D), component means (to be passed into density)
        sigma: np.ndarray(K, D, D), component covariances (to be passed into density)
        dmax: float, the upper bound on d
    Return:
        point: np.ndarray(1, 2), the x and y coordinates of the sampled datapoint
    """
    while(True):
        c = np.array([[np.random.uniform(xmin, xmax), np.random.uniform(ymin, ymax)]])

        p_c = density(c, pi, mu, sigma)[0]

        u = np.random.uniform(0.0, dmax)

        if u <= p_c:
            return c
    

