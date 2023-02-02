import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.optimize import minimize as sp_minimize
from scipy.optimize import Bounds as sp_bounds
from mpi4py import MPI


def cross_entropy(f,p,comm,n_terms=2,maxiter=100,n_samples=100,mu=None,sigma=None): 
    """
    Cross entropy method for fitting a gaussian mixture model for 
    importance sampling.
    
    Suppose we want to compute
        int_a^b f(x)*p(x) dx
    The optimal sampling distribution is g(x) = f(x)*p(x)/l for some normalization 
    constant l. 
    
    We will model p with q. The KL divergence between g and q is 
        KL(g,q) = int glog(g/q) dx = int g*log(g) dx - int g*log(q) dx
    
    To find the optimal q that is closest to g we minimize the KL divergence, which
    is equivalent to maximizing
        Loss(q) = int g*log(q) dx = Expecation[g*log(q)]
    since the first term does not depend on q. 
    
    We do this iterativly by the Cross-Entropy method.
    
    Note that if samples X are drawn from a distribution r(x) then the loss must
    be weighted correctly
        Loss(q) = int g*r*log(q)/r dx = Expecation[g*log(q)/r]
    
    1. initialize the variational parameters u.
    2. while not done
        - generate samples X ~ q(x | u)
        - evaluate f(x)*p(x) and q(X|u) for each sample X
        - update u by solving the empirical problem
              max_v mean[f(X)*p(X)*log(q(X | v))/q(X|U)] 
    
    In the last line we reweight the loss because samples are drawn from q(x|u).

    The empirical problem is solved computationally.

    f: function handle for function f
    p: function handle for pdf p
    n_terms: number of terms in the gaussian symmetric, will double if symmetric
            is True
    maxiter: number of iterations
    n_samples: number of samples per iteration

    return gmix, a gaussian mixture model which is fit with cross entropy
    """
    # TODO: incorporate bounds on the variables for definite-integrals
    # i.e. we should be able to input a lower and upper bound [a,b] on the
    # sampling interval
    # 

    rank = comm.Get_rank()

    # variational parameters
    if mu is None:
        mu = np.zeros(n_terms)
        if rank == 0:
          mu = np.random.randn(n_terms)
        comm.Bcast(mu,root=0)
    if sigma is None:
        sigma = np.ones(n_terms)
    weights = np.ones(n_terms)/n_terms

    for kk in range(maxiter):
        # define a gaussian mixture model
        gmix  = gaussian_mixture(mu,sigma,weights)

        # take some samples
        x_samples = np.zeros(n_samples)
        gmix_val = np.zeros(n_samples)
        if rank == 0:
          # take some samples within the region
          x_samples = gmix._sample(n_samples)
          # evaluate the probability of samples
          gmix_val = gmix._pdf(x_samples)
        comm.Bcast(x_samples,root=0)
        comm.Bcast(gmix_val,root=0)

        fp_val = np.array([f(x)*p(x) for x in x_samples])
        # likelihood ratio
        lr = fp_val/gmix_val
        # set up an optimization to get the new variational parameters
        # TODO: derive analytic gradients
        def obj(v):
            """
            Cross entropy loss for minimization
            """
            # unpack the variational parameters
            mu = np.copy(v[:n_terms])
            sigma = np.copy(v[n_terms:2*n_terms])
            weights = np.copy(v[2*n_terms:])
            weights = np.append(weights,1-np.sum(weights))
            # TODO: dont build a new mixture model for each eval
            # that is way too slow, just hard code the evaluation here.

            # make a new mixture object
            q = gaussian_mixture(mu,sigma,weights)
            # evaluate the loss
            loss = np.mean(lr*np.log(q._pdf(x_samples)))
            return -loss

        # initial variational parameters
        v0 = np.copy(np.hstack((gmix.mu,gmix.sigma,gmix.w[:n_terms-1])))

        # TODO: log-transform out the bound constraints on sigma
        # b/c the constraint is really a true inequality.

        # optimize with scipy
        lb = -np.inf*np.ones(3*n_terms-1)
        lb[n_terms:2*n_terms] = 1e-10 # non-negative std 
        lb[2*n_terms:] = 1e-10 # non-negative weights
        bounds = sp_bounds(lb)
        res = sp_minimize(obj,v0,method='L-BFGS-B',bounds=bounds,options = {'gtol':1e-5})
        v = np.copy(res.x)

        # unpack the new variational parameters
        mu = np.copy(v[:n_terms])
        sigma = np.copy(v[n_terms:2*n_terms])
        weights = np.copy(v[2*n_terms:])
        weights = np.append(weights,1-np.sum(weights))

        # make sure everyone is synced
        comm.Bcast(mu,root=0)
        comm.Bcast(sigma,root=0)
        comm.Bcast(weights,root=0)

        # make sure weights are indeed non-negative
        weights = np.maximum(weights,0.0)
        
    return gmix

class gaussian_mixture:
    """
    Form a mixture of 1D-gaussians,
        pi(x) = sum_{i=1}^n_terms w_i*f(x)
    where w_i are weights and f are gaussian pdfs.
    """

    def __init__(self,mu,sigma,w=None,symmetric=False):
        """
        n_terms: int, number of terms
        mu: float or 1d array of length n_terms, mean of the mixture components.
        sigma: float or 1d array of length n_terms, sigma of the mixture components.
        w: 1d arrayof length n_terms, weights of the mixture, must add to 1.
        symmetric: bool, symmetrizes the model by adding n_terms more components
            where identical std, negative mean, and identical weights.
        """
        self.n_terms = len(mu)
        self.mu = np.array(mu)
        self.sigma = np.array(sigma)
        if w is None:
            self.w = np.ones(self.n_terms)/n_terms
        else:
            assert np.sum(w) == 1, "w must add to 1"
            self.w = w

        # possibly symmetrize
        self.symmetric = symmetric
        if symmetric:
            self.n_terms = 2*self.n_terms
            self.mu = np.hstack((self.mu,-self.mu)) 
            self.sigma = np.hstack((self.sigma,self.sigma)) 
            self.w = np.hstack((self.w,self.w))/2
    
    def _sample(self,n_samples):
        """
        Sample from the gaussian mixture model.

        X can be sampled from a gaussian mixture by first
        sampling a single mixture component Y=i, then sampling
        X from the gaussian Y. 
        We sample the mixture components as a Categorical variable
        with weight w_i.
        This holds by the Law of Total probability,
            p(X) = sum_{i=1}^N p(X|Y=i)p(Y=i)
                 = sum_{i=1}^N p(X|Y=i)w_i
                 = pi(X).
        """
        # define indexes for the mixture components
        c = np.linspace(0,self.n_terms-1,self.n_terms,dtype=int)
        # sample a mixture component according to the weights
        c_idx = np.random.choice(c,size=n_samples,p=self.w) 
        # now sample the gaussians
        X = self.mu[c_idx] + self.sigma[c_idx]*np.random.randn(n_samples)
        return X

    def _pdf(self,X):
        """
        Evaluate the pdf pi(X)

        X: 1d array of points
        """
        ret = np.zeros(len(X))
        for ii,x in enumerate(X):
            # evaluate the Gaussian pdfs
            gaussians = np.exp(-((x-self.mu)**2)/2/self.sigma/self.sigma)/np.sqrt(2*np.pi)/self.sigma
            ret[ii] = np.sum(gaussians*self.w)
        return ret

    def plot(self):
        """
        Plot the pdf
        """
        idx_min = np.argmin(self.mu)
        a = self.mu[idx_min] - 2*self.sigma[idx_min]
        idx_max = np.argmax(self.mu)
        b = self.mu[idx_max] + 2*self.sigma[idx_max]
        pts = np.linspace(a,b,100)  
        plt.plot(pts,self._pdf(pts))
        n_samples = 10000
        plt.hist(self._sample(n_samples),bins=100,density=True)
        plt.show()
        return None


if __name__ == "__main__":
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

     
    # fit to a gaussian
    f = lambda x: 1.0
    mu = 3.0
    sigma = 2.0
    p = lambda x: np.exp(-((x-mu)/sigma)**2 / 2)/np.sqrt(2*np.pi*sigma*sigma)
    gmix = cross_entropy(f,p,comm=comm,n_terms=1,maxiter=20,n_samples=500)
    print(gmix.mu,gmix.sigma,gmix.w)


    # fit to a gaussian mixture with two components
    f = lambda x: 1.0
    mu = 3.0
    sigma = 1.0
    p = lambda x: np.exp(-((x-mu)/sigma)**2 / 2)/np.sqrt(2*np.pi*sigma*sigma) + np.exp(-(2*(x+mu)/sigma)**2 / 2)/np.sqrt(2*np.pi*sigma*sigma/4)
    gmix = cross_entropy(f,p,comm=comm,n_terms=2,maxiter=20,n_samples=500)
    print(gmix.mu,gmix.sigma,gmix.w)


    # do importance sampling on int_a^b f(x)*p(x)dx
    f = lambda x: x**3
    lb_x = 0
    ub_x = 5
    mu = 3.0
    sigma = 0.3
    p = lambda x: np.exp(-((x-mu)/sigma)**2 / 2)/np.sqrt(2*np.pi*sigma*sigma) 
    # fit g with cross entropy 
    gmix = cross_entropy(f,p,comm=comm,n_terms=1,maxiter=30,n_samples=500)
    # plot the curves
    xs = np.linspace(lb_x,ub_x,500)
    fps = f(xs)*p(xs)
    from scipy.integrate import simpson
    tot = simpson(fps,xs)
    # sample from p
    if rank == 0:
      samples = mu + sigma*np.random.randn(1000)
      fs = f(samples)
      print('mean under generic MC',np.mean(fs))
      print('std under generic MC',np.std(fs))
      samples = gmix._sample(1000)
      fs_is = f(samples)*p(samples)/gmix._pdf(samples)
      print('mean under generic MC',np.mean(fs_is))
      print('std under importance',np.std(fs_is))
      print('sample improvement',(np.std(fs)/np.std(fs_is))**2)

