"""
some gaussian utilities
"""
import numpy as np
from jax.config import config; config.update("jax_enable_x64", True)
from jax import numpy as jnp

def Normal_logZ(mu, cov): #tested
    """
    compute the log normalizing constant of a normal distribution given a mean vector and a covariance *vector

    arguments
        mu : jnp.array(Dx)
            mean vector
        cov : jnp.array(Dx)
            covariance vector

    returns
        logZ : float
            log normalization constant
    """
    dim = len(mu)
    logZ = 0.5*dim*jnp.log(2.*jnp.pi) + 0.5 * jnp.log(cov).sum()
    return logZ

unnormalized_Gaussian_logp = lambda x, mu, cov: -0.5*((x-mu)/cov).dot(x-mu)
normalized_Gaussian_logp = lambda x, mu, cov: unnormalized_Gaussian(x, mu, cov) - Normal_logZ(mu, cov)
