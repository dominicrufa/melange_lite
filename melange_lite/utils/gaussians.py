"""
some gaussian utilities and propagators
"""
import numpy as np
from jax.config import config; config.update("jax_enable_x64", True)
from jax import numpy as jnp
from jax import grad, random

unnormalized_Gaussian_logp = lambda x, mu, cov: -0.5*((x-mu)/cov).dot(x-mu)

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

normalized_Gaussian_logp = lambda x, mu, cov: unnormalized_Gaussian_logp(x, mu, cov) - Normal_logZ(mu, cov)


def EL_mu_sigma(x, potential, ldt, parameters):
    """
    given a potential and its associated parameters, compute an unadjusted langevin move with a log time increment and a starting position
    """
    tau = jnp.exp(ldt)/2.
    force = -grad(potential)(x, parameters)
    mu = x + tau * force
    cov = jnp.array([2*tau]*len(x))
    return mu, cov

def ULA_move(x, potential, ldt, seed, parameters):
    mu, _cov = EL_mu_sigma(x, potential, ldt, parameters)
    return random.normal(seed, x.shape) * jnp.sqrt(_cov) + mu
