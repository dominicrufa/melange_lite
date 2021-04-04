"""
some simple analysis functionality
"""
import numpy as np
from jax.config import config; config.update("jax_enable_x64", True)
from jax import numpy as jnp
from jax import grad, random
from jax.scipy.special import logsumexp
from jax import lax

def free_energy(works):
    """
    compute the free energy from a work array
    """
    N = len(works)
    w_min = jnp.min(works)
    return w_min - logsumexp(-works + w_min) + jnp.log(N)

def ESS(works):
    log_weights = -works
    Ws = jnp.exp(log_weights - logsumexp(log_weights))
    ESS = 1. / jnp.sum(Ws**2) / len(works)
    return ESS

def bootstrap(data, seed, function, num_bootstraps, function_kwargs):
    """
    generic bootstrapper
    """
    from functools import partial
    data_collector = []
    random_data = random.choice(seed, data, shape=(len(data), num_bootstraps))
    pfunction = partial(function, **function_kwargs)
    data_collector = lax.map(pfunction, random_data)
    return data_collector

"""
free energy calculator
"""
def symmetric_f(forward_works, backward_works):
    """
    BAR-derived symmetric protocol free energy estimator

    arguments
        forward_works : jnp.array(N)
            reduced works accumulated up to time t' (halfway through the protocol)
        backward_works : jnp.array(N)
            reduced works accumulated from t' to T (second half of protocol)
    """
    N = len(forward_works)
    max_forwards = jnp.max(forward_works)
    offset_forwards = forward_works - max_forwards
    offset_backwards = backward_works + max_forwards

    summant = 1./(jnp.exp(offset_forwards) + jnp.exp(-offset_backwards))
    f = jnp.log(2) + max_forwards - jnp.log(jnp.sum(summant)) + jnp.log(N)

    return f

def symmetric_f_v2(forward_works, backward_works):
    """
    BAR-derived symmetric protocol free energy estimator; version 2 should recover the same free energy as version 1 (above)
    """
    N = len(forward_works)
    max_backwards = jnp.max(backward_works)
    offset_forwards = forward_works + max_backwards
    offset_backwards = backward_works - max_backwards

    summant = 1./(jnp.exp(offset_forwards) + jnp.exp(-offset_backwards))
    f = jnp.log(2) - max_backwards - jnp.log(jnp.sum(summant)) + jnp.log(N)

    return f

def there_and_back_f(forward_works, backward_works):
    """
    f = -ln(2) - ln(sum(e^(-w_f))) + ln(sum(1+e^(-w_{tot}))); where w_{tot} = w_f + w_r
    """
    total_works = forward_works + backward_works
    w_f_min = jnp.min(forward_works)
    f = -jnp.log(2) + w_f_min - logsumexp(-(forward_works - w_f_min)) + jnp.log(jnp.sum(1 + jnp.exp(-total_works)))
    return f
