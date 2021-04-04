"""
isings-type models
"""
from __future__ import division, print_function

from jax import numpy as jnp
import numpy as np
from jax.config import config; config.update("jax_enable_x64", True)
from jax import lax, ops, vmap, jit, grad, random
from melange_lite.melange_lite import SMCSamplerFactory
from melange_lite.magnets.utils import *

class IsingsModellSMCSampler(SMCSamplerFactory):
    """
    inherit the general functionality of an smc sampler

    parameter_dict = {
                        'M_parameters' : Array, # singularly the triple (tuple) of potential parameters, namely (J, h, beta) if equal to the IW energy fn
                     }

    X = {
         x : Array, # positions of shape (N, L, L)
         seed : Array, # jax.random.PRNGKey of shape (N),
        }
    """


    def __init__(self,
                 T,
                 N,
                 IW_energy_fn,
                 IW_parameters,
                 L,
                 full_scan=True,
                 MCMC=True,
                 M_energy_fn = canon_nn_isings_local_potential,
                 **kwargs):
        """
        arguments
            L : Array (float)
                lattice side length (assumed to be a square)
            full_scan : boolean, default True
                whether to deterministically scan through all lattice position
            MCMC : boolean, default True
                whether one is conducting MCMC w.r.t. the IW_parameter-parameterized IW_energy_fn
            M_energy_fn : Callable, default canon_nn_isings_local_potential
                propagation energy fn for kernel MCMC
        """
        self.L = L
        self.full_scan = full_scan
        self.MCMC = MCMC
        self.M_energy_fn = canon_nn_isings_local_potential if MCMC else M_energy_fn
        super().__init__(T, N, IW_energy_fn, IW_parameters, **kwargs)

    def _handle_M0_kernel(self, **kwargs):
        _M0 = default_M0(N = self.N,
                         L = self.L,
                         start_beta = self._beta0)

        #set the attrs
        self.M0 = _M0

    def _handle_logG0(self, **kwargs):

        def logG0(Xs, parameter_dict):
            return jnp.zeros(self.N)

        self.logG0 = logG0

    def _handle_M(self):
        MCMC_prop = get_MCMC_proposal(self.full_scan, self.M_energy_fn)

        def _M(X, parameter_dict, t):
            """
            proposal for a singular lattice x
            """
            # grab the parameters
            x = X['x']
            run_seed, seed = random.split(X['seed'])

            potential_parameters = lax.cond(self.MCMC, lambda x: self.IW_parameters[x+1], lambda x: parameter_dict['kernel_parameters'][x], t)

            # do a move
            out_x, log_ratio = MCMC_prop(x, run_seed, potential_parameters)

            return {'x': out_x, 'seed': seed, 'log_kernel_ratio': log_ratio}

        self._M = _M #set that attr
        self.M = vmap(
                      _M,
                      in_axes=(0, None, None)
                      )
    def works(self):
        """
        modify the work fn in place so that
        """
        super_work_fn = super().works()
        def work_fn(parameter_dict):
            return super_work_fn(parameter_dict)

        return work_fn

class IsingsModellSMCMCMCSampler(IsingsModellSMCSampler):
    """
    inherit the general functionality of an smc sampler
    Important : the initial distribution corresponds to the infinite temperature limit (i.e. beta=0.)
        you will have to modify this functionality to make BAR estimator work the other way around...

    parameter_dict = {
                        'seed' : Array # random.PRNGKey to start.
                     }

    X = {
         x : Array, # positions of shape (N, L, L)
         seed : Array, # jax.random.PRNGKey of shape (N),
        }
    """


    def __init__(self,
                 T,
                 N,
                 direction,
                 L,
                 full_scan=True,
                 MCMC = True,
                 M_energy_fn = canon_nn_isings_local_potential,
                 **kwargs):
        energy_fn = canon_nn_isings_potential
        assert set(direction).issubset(set([0,1]))
        self._direction = jnp.int64(direction)
        self._beta0 = self._direction[0]
        IW_parameters = jnp.hstack([jnp.ones(T)[..., jnp.newaxis], #J
                            jnp.zeros(T)[..., jnp.newaxis], #h
                            jnp.linspace(self._direction[0],self._direction[1],T)[..., jnp.newaxis] #beta
                           ])
        super().__init__(T, N, energy_fn, IW_parameters, L, full_scan, MCMC, M_energy_fn, **kwargs)

    def _handle_logG(self):

        def _logG(Xp, X, parameter_dict, t):
            xp, x = Xp['x'], X['x']

            #compute importance_weight (there is 1 more index in the IW parameters than in everything else...)
            params_t, params_tm1 = self.IW_parameters[t+1], self.IW_parameters[t]

            IWs = -self._IW_energy_fn(xp, params_t) + self._IW_energy_fn(xp, params_tm1)

            #finalize and return
            lws = IWs
            return lws

        self._logG = _logG
        self.logG = vmap(_logG, in_axes=(0, 0, None, None))

class TrainableIsingsModellSMCSampler(IsingsModellSMCSampler):
    """
    specialty class that allows for trainability of the target potential

    NOTE:
    you must still abide by an Importance Potential that starts at beta=0 or 1. if beta=1, this is biased, of course

    a good use case for IW parameters should be something like...

    IW_parameters = jnp.hstack([jnp.ones(T)[..., jnp.newaxis], #J
               jnp.zeros(T)[..., jnp.newaxis], #h
               jnp.linspace(0,1,T)[..., jnp.newaxis] #beta
              ])

    """
    def __init__(self,
                 T,
                 N,
                 IW_parameters,
                 L = 32,
                 full_scan=True,
                 MCMC=False,
                 M_energy_fn = canon_nn_isings_local_potential):
        """
        Expose the IW parameters so that these are togglable for experimentation
        """
        energy_fn = canon_nn_isings_potential #define the canonical isings potential function
        self._start = IW_parameters[0] # define the starting values of the IW parameters
        assert jnp.allclose(self._start[:2], jnp.array([1., 0.])) #check that the J and h values aren't unexpected
        self._beta0 = self._start[2] #define beta 0
        super().__init__(T, N, energy_fn, IW_parameters, L, full_scan, MCMC, M_energy_fn)

    def _handle_logG(self):

        def _logG(Xp, X, parameter_dict, t):
            xp, x = Xp['x'], X['x']
            log_l_by_k = X['log_kernel_ratio']

            #compute importance_weight (there is 1 more index in the IW parameters than in everything else...)
            params_t, params_tm1 = self.IW_parameters[t+1], self.IW_parameters[t]

            IWs = -self._IW_energy_fn(x, params_t) + self._IW_energy_fn(xp, params_tm1)

            #finalize and return
            lws = IWs + log_l_by_k
            return lws

        self._logG = _logG
        self.logG = vmap(_logG, in_axes=(0, 0, None, None))
