"""
melange_lite.py
A short description of the project.

Handles the primary functions
"""
from __future__ import division, print_function

from jax import numpy as jnp
import numpy as np
from jax.config import config; config.update("jax_enable_x64", True)
from jax import lax, ops
from jax_md import quantity


class SMCSamplerFactory(object):
    """
    def run an SMC Sampler with a FeynmanKac model

    Particles
    =========
    `Particles` are defined as batches of _individual particle quantities (like positions, latent vars, velocities, etc);
    the forward propagation kernels `M0` and `M` (for arbitrary time t) return `Particles` as X, a dictionary s.t.
    X = {'x' : Array, # of shape (self.N, Dx) where Dx is the dimension of each particle's quantity 'x'
         'v' : Array, # of shape (self.N, Dv) where Dv is the dimension of each particle's quantity 'v'
         ...
         'seed' : Array, # if dynamics are stochastic, we need a different jax.random.PRNGKey for each particle (and are considered 'per-particle' quantities)
         ...
    }

    Methods
    =========
    the four 'important' methods that we call are M0, M, logG0, logG.
    As canon, each method looks like the following
        * M0(parameter_dict: Dict[str, Array]): -> Dict[str, Array] # returns X (see above)

        * logG0(parameter_dict: Dict[str, Array]): -> Array # returns lws (log weights of the particles)

        * M(X: Dict[str, Array], parameter_dict: Dict[str, Array], t: Array): -> Dict[str, Array] # returns new X particles container

        *logG(Xp: Dict[str, Array], X: Dict[str, Array], parameter_dict: Dict[str, Array], t: Array): -> Array # returns new X particles container

    See the 'dummy' methods for further documentation

    """
    def __init__(self,
                 T, #number of steps in the SMC propagator
                 N, #number of particles
                 IW_energy_fn, #uncanonicalized importance weight energy fn
                 IW_parameters, #importance weight parameters (immutable)
                 **kwargs #kwargs omitted in base FKSMCSampler
                 ):
        """generic init class

        arguments
            T : int
                number of steps in the sampler
            N : int
                number of particles
            IW_energy_fn : fn
                importance weight energy function
                def IW_energy_fn(X: Array, parameters: Array) -> Array # float output (as energy)
            IW_parameters : Array
                parameters that are passed to IW_energy_fn

        parameters
            T : T
            N : N
            IW_parameters : IW_parameters
            _IW_energy_fn : Callable[Dict[str, Array], Dict[str, Array]]
                canonicalized importance weight energy function
        """

        #default arguments
        self.T = T
        self.N = N

        self._check_IW_parameters(IW_parameters)
        self.IW_parameters = IW_parameters

        #importance weight energy function
        from melange.md.utils import canonicalize_fn
        self._IW_energy_fn = jit(vmap(IW_energy_fn, in_axes = (0,None)))

        #handle all of the methods...
        self._handle_methods(**kwargs) #pass all kwargs in hopes the `_handle` function will pick these up

    def _handle_methods(self, **kwargs):
        """
        call:
            `_handle_M0_kernel`,
            `_handle_logG0`,
            `_handle_M`,
            `_handle_logG`

        to set up necessary callable methods.
        """
        self._handle_M0_kernel(**kwargs)
        self._handle_logG0(**kwargs)

        self._handle_M(**kwargs)
        self._handle_logG(**kwargs)

    def _check_IW_parameters(self, IW_parameters):
        """helper function...
        check the importance weight parameters to assert that it is an Array object
        """
        from melange.md.utils import Array
        assert type(IW_parameters) == Array

    def _handle_M0_kernel(self, **kwargs):
        """
        M0 kernel is _usually_ a special case. this handles M0 separately.
        creates an instance method called 'M0'


        def M0_kernel(parameter_dict: Dict[str, Array]) -> Dict[str, Array]:
            arguments
                parameter_dict : Dict

            returns
                X : Dict
                    particles definitions (see class documentation)


        self.M0 = M0_kernel # set the method
        """
        def M0_kernel(parameter_dict):
            raise NotImplementedError(f"you have to define the M0 kernel yourself, dumbass.")

        self.M0 = M0_kernel

    def _handle_logG0(self, **kwargs):
        """
        logG0 calculator is _usually_ a special case (like M0), so it is also defined separately.
        creates an instance method called `logG0`

        def logG0(X : Dict[str, Array], parameter_dict: Dict[str, Array]) -> Dict[str, Array]:
            arguments
                X : Dict
                    particles container from self.M0

                parameter_dict : Dict

            returns
                lws : Array
                    log weights of shape (self.N)

        self.logG0 = logG0
        """
        def logG0(X, parameter_dict):
            raise NotImplementedError(f"you have to define the logG0 kernel yourself, dumbass.")
        #set the attrs
        self.logG0 = jit(logG0)

    def _handle_M(self, M_kernel_fn, M_kernel_energy_fn, M_shift_fn, **kwargs):
        """
        creates an instance method called 'M'

        def M(X : Dict[str, Array], parameter_dict: Dict[str, Array]) -> Dict[str, Array]:
            arguments
                X : Dict
                parameter_dict : Dict
                t : Array

            returns
                X : Dict


        self.M = M # set the method
        """
        def M(X, parameter_dict, t):
            raise NotImplementedError(f"you have to define the M kernel yourself, dumbass.")

        self.M = M

    def _handle_logG(self):
        """
        creates an instance method called `logG`

        def logG(Xp : Dict[str, Array], X : Dict[str, Array], parameter_dict: Dict[str, Array], t : Array) -> Dict[str, Array]:
            arguments
                Xp : Dict
                    previous particles container
                X : Dict
                    current particles container
                parameter_dict : Dict
                t : Array
                    time increment

            returns
                lws : Array
                    log weights of shape (self.N)

        self.logG = logG
        """
        def logG(Xp, X, parameter_dict, t):
            raise NotImplementedError(f"you have to define the M kernel yourself, dumbass.")

        self.logG = logG


    def build_force_fn(self, energy_fn):
        """make a force_fn
        """
        force_fn = -1. * grad(energy_fn) #default argnums=0
        return force_fn


    def works(self):
        """get the run function
        """
        from jax.scipy.special import logsumexp

        #get a scan fn
        def run_scan_fn(carrier, t):
            Xp, params_dict = carrier
            X = self.M(Xp, params_dict, t)
            lws = self.logG(Xp, X, params_dict, t)
            return (X, params_dict), lws

        def works_fn(parameter_dict):
            init_Xs = self.M0(parameter_dict)
            init_lws = self.logG0(init_Xs, parameter_dict)

            carrier = (init_Xs, parameter_dict)
            (out_Xs, _), stacked_lws = lax.scan(run_scan_fn, carrier, jnp.arange(1,self.T))
            all_lws = jnp.vstack((init_lws[jnp.newaxis, ...], stacked_lws))
            last_lws = jnp.cumsum(all_lws, axis=0)[-1]
            return -last_lws
        return works_fn

class GaussianSMCSamplerFactory(SMCSamplerFactory):
    """
    a Gaussian SMC Sampler factory with Gaussian proposals

    parameter_dict = {
                      # M0 parameters
                      mu : Array, # means of the prior distribution with shape (Dx)
                      lcov : Array, # log covariance array of the prior distribution with shape (Dx)
                      seed : Array, # jax.random.PRNGKey
                      M_parameters : Dict,
                            {
                            'mu': Array, # mu array of shape (T, Dx)
                            'lcov': Array, # log covariance of shape (T, Dx)
                            }
                      L_parameters : Dict,
                            same as M_parameters
                     }

    X = {
         x : Array, # positions of shape (N,Dx)
         seed : Array, # jax.random.PRNGKey of shape (N)
        }

    """
    def __init__(self, T, N, IW_energy_fn, IW_parameters, **kwargs):
        super().__init__(self, *args, **kwargs)

        #define a vmapped unnormalized logp function
        from melange_lite.utils.gaussians import normalized_Gaussian_logp
        self._vNormal_logp = vmap(normalized_Gaussian_logp, in_axes = (0, None, None))


    def _handle_M0_kernel(self, **kwargs):

        def gaussian_kernel(parameter_dict):
            mu = parameter_dict['mu']
            Dx = len(mu)
            cov_vector = jnp.transpose(jnp.exp(parameter_dict['lcov']))
            seed = parameter_dict['seed']
            v_seed, x_seed = random.split(seed)

            #make xs
            xs = random.normal(x_seed, (self.N, Dx))*jnp.sqrt(cov_vector) + mu

            return {'x': xs, 'seed': x_seed}


        #set the attrs
        self.M0 = jit(gaussian_kernel)

    def _handle_logG0(self, **kwargs):

        def logG0(Xs, parameter_dict):
            return jnp.zeros(self.N)

        self.logG0 = jit(logG0)

    def _handle_M(self):
        def M(X, parameter_dict, t):
            xs = X['x']
            params = parameter_dict['M_parameters']
            mu = params['mu'][t]
            Dx = len(mu)
            cov = jnp.exp(params['lcov'][t])
            forces = -(xs-mu)
            run_seed, x_seed = random.split(x_seed)
            new_xs = random.normal(run_seed, (self.N, Dx))*jnp.sqrt(cov) + forces
            return {'x': new_xs, 'seed': x_seed}

        self.M = M

    def _handle_logG(self):
        def logG(Xp, X, parameter_dict, t):
            xp, x = Xp['x'], X['x']

            #compute importance_weight
            mu_t, mu_tm1 = self.IW_parameters['mu'][t], self.IW_parameters['mu'][t-1]
            cov_t, cov_tm1 =  jnp.exp(self.IW_parameters['lcov'][t]), jnp.exp(self.IW_parameters['lcov'][t])

            IWs = -self._IW_energy_fn(x, jnp.vstack((mu_t, cov_t))) + self._IW_energy_fn(xp, jnp.vstack((mu_tm1, cov_tm1)))

            #compute M_kernel_logp
            Mparams = parameter_dict['M_parameters']
            k_t_logps = self._vNormal_logp(x, -(xp - Mparams['mu'][t]), jnp.exp(Mparams['lcov'][t]))

            #compute l_kernel_logp
            Lparams = parameter_dict['L_parameters']
            l_tm1_logps = self._vNormal_logp(xp, -(x - Lparams['mu'][t]), jnp.exp(Lparams['lcov'][t]))


            #finalize and return
            lws = IWs + l_tm1_logps - k_t_logps
            return lws

        self.logG = logG
