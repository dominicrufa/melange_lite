"""
some magnetic utilities
"""
from __future__ import division, print_function

from jax import numpy as jnp
import numpy as np
from jax.config import config; config.update("jax_enable_x64", True)
from jax import lax, ops, vmap, jit, grad, random


"""
2D ising model utils
"""

def periodic_shift(L, x, dx):
    """
    do a periodic shift

    arguments
        L : Array
            length of dimension
        x : Array
            positions
        dx : Array
            position increment
    """
    return jnp.mod(x + dx, L)

def pull_spin(x, idx_tuple):
    target_spin = x[idx_tuple]
    L = x.shape[0] #presume it is a periodic hypercube
    Dx = len(x.shape)
    left_spin_index = periodic_shift(L, jnp.array(idx_tuple), jnp.array([-1,0]))
    right_spin_index = periodic_shift(L, jnp.array(idx_tuple), jnp.array([1,0]))

    up_spin_index = periodic_shift(L, jnp.array(idx_tuple), jnp.array([0,1]))
    down_spin_index = periodic_shift(L, jnp.array(idx_tuple), jnp.array([0,-1]))

    return left_spin_index, right_spin_index, up_spin_index, down_spin_index


def nearest_neighbor_energy(index, x, J):
    L = x.shape[0]
    Dx = len(x.shape)
    left_idx, right_idx, up_idx, down_idx = pull_spin(x, index)
    return -J * (x[left_idx[0], left_idx[1]] + x[right_idx[0], right_idx[1]] + x[up_idx[0], up_idx[1]] + x[down_idx[0], down_idx[1]]) * x[index[0], index[1]]


def nearest_neighbor_energy_edge(index, x, J, edge_scale=1.):
    """
    the special case where the left and top edges of the box are spins -1 and the right and bottom are spins 1
    """
    L = x.shape[0]
    Dx = len(x.shape)
    left_idx, right_idx, up_idx, down_idx = pull_spin(x, index)
    spin = x[index[0], index[1]]

    #left
    left_e = lax.cond(index[1] == 0, lambda s: -J*-1*s*edge_scale, lambda s: -J*x[left_idx[0], left_idx[1]]*s, spin)

    #up
    up_e = lax.cond(index[0] == 0, lambda s: -J*-1*s*edge_scale, lambda s: -J*x[up_idx[0], up_idx[1]]*s, spin)

    #right
    right_e = lax.cond(index[1] == L-1, lambda s: -J*s*edge_scale, lambda s: -J*x[right_idx[0], right_idx[1]]*s, spin)

    #down
    down_e = lax.cond(index[0] == L-1, lambda s: -J*s*edge_scale, lambda s: -J*x[down_idx[0], down_idx[1]]*s, spin)

    return left_e + up_e + right_e + down_e

def half_correction_edge(index, x, J):
    """
    add a half kT correction to the edge spins
    """
    pass



full_nearest_neighbor_energy = lambda indices, x, J: 0.5 * vmap(nearest_neighbor_energy, in_axes=(0, None, None))(indices, x, J)

def get_all_indices(L):
    """
    get combinatorial indices of an LxL square (zero_indexed)
    """
    all_indices = jnp.transpose(jnp.array(jnp.meshgrid(jnp.arange(L, dtype=jnp.int32), jnp.arange(L, dtype=jnp.int32)))).reshape(-1, 2)
    return all_indices

def nn_isings_potential(x, J=1., h=0.):
    """
    isings model potential for nearest-neighbor interactions

    arguments
        x : Array
            spins
        J : Array
            spin coupling (in kT)
        h : array
            external field (in kT)
    """
    L = x.shape[0] #presume it is a square

    #magnetization
    external_e = (h*x).sum()

    #nearest neighbors
    all_indices = get_all_indices(L)
    _full_nearest_neighbor_energy = full_nearest_neighbor_energy(all_indices, x, J).sum()

    return external_e + _full_nearest_neighbor_energy

def nn_isings_potential_edge(x, J=1., h=0.):
    """
    copy of the above, except the edge indices are specified with *
    """
    L = x.shape[0] #presume it is a square

    #magnetization
    external_e = (h*x).sum()

    #





def nn_isings_local_potential(x_index_pair, J=1., h=0.):
    """
    given a spin matrix and a local index, compute the energy around the local index

    arguments
        x_index_pair : tuple
            tuple of (spin_state, index)
        J : Array (float)
            coupling term (in kT)
        h : Array (float)
            external magnetic field (in kT)

    """
    x, index = x_index_pair
    local_spin = x[index[0], index[1]]

    #magnetization
    external_e = local_spin*h

    #nearest_neighbors
    nn_energy = nearest_neighbor_energy(index, x, J)

    return external_e + nn_energy

def canon_nn_isings_potential(x, parameters):
    J, h, beta = parameters[:3]
    return beta * nn_isings_potential(x, J, h)

def canon_nn_isings_local_potential(x_index_pair, parameters):
    """
    a canonicalization of the local isings model potential

    arguments
        x_index_pair : tuple
            tuple of (spin_state, index)
        parameters : Array
            J : Array (float)
                coupling term (in kT)
            h : Array (float)
                external magnetic field (in kT)
            beta: Array (float)
                1/kT
    """
    J, h, beta = parameters[:3]
    return beta * nn_isings_local_potential(x_index_pair, J, h)

def do_MCMC_move(index, x, parameters, seed, local_energy_fn):
    """
    do an MCMC move

    returns
        returnable_x : Array
            out_x
        log_l_by_k : Array (float)
            log ratio of backward_to_forward kernel ratio
    """
    #compute starting energy
    start_e = local_energy_fn((x, index), parameters)

    start_spin = x[index[0], index[1]]

    #modify x
    mod_x = ops.index_update(x,
                             ops.index[index[0], index[1]],
                             -x[index[0], index[1]]
                            )

    mod_spin = mod_x[index[0], index[1]]

    #compute ending energy
    end_e = local_energy_fn((mod_x, index), parameters)

    accept = lax.cond(end_e - start_e <= 0,
                      lambda x: True,
                      lambda x: random.choice(seed, jnp.array([True, False]), p=jnp.array([jnp.exp(-x), 1. - jnp.exp(-x)])),
                      end_e - start_e
                     )

    returnable_x = lax.cond(accept, lambda x: x[0], lambda x: x[1], (mod_x, x))
    final_e = local_energy_fn((returnable_x, index), parameters)

    return returnable_x, -(start_e - final_e)

"""
MCMC functionality for MH algorithm
"""

def get_MCMC_proposal(full_scan, local_energy_fn):
    """
    return an MCMC proposal function

    arguments
        full_scan : boolean
            whether to scan deterministically over all spins
        local_energy_fn : Callable[Tuple[Array, Array], Array]
            inputs are a tuple of (lattice, index) and parameters

    returns
        MCMC_propagator_fn : Callable[Array, Array, Array]:->Tuple[Array, Array]
    """

    def MCMC_scan_fn(carry, index):
        x, seed, potential_parameters = carry
        run_seed, seed = random.split(seed)
        out_x, log_ratio = do_MCMC_move(index, x, potential_parameters, run_seed, local_energy_fn)
        return (out_x, seed, potential_parameters), log_ratio

    def MCMC_propagator(x,
                        seed,
                        potential_parameters):
        """
        scan over the lattice sites with L**2 random choices and then do an MCMC move
        """
        L = x.shape[0]
        choice_seed, run_seed = random.split(seed)
        all_indices = get_all_indices(L)
        num_proposals = L**2
        choice_indices = lax.cond(full_scan, lambda x: all_indices, lambda x: all_indices[random.choice(choice_seed, jnp.arange(L**2), shape=(num_proposals,))], 0.)
        (out_x, out_seed, _), log_ratios = lax.scan(MCMC_scan_fn, (x, run_seed, potential_parameters), choice_indices)
        return out_x, log_ratios.sum()

    return MCMC_propagator

"""
rendering functionality
"""

def to_two_color(lattice):
    blue = np.ones(lattice.shape, dtype=np.int)*255
    red = np.zeros(lattice.shape, dtype=np.int)
    red[lattice < 0] = 255
    green = red
    return np.array([red, green, blue])

def to_gif(dataset, filename, fps=8):
    from array2gif import write_gif
    print("Frames: {}".format(len(dataset)))
    colors = []
    write_gif(
        [to_two_color(lattice) for lattice in dataset],
        filename,
        fps=fps
                )

def show_gif(fname):
    import base64
    from IPython import display
    with open(fname, 'rb') as fd:
        b64 = base64.b64encode(fd.read()).decode('ascii')
    return display.HTML(f'<img src="data:image/gif;base64,{b64}" />')

"""
some default functionality for methods
"""
def default_M0(N, L, start_beta):
    """
    function to get a default _M0 proposal

    arguments
        N : Array (int)
            number of particles
        L : Array (int)
            lattice side length
        start_beta : Array (int)
            starting value of beta

    return
        _M0 : Callable[Dict]: -> Dict
            function
    """

    def _M0(parameter_dict):
        """
        we are only allowed to sample from beta=0 (inf temperature)
        """
        seed = parameter_dict['seed']
        generator_seed, seed = random.split(seed)
        xs = lax.cond(start_beta == 0,
                      lambda x : random.choice(generator_seed, jnp.array([-1, 1]), shape=(N, L, L)),
                      lambda x: jnp.ones((N, L, L), dtype=jnp.int64) * random.choice(generator_seed, jnp.array([-1, 1]), shape=(N,))[..., jnp.newaxis, jnp.newaxis],
                      0.)
        return {'x': xs, 'seed': random.split(seed, N), 'log_kernel_ratio': jnp.zeros(N)}

    return _M0
