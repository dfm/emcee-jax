# nopycln: file

__bibtex__ = """
@article{emcee,
   author = {{Foreman-Mackey}, D. and {Hogg}, D.~W. and {Lang}, D. and {Goodman}, J.},
    title = {emcee: The MCMC Hammer},
  journal = {PASP},
     year = 2013,
   volume = 125,
    pages = {306-312},
   eprint = {1202.3665},
      doi = {10.1086/670067}
}
"""
__uri__ = "https://emcee.readthedocs.io"
__author__ = "Daniel Foreman-Mackey"
__email__ = "foreman.mackey@gmail.com"
__license__ = "MIT"
__description__ = "The Python ensemble sampling toolkit for MCMC"

from emcee_jax import host_callback as host_callback, moves as moves
from emcee_jax._src.sampler import sampler as sampler
from emcee_jax.emcee_jax_version import __version__ as __version__
