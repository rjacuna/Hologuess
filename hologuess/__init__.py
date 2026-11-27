"""Modular reconstruction for sequences, Taylor series and observable Ore modules.

Importing Hologuess starts no jobs. SageMath is required for generic exact
coefficient domains; ore_algebra is required only by the Ore adapter/output.
"""
import os as _os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
              'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    _os.environ.setdefault(_name, '1')

from .adapters import guess, prepare, OreGuessResult
from .module import ModuleProblem, ModuleResult, solve_module
from .qsequence import QSequence, QGuess, guess_qsequence
from .interpolation import ActionOracle, InterpolatedRelation, interpolate_relation
from .actions import OreAction, act, module_problem
from .parameter import interpolate_problem
from ._qhsolve import MemoryKernelStore, SQLiteKernelStore
from ._display import operator_latex, set_latex_symbols

__version__ = '0.3.0'
__all__ = ['guess', 'prepare', 'OreGuessResult', 'ModuleProblem', 'ModuleResult',
           'solve_module', 'QSequence', 'QGuess', 'guess_qsequence', 'ActionOracle',
           'InterpolatedRelation', 'interpolate_relation', 'OreAction', 'act',
           'module_problem', 'interpolate_problem', 'MemoryKernelStore', 'SQLiteKernelStore',
           'operator_latex', 'set_latex_symbols']
