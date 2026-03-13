import numpy as np
import qiskit
import qiskit_aer
import cmath
import time
import copy
import random
import scipy
import matplotlib.pyplot as plt
import scipy.sparse as spar
import argparse
import statsmodels.api as sm
from scipy.sparse.linalg import eigsh
import itertools

# self.n_dynamical_links = self.n_links - (self.n_fermion_qubits - 1)
# 2x2 : any one link - [[((0,0),1)], [((0,0),2)], [((1,0),2)], [((0,1),1)]]
# 3x2 : any two links
# 3x3 : 
# For now, I will not generalize, and will use the ones that they used

CPU_NOISELESS_SIMULATOR = qiskit_aer.AerSimulator()

GPU_NOISELESS_SIMULATOR = qiskit_aer.AerSimulator(
    method='statevector',
    device='GPU',
    precision='single',
    shots=0, 
    max_parallel_threads=1,
    max_parallel_shots=1024,
)

PAULI_PHASES = {
    'I': {
        'I': ('I', 1),
        'X': ('X', 1),
        'Y': ('Y', 1),
        'Z': ('Z', 1),
    },
    'X': {
        'I': ('X', 1),
        'X': ('I', 1),
        'Y': ('Z', 1j),
        'Z': ('Y', -1j),
    },
    'Y': {
        'I': ('Y', 1),
        'X': ('Z', -1j),
        'Y': ('I', 1),
        'Z': ('X', 1j),
    },
    'Z': {
        'I': ('Z', 1),
        'X': ('Y', 1j),
        'Y': ('X', -1j),
        'Z': ('I', 1),
    },
}

def smart_round(number, dec_places):
    if isinstance(number, dict):
        for key in list(number):
            number[key] = smart_round(number[key], dec_places)
        return number
    elif isinstance(number, list) or isinstance(number, np.ndarray):
        for k in range(len(number)):
            number[k] = smart_round(number[k], dec_places)
        return number
    else:
        re = 0.0 if abs(number.real) < 1e-8 else number.real
        im = 0.0 if abs(number.imag) < 1e-8  else number.imag

        re = round(re, dec_places)
        im = round(im, dec_places)

        if im == 0.0:
            return float(re)
        return complex(re, im)