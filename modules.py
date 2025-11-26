import numpy as np
import qiskit
import qiskit_aer
import cmath
import time
import copy
import random
import scipy
import matplotlib.pyplot as plt

# self.n_dynamical_links = self.n_links - (self.n_fermion_qubits - 1)
# 2x2 : any one link - [[((0,0),1)], [((0,0),2)], [((1,0),2)], [((0,1),1)]]
# 3x2 : any two links
# 3x3 : 
# For now, I will not generalize, and will use the ones that they used

DYNAMICAL_LINKS = [((1,0),2), ((2,0),2)]

NOISELESS_SIMULATOR = qiskit_aer.AerSimulator()
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