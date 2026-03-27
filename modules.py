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
import sympy as sp

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

def gauss_solver(lattice):     
    link_variables = []
    charge_variables = []
    link_variable_dict = {}
    reverse_link_variable_dict = {}
    for k in range(len(lattice.links_list)):
        n = lattice.reverse_labels[lattice.links_list[k][0]]
        direction = lattice.links_list[k][1]
        
        link_variables.append(sp.symbols(f"E{n}{direction}"))
        
        link_variable_dict[lattice.links_list[k]] = link_variables[-1]
        reverse_link_variable_dict[link_variables[-1]] = lattice.links_list[k]
        
    eqs = []
    dependant_variables = []
    independant_variables = []
    for n in range(lattice.n_fermion_qubits):
        
        q = sp.symbols(f"q{n}")
        # dependant_variables.append(q)
        charge_variables.append(q)
        G_n = 0
        site = lattice.labels[n]
        if site == lattice.charge_site:
            G_n += 1
        elif site == lattice.anticharge_site:
            G_n -= 1
        for direction in [1,2]:
            prev_site = list(copy.copy(site))
            prev_site[direction - 1] -= 1
            prev_site = tuple(prev_site)
            if (prev_site, direction) in lattice.links_list:
                G_n += link_variable_dict[(prev_site, direction)]# + lattice.E_0[direction - 1]
                
            if (site, direction) in lattice.links_list:
                G_n -= link_variable_dict[(site, direction)]# + lattice.E_0[direction - 1]
            
                if (site, direction) not in lattice.dynamical_links_list:
                    dependant_variables.append(link_variable_dict[(site, direction)]) 
                else:
                    independant_variables.append(link_variable_dict[(site, direction)]) 
        G_n -= q
        eqs.append(G_n)
        
    independant_variables += charge_variables
     
    sol = sp.solve(eqs[:-1], dependant_variables, dict=True)

    results = {'solution': sol[0], 
               'dependant_variables': dependant_variables,
               'independant_variables': independant_variables,
               'link_variable_dict': link_variable_dict,
               'reverse_link_variable_dict': reverse_link_variable_dict,
               'equations': eqs,
               'link_variables' : link_variables,
               'charge_variables': charge_variables}
    
    return results