# == Some notes ==
# Want to reproduce the results from https://arxiv.org/pdf/2411.05628 on using QC for (2+1)-QED

# External modules
import QED_hamiltonian
from modules import *

# My modules
from classes import *
from circuit_helpers import *
from plot_helpers import *
from QED_hamiltonian import *
from running_helpers import *

parameters = {
    'L_x': 2,
    'L_y': 2,
    'gauge_truncation': 1,
    'n_fermion_layers': 10,
    'shots': 5000,
    'm': 0.5,
    'g': 1.0,
    'a': 1.0,
    'E_0': [0.0, 0.0],
    'dynamical_links': [],
    'gauss_weight': 1.0, # Enforces Gauss' law
    'charge_weight': 0.0,
    'simulator': CPU_NOISELESS_SIMULATOR 
}

print("RUNNING TEST")
print("parameters:")
for key in list(parameters):
    print(key,":",parameters[key])
# General parameters

mass_multi = 1
electric_multi = 1
magnetic_multi = 1
kinetic_multi = 1

lattice = Lattice(parameters['L_x'], parameters['L_y'], parameters['gauge_truncation'], parameters['dynamical_links'], charge_site = (), anticharge_site = (), E_0 = [0.0, 0.0])

fuller_runner(parameters, lattice, sparse_test = True, noisy = False, diagnostics = True)