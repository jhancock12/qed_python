# == Some notes ==
# Want to reproduce the results from https://arxiv.org/pdf/2411.05628 on using QC for (2+1)-QED

# External modules
from modules import *

# My modules
from classes import *
from circuit_helpers import *
from plot_helpers import *
from QED_hamiltonian import *
from running_helpers import *

DYNAMICAL_LINKS = [((0,0),1), ((0,0),2), ((1,0),1), ((1,0),2), ((0,1),1), ((1,1),1), ((2,0),2)]

parameters = {
    'L_x': 3,
    'L_y': 2,
    'gauge_truncation': 1,
    'n_fermion_layers' : 5,
    'shots': 1e5,
    'dynamical_links': DYNAMICAL_LINKS,
    'm': 10.,
    'g': 1.0,
    'a': 5.0,
    'E_0': 0.0,
    'max_iters': 1000,
}

print("RUNNING")

E_0_values = np.linspace(0.0, 2, 20)
# run_and_print_circuit(parameters, E_0_values)
run_and_print_sparse(parameters, E_0_values)