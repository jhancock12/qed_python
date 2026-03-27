from modules import *
from classes import *
from running_helpers import *
from QED_hamiltonian import *
from my_spsa import *

parameters = {
    'L_x': 3,
    'L_y': 2,
    'gauge_truncation': 1,
    'n_fermion_layers': 2,
    'n_extra_layers': 0,
    'shots': 10000,
    'm': 3.0,
    'g': 1.0,
    'a': 1.0,
    'E_0': [0.0, 0.0],
    'dynamical_links': [((1,0), 2), ((2,0), 2)],
    'gauss_weight': 0.0, # Enforces Gauss' law
    'charge_weight': 100.0,
    'simulator': CPU_NOISELESS_SIMULATOR,
    'noisy' : False
}
    
SPSA_parameters = {
    'max_iters' : 10000,
    'average_length' : 5,
    'grad_tol' : 1e-12,
    'average_tol' : 1e-10,
    'a' : 0.08,
    'c' : 0.03,
    'prints' : False,
    'diagnostics' : False
    }

print("RUNNING")
print("parameters:")
for key in list(parameters):
    print(key,":",parameters[key])
    
for key in list(SPSA_parameters):
    print(key,":",SPSA_parameters[key])

lattice = Lattice(parameters['L_x'], parameters['L_y'], parameters['gauge_truncation'], parameters['dynamical_links'], charge_site = (0,0), anticharge_site = (2,1), E_0 = [0.0, 0.0])

hamiltonian = generate_qed_hamiltonian(parameters, lattice)
hamiltonian.latex_plot()
H = hamiltonian.to_sparse_matrix()
H = H.astype(np.complex128)

evals, evecs = eigsh(H, k=1, which='SA')
    
E0 = evals[0].real
psi_vec = evecs[:, 0]
    
print("Final energy (normal):", E0)
        
results = observe_and_print_reduced_from_statevector(psi_vec, lattice)

