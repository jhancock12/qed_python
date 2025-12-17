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

def parse_arguments():
    parser = argparse.ArgumentParser()

    # Main parameters
    parser.add_argument('--L_x', type=int, default=2)
    parser.add_argument('--L_y', type=int, default=2)
    parser.add_argument('--gauge_truncation', type=int, default=1)
    parser.add_argument('--n_fermion_layers', type=int, default=1)
    parser.add_argument('--shots', type=int, default=int(1e5))
    parser.add_argument('--m', type=float, default=10.0)
    parser.add_argument('--g', type=float, default=1.0)
    parser.add_argument('--a', type=float, default=5.0)
    parser.add_argument('--E_0', type=float, default=0.0)
    parser.add_argument('--max_iters', type=int, default=1000)
    parser.add_argument('--charge_weight', type=float, default=0.0)
    return parser.parse_args()

args = parse_arguments()

parameters = {
    'L_x': args.L_x,
    'L_y': args.L_y,
    'gauge_truncation': args.gauge_truncation,
    'n_fermion_layers': args.n_fermion_layers,
    'shots': args.shots,
    'm': args.m,
    'g': args.g,
    'a': args.a,
    'E_0': args.E_0,
    'max_iters': args.max_iters,
    'dynamical_links': [],
    'charge_weight': args.charge_weight
}

parameters['simulator'] = CPU_NOISELESS_SIMULATOR
print("RUNNING")

E_0_values = np.linspace(0.0, 3, 5)
E_0_values = [0.0]
tok = time.perf_counter()
run_circuit(parameters, E_0_values)
tik = time.perf_counter()
print("="*10)
print("Total time taken: ",round(tik - tok, 5))
#run_and_print_sparse(parameters, E_0_values)