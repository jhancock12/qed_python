from modules import *

def multiply_pauli_strings(p1, p2):
    result = []
    phase = 1+0j
    for a, b in zip(p1, p2):
        res, ph = PAULI_PHASES[a][b]
        result.append(res)
        phase *= ph
    return "".join(result), phase   

def multiply_hamiltonian_by_constant(hamiltonian, c):
    new_hamil = {}
    for term, coeff in hamiltonian.items():
        new_hamil[term] = c * coeff
    return new_hamil

def possible_directions(lattice):
    labels = lattice.labels
    possible_labels = list(lattice.reverse_labels)
    directions = {}
    link_indexing = {}
    dynamical_link_indexing = {}
    counter = 0
    dynamical_counter = 0
    for j in range(len(labels)):
        directions[j] = []
        if (labels[j][0] + 1, labels[j][1]) in possible_labels: # x_link
            directions[j].append(1)
            link_indexing[((labels[j][0], labels[j][1]), 1)] = counter
            counter += 1
        if (labels[j][0], labels[j][1]+1) in possible_labels: # y_link
            directions[j].append(2)
            link_indexing[((labels[j][0], labels[j][1]), 2)] = counter
            counter += 1
        
        if ((labels[j][0], labels[j][1]), 1) in lattice.dynamical_links_list:
            dynamical_link_indexing[((labels[j][0], labels[j][1]), 1)] = dynamical_counter
            dynamical_counter += 1
        if ((labels[j][0], labels[j][1]), 2) in lattice.dynamical_links_list:
            dynamical_link_indexing[((labels[j][0], labels[j][1]), 2)] = dynamical_counter
            dynamical_counter += 1

    return directions, link_indexing, dynamical_link_indexing

def possible_links(lattice):
    labels = lattice.labels
    possible_labels = list(lattice.reverse_labels)
    links = []
    for j in range(len(labels)):
        if (labels[j][0] + 1, labels[j][1]) in possible_labels: # x_link
            links.append(((labels[j][0], labels[j][1]), 1))
        if (labels[j][0], labels[j][1]+1) in possible_labels: # y_link
            links.append(((labels[j][0], labels[j][1]), 2))

    return links

def possible_plaquettes(lattice):
    plaquette_ns = []
    indices_list = list(lattice.reverse_labels)
    for key in lattice.labels:
        indices = lattice.labels[key]
        if (indices[0], indices[1] + 1) in indices_list:
            if (indices[0] + 1, indices[1]) in indices_list:
                if (indices[0] + 1, indices[1] + 1) in indices_list:
                    plaquette_ns.append(key)
    return plaquette_ns

class Lattice:
    def __init__(self, L_x, L_y, gauge_truncation, dynamical_links_list, charge_site = (), anticharge_site = (), E_0 = [0.0, 0.0]):
        if L_x*L_y % 2 == 1:
            raise ValueError(f"L_x * L_y must be even")    
        self.L_x = L_x
        self.L_y = L_y
        self.labels = {}
        self.reverse_labels = {}
        self.gauge_truncation = gauge_truncation
        self.charge_site = charge_site
        self.anticharge_site = anticharge_site
        self.E_0 = E_0
        self.E_0_directions = []
        if abs(E_0[0]) > 0.00001:
            self.E_0_directions.append(1)
        if abs(E_0[1]) > 0.00001:
            self.E_0_directions.append(2)
        counter = 0

        for y in range(L_y):
            for x in range(L_x):
                self.labels[counter] = (x, y)
                self.reverse_labels[(x, y)] = counter
                counter += 1
        self.n_fermion_qubits = counter
        
        if dynamical_links_list == []:
            self.dynamical_links_list = possible_links(self)
        elif dynamical_links_list == ['none']:
            self.dynamical_links_list = []
        else:
            self.dynamical_links_list = dynamical_links_list
        
        self.n_links = (self.L_x - 1) * self.L_y + self.L_x * (self.L_y - 1) # OBC 
        self.qubits_per_gauge = 0 if gauge_truncation == 0 else int(np.ceil(np.log2(2*gauge_truncation+1)))
        self.n_gauge_qubits = self.n_links*self.qubits_per_gauge
    
        self.optimal_n_dynamical_links = self.n_links - (self.n_fermion_qubits - 1)
        self.n_dynamical_links = len(self.dynamical_links_list)
        
        if self.n_dynamical_links < self.optimal_n_dynamical_links:
            print(f"This lattice requires at least {self.optimal_n_dynamical_links} dynamical links, you gave: {self.n_dynamical_links}")

        self.n_dynamical_gauge_qubits = self.n_dynamical_links*self.qubits_per_gauge
        self.n_qubits = self.n_fermion_qubits + self.n_dynamical_gauge_qubits
        
        self.directions, self.link_indexing, self.dynamical_link_indexing = possible_directions(self)
        self.plaquettes = possible_plaquettes(self)

    def get_index(self, x, y):
        if (x, y) in self.reverse_labels:
            return self.reverse_labels[(x, y)]
        raise ValueError(f"Site ({x}, {y}) is out of bounds.")

    def get_indices(self, index):
        if index in self.labels:
            return self.labels[index]
        raise ValueError(f"Index {index} is out of bounds.")

    def get_neighbors(self, x, y):
        neighbors = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.L_x and 0 <= ny < self.L_y:
                neighbors.append((nx, ny))
        return neighbors
    
class Hamiltonian:
    def __init__(self, n_qubits = 1):
        self.hamiltonian = {'I'*n_qubits : 0}
        self.n_qubits = n_qubits

    def latex_print(self):
        keys = list(self.hamiltonian)
        string_to_print = ""
        for key in keys:
            matrix_list = list(key)
            if matrix_list == ['I']*len(matrix_list):
                string_to_print += str(self.hamiltonian[key])
            else:
                temp_string = ""
                for i in range(len(matrix_list)):
                    if matrix_list[i] != 'I':
                        temp_string += matrix_list[i] + r"_{" + str(i) + r"} "    
                              
                if self.hamiltonian[key] < 0:
                    string_to_print += r" - " + str(abs(self.hamiltonian[key])) + " " + temp_string
                else:
                    string_to_print += r" + " + str(abs(self.hamiltonian[key])) + " " + temp_string
        print(string_to_print)

    def latex_plot(self, save=False):
        keys = list(self.hamiltonian)
        string_to_print = "H = &"
        counter = 0
        for key in keys:
            matrix_list = list(key)
            if matrix_list == ['I']*len(matrix_list):
                term = str(self.hamiltonian[key])
                if self.hamiltonian[key] < 0:
                    string_to_print += r" - " + term
                else:
                    string_to_print += r" + " + term
            else:
                temp_string = ""
                for i in range(len(matrix_list)):
                    if matrix_list[i] != 'I':
                        temp_string += matrix_list[i] + r"_{" + str(i) + r"} "    
                            
                if self.hamiltonian[key] < 0:
                    string_to_print += r" - " + str(abs(self.hamiltonian[key])) + " " + temp_string
                else:
                    string_to_print += r" + " + str(abs(self.hamiltonian[key])) + " " + temp_string
                counter += 1
                if (counter % 4) == 0:
                    string_to_print += r"\\ &" 
        
        # Remove leading " + " if present
        if string_to_print.startswith(" + "):
            string_to_print = string_to_print[3:]
        elif string_to_print.startswith(" - "):
            string_to_print = string_to_print[1:]  # Keep the minus but remove space
        
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "text.latex.preamble": r"\usepackage{amsmath}"  # ADD THIS LINE
        })

        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
        
        # Use text with wrapping
        ax.text(0.5, 0.5, r"$\begin{aligned}" + string_to_print + r"\end{aligned}$", 
                fontsize=14, ha='center', va='center')
                 
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        if save:
            plt.savefig(f"equation_temp.pdf", bbox_inches='tight')
        plt.show()
             
    def cleanup(self):
        new_hamil = {}
        for key in self.hamiltonian:
            if np.abs(self.hamiltonian[key]) > 0.0001:
                new_hamil[key] = self.hamiltonian[key]
                if self.hamiltonian[key].imag > 0.0001:
                    print("This Hamiltonian will be non-Hermitian, due to the term:", key, ":", self.hamiltonian[key])
                else:
                    new_hamil[key] = new_hamil[key].real
        self.hamiltonian = new_hamil
    
    def multiply_hamiltonians(self, other):
        hamil_1 = self.hamiltonian
        if isinstance(other, Hamiltonian): hamil_2 = other.hamiltonian
        else: hamil_2 = other
        new_hamil = {}
        for term1, coeff1 in hamil_1.items():
            for term2, coeff2 in hamil_2.items():
                new_term, phase = multiply_pauli_strings(term1, term2)
                new_coeff = coeff1 * coeff2 * phase
                if abs(new_coeff.imag) < 1e-12:
                    new_coeff = new_coeff.real
                if new_term in new_hamil:
                    new_hamil[new_term] += new_coeff
                else:
                    new_hamil[new_term] = new_coeff
        self.hamiltonian = new_hamil

    def add_term(self, term, coeff):
        if len(term) != self.n_qubits:
            raise ValueError(f"Terms must be of length {self.n_qubits} for this Hamiltonian, you have entered a term of length {len(term)}")
        else:
            if term in list(self.hamiltonian):
                self.hamiltonian[term] += coeff
            else:
                self.hamiltonian[term] = coeff
    
    def add_hamiltonians(self, other):
        if isinstance(other, Hamiltonian): hamil_other = other.hamiltonian
        else : hamil_other = other
        for key in hamil_other:
            self.add_term(key, hamil_other[key])
    
    def to_matrix(self):
        matrix_dict = {
            'I': np.array([[1,0],[0,1]]),
            'X': np.array([[0,1],[1,0]]),
            'Y': np.array([[0,-1j],[1j,0]]),
            'Z': np.array([[1,0],[0,-1]])
            }
        matrix = np.zeros((2**self.n_qubits,2**self.n_qubits), dtype=complex)
        terms = list(self.hamiltonian)
        for term in terms:
            temp_matrix = np.array([1])
            for t in list(term):
                temp_matrix = np.kron(matrix_dict[t], temp_matrix)
            matrix += self.hamiltonian[term]*temp_matrix
        return matrix
    
    def to_sparse_matrix(self):
        
        I = spar.csc_matrix([[1, 0], [0, 1]], dtype=complex)
        X = spar.csc_matrix([[0, 1], [1, 0]], dtype=complex)
        Y = spar.csc_matrix([[0, -1j], [1j, 0]], dtype=complex)
        Z = spar.csc_matrix([[1, 0], [0, -1]], dtype=complex)

        matrix_dict = {'I': I, 'X': X, 'Y': Y, 'Z': Z}

        # start with a zero sparse matrix
        dim = 2 ** self.n_qubits
        H = spar.csc_matrix((dim, dim), dtype=complex)

        for term, coeff in self.hamiltonian.items():
            # Build kron product for this term
            temp = spar.csc_matrix([1], dtype=complex)

            for t in term:       # e.g. "XIZY"
                temp = spar.kron(matrix_dict[t], temp, format='csc')

            H += coeff * temp

        return H
    
    def to_conjugate(self):
        # Of note here, Pauli strings are self-conjugate
        hamil = self.hamiltonian
        keys = list(hamil)
        new_hamil = {}
        for key in keys:
            new_hamil[key] = hamil[key].real - 1j*hamil[key].imag
        self.conjugate = new_hamil
        return new_hamil

class CircuitBuilder:
    def __init__(self, n_fermion_qubits, n_gauge_qubits):
        self.n_fermion_qubits = n_fermion_qubits
        self.n_gauge_qubits = n_gauge_qubits
        self.n_qubits = n_fermion_qubits + n_gauge_qubits
        self.circuit = qiskit.QuantumCircuit(self.n_qubits,self.n_qubits)
    
    def iSwap(self, theta, j, k):
        self.circuit.ryy(theta/2, j, k)
        self.circuit.rxx(theta/2, j, k)
        return self
    
    def iSwap_block_calculate(self):
        total_layers = int(self.n_fermion_qubits / 2)
        counter = 0
        for layer in range(total_layers):
            qubits_to_use = list(range(layer,self.n_fermion_qubits-layer))
            for _ in range(int(len(qubits_to_use)/2)):
                counter += 1
        return counter

    def iSwap_block(self,thetas_slice):
        total_layers = int(self.n_fermion_qubits / 2)
        counter = 0
        for layer in range(total_layers):
            qubits_to_use = list(range(layer,self.n_fermion_qubits-layer))
            for j in range(int(len(qubits_to_use)/2)):
                self.iSwap(thetas_slice[counter], self.n_gauge_qubits+qubits_to_use[0]+2*j, self.n_gauge_qubits+qubits_to_use[0]+2*j+1)
                counter += 1
        self.circuit.barrier()
        return self

    def R_Y_layer(self,thetas_slice):
        start_qubit = self.n_gauge_qubits
        n_fermion_qubits = len(thetas_slice)
        for qubit in range(n_fermion_qubits):
            self.circuit.ry(thetas_slice[qubit], start_qubit + qubit)
        for qubit in range(n_fermion_qubits - 1):
            self.circuit.cx(start_qubit + qubit, start_qubit + qubit + 1)
        self.circuit.barrier()
        return self

    ## I know that this gauge section is not completely right yet
    # We would have to consider the direction of the connections that are being examined, i.e., U_{\mu,x}

    def gauge_gate_2(self, thetas, start_qubit):
        self.circuit.ry(thetas[0], start_qubit)
        self.circuit.cry(thetas[1], start_qubit, start_qubit+1)
        return self
    
    def gauge_gate_3(self,thetas,start_qubit):
        self.circuit.ry(thetas[0], start_qubit)
        self.circuit.ry(thetas[1], start_qubit+1)
        self.circuit.cry(thetas[2], start_qubit+1, start_qubit+2)
        self.circuit.cry(thetas[3], start_qubit, start_qubit+2)
        return self

    def gauge_block(self,thetas_slice,truncation):
        if truncation == 0:
            return self
        
        gauge_gates = {
            2: self.gauge_gate_2,
            3: self.gauge_gate_3
            }
        
        thetas_per_gauge = {
            2: 2,
            3: 4
        }
        
        qubits_per_gauge = int(np.ceil(np.log2(2*truncation+1)))
        for j in range(int(self.n_gauge_qubits/qubits_per_gauge)):
            thetas = thetas_slice[thetas_per_gauge[qubits_per_gauge]*j:thetas_per_gauge[qubits_per_gauge]*j+(thetas_per_gauge[qubits_per_gauge])]
            gauge_gates[qubits_per_gauge](thetas,qubits_per_gauge*j)
        self.circuit.barrier()
        return self

    def initialize_fermions(self):
        for k in range(self.n_gauge_qubits // 2, self.n_qubits // 2):
            self.circuit.x(2*k + 1)
        self.circuit.barrier()
        return self
    
    def build(self):
        return self.circuit.copy()

class Measurements_gpu:
    def __init__(self, simulator, n_qubits=0, meas_matrix=None):
        self.simulator = simulator
        self.n_qubits = n_qubits
        self.meas_matrix = meas_matrix  # stores calibration matrix for MEM

    # --------- Transpile Circuit (GPU-optimized) ---------
    def _transpile(self, circuit):
        return qiskit.transpile(
            circuit,
            backend=self.simulator,
            optimization_level=0,
            layout_method="trivial",
            seed_transpiler=1
        )

    # --------- Build & Run Measurement Circuit ---------
    def measure_circuit(self, circuit, terms, shots):
        circuit_copy = circuit.copy()

        for qubit, term in enumerate(terms):
            if term == 'X':
                circuit_copy.h(qubit)
            elif term == 'Y':
                circuit_copy.sdg(qubit)
                circuit_copy.h(qubit)

            if term in ['X', 'Y', 'Z']:
                circuit_copy.measure(qubit, qubit)

        transpiled = self._transpile(circuit_copy)
        job = self.simulator.run(transpiled, shots=shots)
        return job.result().get_counts(transpiled)

    # --------- Build MEM Calibration Matrix ---------
    def build_measurement_error_mitigation(self, shots=1024):
        qr = qiskit.QuantumRegister(self.n_qubits)
        cal_circuits = []
        labels = []
        for i in range(2**self.n_qubits):
            bitstr = format(i, f'0{self.n_qubits}b')
            qc = qiskit.QuantumCircuit(qr)
            for q, b in enumerate(bitstr):
                if b == '1':
                    qc.x(q)
            qc.measure_all()
            cal_circuits.append(qc)
            labels.append(bitstr)

        transpiled = [self._transpile(qc) for qc in cal_circuits]
        job = self.simulator.run(transpiled, shots=shots)
        res = job.result()

        n_states = 2**self.n_qubits
        M = np.zeros((n_states, n_states))
        for j, label in enumerate(labels):
            counts = res.get_counts(j)
            total = sum(counts.values())
            for k, v in counts.items():
                idx = int(k, 2)
                M[idx, j] = v / total
        self.meas_matrix = M

    # --------- Apply MEM Using Inverse of Calibration Matrix ---------
    def apply_meas_filter(self, counts):
        if self.meas_matrix is None:
            return counts

        n = 2**self.n_qubits
        p_meas = np.zeros(n)
        for bit, c in counts.items():
            idx = int(bit, 2)
            p_meas[idx] = c
        p_meas /= p_meas.sum()

        # invert calibration matrix
        try:
            p_true = np.linalg.solve(self.meas_matrix, p_meas)
        except np.linalg.LinAlgError:
            p_true = np.linalg.lstsq(self.meas_matrix, p_meas, rcond=None)[0]

        corrected_counts = {}
        total_shots = sum(counts.values())
        for i, p in enumerate(p_true):
            corrected_counts[format(i, f'0{self.n_qubits}b')] = max(p * total_shots, 0.0)
        return corrected_counts

    def zero_charge_counts_filter(self, counts, lattice):
        new_counts = {}
        for key in list(counts):
            s = 0
            for k in range(lattice.n_gauge_qubits, len(key)):
                indices = lattice.labels[k - lattice.n_gauge_qubits]
                s += (-1)**(indices[0] + indices[1]) * int(key[k])
            if s < 1e-8:
                new_counts[key] = copy.copy(counts[key])
        return new_counts   
    
    def gauge_truncation_filter(self, counts, lattice):
        new_counts = {}
        unacceptable_keys = {
            2 : '10', # I have checked these
            3 : '100'
            }
        for key in list(counts):
            score = 0
            for n in range(lattice.n_dynamical_links):
                part = key[n * lattice.qubits_per_gauge:(n + 1) * lattice.qubits_per_gauge]
                if part == unacceptable_keys[lattice.qubits_per_gauge]:
                    score += 1
            if score == 0:
                new_counts[key] = copy.copy(counts[key])
        return new_counts

    # --------- Expectation From Counts ---------
    def expected_value_from_counts(self, counts, term):
        total_shots = sum(counts.values())
        ev = 0
        for bitstring, count in counts.items():
            parity = 1
            for i, p in enumerate(term):
                if p != 'I' and bitstring[-1 - i] == '1':
                    parity *= -1
            ev += parity * count / total_shots
        return ev

    # --------- Expectation Value of Pauli Hamiltonian ---------
    def expected_value_hamiltonian(self, hamiltonian, circuit, shots=1024):
        total = 0
        for pauli_string, coefficient in hamiltonian.hamiltonian.items():

            if pauli_string == 'I' * len(pauli_string):
                ev = 1.0
            else:
                counts = self.measure_circuit(circuit, pauli_string, shots)
                if self.meas_matrix is not None:
                    counts = self.apply_meas_filter(counts)
                ev = self.expected_value_from_counts(counts, pauli_string)

            total += coefficient * ev
        return total

    def expected_value_hamiltonian_qed(self, hamiltonian, circuit, lattice, shots=1024):
        total = 0
        for pauli_string, coefficient in hamiltonian.hamiltonian.items():
            if pauli_string == 'I' * len(pauli_string):
                # print("Hit an I term")
                ev = 1.0
            else:
                counts = self.measure_circuit(circuit, pauli_string, shots)
                # print("counts:",counts)
                if self.meas_matrix is not None:
                    counts = self.apply_meas_filter(counts)
                counts = self.zero_charge_counts_filter(counts, lattice)
                counts = self.gauge_truncation_filter(counts, lattice)
                ev = self.expected_value_from_counts(counts, pauli_string)

            total += coefficient * ev
        return total

def Es_function(qubits_per_gauge):
    I_string_list = list('I' * qubits_per_gauge)
    coeff = -0.5
    Es = {}
    
    for k in range(qubits_per_gauge - 1):
        temp_string = copy.copy(I_string_list)
        temp_string[k] = 'Z'
        Es["".join(temp_string)] = coeff * 2**(k)
        
    first_string = copy.copy(I_string_list)    
    first_string[qubits_per_gauge - 1] = 'Z'
    Es["".join(first_string)] = coeff * (2**(qubits_per_gauge - 1) - 1)
    return Es

def electric_field_linear_term_n_direction(hamiltonian, lattice, n, direction, dynamical_links):
    # print("Linear EF called")
    index = lattice.labels[n]
    if (index, direction) in dynamical_links:
        link_index = lattice.dynamical_link_indexing[(index, direction)]
        fermion_string = 'I' * lattice.n_fermion_qubits

        gauge_before = 'I' * ((link_index) * lattice.qubits_per_gauge)
        gauge_after = 'I' * (lattice.n_dynamical_gauge_qubits - (link_index + 1) * lattice.qubits_per_gauge)

        Es = Es_function(lattice.qubits_per_gauge)

        for key in list(Es.keys()):
            term = gauge_before + key + gauge_after + fermion_string
            coeff = Es[key]
            hamiltonian.add_term(term, coeff)
    else:
        hamiltonian.add_term('I'*lattice.n_qubits, 0)

    return hamiltonian

def U_term_n(hamiltonian, lattice, n, direction, dynamical_links): 
    # print("U called")
    index = lattice.labels[n]
    if (index,direction) in dynamical_links:
        link_index = lattice.dynamical_link_indexing[(index, direction)]
        
        fermion_string = 'I' * lattice.n_fermion_qubits

        gauge_before = 'I' * (link_index * lattice.qubits_per_gauge)
        gauge_after = 'I' * (lattice.n_dynamical_gauge_qubits - (link_index + 1) * lattice.qubits_per_gauge)

        Us = {
                2: {
            "XI": 1.0,
            "XX": 1.0,
            "YI": -1.0j,
            "YX": 1.0j
            },        
        
                3:  { # This looks so wrong - not using 3 per gauge atm
            "IIX": 2.0,
            "IXX": 1.0,
            "XXX": 1.0,
            "IYX": 1.0j, 
            "XYX": 1.0j,
            "YII": 2.0j,
            "YXI": 1.0j,
            "YXX": 1.0j,
            "YYI": 1.0,
            "YYX": -1.0
            },
            
                4 : {
            "IIIX": 4.0,      # 4 X0
            "IIXX": 2.0,      # 2 X0 X1
            "IXXX": 1.0,      # X0 X1 X2
            "XXXX": 1.0,      # X0 X1 X2 X3
        
            "IXXY": 1.0j,     # i X0 X1 Y2
            "XXXY": 1.0j,    # i X0 X1 Y2 X3  (string below is correct)
        
            "IIYX": 2.0j,     # i*2 X0 Y1
            "IXYX": 1.0j,     # i X0 Y1 X2
            "XXYX": 1.0j,     # i X0 Y1 X2 X3
        
            "IYYX": 1.0,      # X0 Y1 Y2
            "XYYX": 1.0,     # - X0 Y1 Y2 X3
        
            "IIIY": 4.0j,     # i*4 Y0
            "IIXY": 2.0j,     # i*2 Y0 X1
            "IXXY": 1.0j,     # i Y0 X1 X2
            "XXXY": 1.0j,    # i Y0 X1 X2 X3 
        
            "IXYY": 1.0,      # Y0 X1 Y2
            "XXYY": -1.0,     # - Y0 X1 Y2 X3
        
            "IYYY": 2.0,      # 2 Y0 Y1
            "IXYY": -1.0,     # - Y0 Y1 X2
            "XXYY": -1.0,     # - Y0 Y1 X2 X3
        
            "IYYY": 1.0j,     # i Y0 Y1 Y2
            "XYYY": 1.0j      # i Y0 Y1 Y2 X3
            }
            }
        for key in list(Us[lattice.qubits_per_gauge]):
            term = gauge_before + key + gauge_after + fermion_string
            value = Us[lattice.qubits_per_gauge][key]
            hamiltonian.add_term(term, value*(1/(2**(lattice.qubits_per_gauge-1))))  
        
    else:
        hamiltonian.add_term('I'*lattice.n_qubits, 1.0)
    return hamiltonian

def magnetic_term_n(hamiltonian, lattice, n, dynamical_links):
    # print("MF called")
    index = lattice.labels[n]
    ns_directions = [
        (lattice.reverse_labels[(index[0],index[1])], 1),
        (lattice.reverse_labels[(index[0]+1,index[1])], 2),
        (lattice.reverse_labels[(index[0],index[1]+1)], 1),
        (lattice.reverse_labels[(index[0],index[1])], 2)
    ]

    Us = [Hamiltonian(lattice.n_qubits) for _ in range(4)]
    for i in range(4): Us[i] = U_term_n(Us[i], lattice, ns_directions[i][0], ns_directions[i][1], dynamical_links)
    Us[2].hamiltonian = Us[2].to_conjugate()
    Us[3].hamiltonian = Us[3].to_conjugate()

    P_n = Hamiltonian(lattice.n_qubits)
    P_n.add_term('I'*lattice.n_qubits, 1.0)

    for i in range(4): P_n.multiply_hamiltonians(Us[i])

    P_n.to_conjugate()

    P_n_dagger = Hamiltonian(lattice.n_qubits)
    P_n_dagger.hamiltonian = P_n.conjugate

    hamiltonian.add_hamiltonians(P_n)
    hamiltonian.add_hamiltonians(P_n_dagger)

    return hamiltonian

def particle_n_hamiltonian(lattice, indices):
    x, y = indices
    n = lattice.get_index(x, y)
    pn_hamiltonian = Hamiltonian(lattice.n_qubits)

    # Identity string
    I_term = 'I' * lattice.n_qubits
    Z_term = list(I_term)
    Z_term[lattice.n_dynamical_gauge_qubits + n] = 'Z'

    # bare occupancy: +0.5 I - 0.5 Z
    pn_hamiltonian.add_term(I_term, +0.5)
    pn_hamiltonian.add_term(''.join(Z_term), -0.5)

    # subtract Dirac-sea occupancy for odd sites
    n_vac = 1 if ((x + y) % 2 == 1) else 0
    if n_vac:
        pn_hamiltonian.add_term(I_term, -1.0)

    return pn_hamiltonian

def gauss_operator_n(lattice, n):
    G = Hamiltonian(lattice.n_qubits)

    for direction in lattice.directions[n]:
        G = electric_field_linear_term_n_direction(G, lattice, n, direction, lattice.dynamical_links_list)
    
    indices = lattice.labels[n]
    Q = charge_n_hamiltonian(lattice, indices)
    Q.hamiltonian = multiply_hamiltonian_by_constant(Q.hamiltonian, -1.0)
    G.add_hamiltonians(Q)

    return G

def charge_n_hamiltonian(lattice, indices):
    x, y = indices
    n = lattice.get_index(x, y)
    charge_hamiltonian = Hamiltonian(lattice.n_qubits)

    I_term = 'I' * lattice.n_qubits
    Z_term = list(I_term)
    Z_term[lattice.n_dynamical_gauge_qubits + n] = 'Z'

    # staggered phase
    parity = (-1)**(x + y)

    # bare staggered charge: -0.5 * parity * Z + 0.5 * I
    charge_hamiltonian.add_term(''.join(Z_term), -0.5 * parity)
    charge_hamiltonian.add_term(I_term, +0.5)

    return charge_hamiltonian




# Need to redo this:

class ObservableCalculator:
    def __init__(self, lattice, measurement_manager):
        #
        # Fermion qubits sit on the left end of the circuit
        #
        self.lattice = lattice
        self.measurer = measurement_manager

    def _fermion_qubit_index(self, indices):
        n = self.lattice.get_index(indices[0], indices[1])
        return self.lattice.n_dynamical_gauge_qubits + n

    # ============================================================
    # Existing fermionic observables (unchanged)
    # ============================================================

    def particle_n(self, indices, circuit, shots=1024):
        x, y = indices
        qidx = self._fermion_qubit_index(indices)
    
        pn = Hamiltonian(self.lattice.n_qubits)
        I = 'I' * self.lattice.n_qubits
        Z = list(I)
        Z[qidx] = 'Z'
    
        pn.add_term(I, +0.5)
        pn.add_term(''.join(Z), -0.5)
    
        n_vac = 1 if ((x + y) % 2 == 1) else 0
        pn.add_term(I, -n_vac)
    
        return self.measurer.expected_value_hamiltonian(pn, circuit, shots)

    def particle_number(self, circuit, shots=1024):
        total = 0.0
        for x in range(self.lattice.L_x):
            for y in range(self.lattice.L_y):
                total += self.particle_n((x, y), circuit, shots)
        return total

    def n_n(self, indices, circuit, shots=1024):
        x, y = indices
        qidx = self._fermion_qubit_index(indices)
    
        pn = Hamiltonian(self.lattice.n_qubits)
        Z = ['I'] * self.lattice.n_qubits
        Z[qidx] = 'Z'
    
        pn.add_term(''.join(Z), (-1) ** (x + y))
        return self.measurer.expected_value_hamiltonian(pn, circuit, shots)

    def chiral_condensate(self, circuit, shots=1024):
        total = 0.0
        for x in range(self.lattice.L_x):
            for y in range(self.lattice.L_y):
                total += self.n_n((x, y), circuit, shots)
        return total

    def charge_n(self, indices, circuit, shots=1024):
        x, y = indices
        qidx = self._fermion_qubit_index(indices)
    
        h = Hamiltonian(self.lattice.n_qubits)
        Z = ['I'] * self.lattice.n_qubits
        Z[qidx] = 'Z'
    
        parity = (-1) ** (x + y)
        h.add_term(''.join(Z), -0.5 * parity)
        h.add_term('I' * self.lattice.n_qubits, +0.5)
    
        return self.measurer.expected_value_hamiltonian(h, circuit, shots)

    def charge_total(self, circuit, shots=1024):
        total_charge = 0.0
        for x in range(self.lattice.L_x):
            for y in range(self.lattice.L_y):
                total_charge += self.charge_n((x, y), circuit, shots)
        return total_charge

    def energy(self, circuit, hamiltonian, shots=1024):
        return self.measurer.expected_value_hamiltonian(hamiltonian, circuit, shots)

    def electric_field_values(self, circuit, shots=1024):
        values = {}
        for n in range(self.lattice.n_fermion_qubits):
            for direction in self.lattice.directions[n]:
                h = Hamiltonian(self.lattice.n_qubits)
                h = electric_field_linear_term_n_direction(h, self.lattice, n, direction, self.lattice.dynamical_links_list)
                value = self.measurer.expected_value_hamiltonian(h, circuit, shots)
                values[(self.lattice.labels[n], direction)] = value

        return values

    def magnetic_field_values(self, circuit, shots=1024):
        values = {}
        for n in self.lattice.plaquettes:
            h = Hamiltonian(self.lattice.n_qubits)
            h = magnetic_term_n(h, self.lattice, n, self.lattice.dynamical_links_list)
            value = self.measurer.expected_value_hamiltonian(h, circuit, shots)
            values[self.lattice.labels[n]] = value

        return values

    def particle_number_values(self, circuit, shots=1024):
        values = {}
        for x in range(self.lattice.L_x):
            for y in range(self.lattice.L_y):
                pn = particle_n_hamiltonian(self.lattice, (x, y))
                value = self.measurer.expected_value_hamiltonian(pn, circuit, shots)
                values[(x, y)] = value

        return values

    def gauss_law_values(self, circuit, shots=1024):
        values = {}
        for x in range(self.lattice.L_x):
            for y in range(self.lattice.L_y):
                n = self.lattice.reverse_labels[(x, y)]
                g_n = gauss_operator_n(self.lattice, n)
                value = self.measurer.expected_value_hamiltonian(g_n, circuit, shots)
                values[(x, y)] = value

        return values