from modules import *

def SPSA(cost_function, guess, SPSA_parameters):
    _costs, _grads, _paras = [], [], []

    paras = np.array(guess, dtype=float)
    n_paras = len(paras)

    A = 0.1 * SPSA_parameters['max_iters']
    alpha = 0.602
    gamma = 0.101
    
    a = SPSA_parameters['a']
    c = SPSA_parameters['c']
    max_iters = SPSA_parameters['max_iters']
    grad_tol = SPSA_parameters['grad_tol']
    avg_len = SPSA_parameters['average_length']
    avg_tol = SPSA_parameters['average_tol']

    stop_reason = None
    pi_over_2 = np.pi / 2
    iteration = 1

    def stopping_criterion(costs, l, tol):
        if len(costs) < l + 1:
            return False
        old_avg = np.mean(costs[-l-1:-1])
        new_avg = np.mean(costs[-l:])
        return abs(old_avg - new_avg) < tol

    while iteration <= max_iters:
        delta = np.random.choice([-1.0, 1.0], size=n_paras)
        
        shift = c / (iteration ** gamma)
        
        cost_plus = cost_function(paras + shift * delta)
        cost_minus = cost_function(paras - shift * delta)

        grad_vector = (cost_plus - cost_minus) / (2 * shift * delta)
        step_size = a / ((A + iteration) ** alpha)
        paras -= step_size * grad_vector

        _paras.append(paras.copy())
        _grads.append(grad_vector.copy())
        _costs.append(cost_function(paras))
        '''
        if np.linalg.norm(grad_vector) < grad_tol:
            stop_reason = 'grad_tol'
            break
        '''
        if stopping_criterion(_costs, avg_len, avg_tol):
            stop_reason = 'average_tol'
            break

        iteration += 1
        
        if ((SPSA_parameters['prints']) and (iteration % 50 == 0)): print("Current energy:",_costs[-1])

    if stop_reason is None:
        stop_reason = 'max_iters'

    results = {
        'paras': _paras,
        'costs': _costs,
        'grads': _grads,
        'final_paras': _paras[-1],
        'final_cost': _costs[-1],
        'final_grads': _grads[-1],
        'stop_reason': stop_reason
    }
    return results

def NG_SPSA(cost_function, guess, SPSA_parameters):
    _costs, _grads, _paras, delta_Es = [], [], [], []

    paras = np.array(guess, dtype=float)
    n_paras = len(paras)

    # --- SPSA schedules ---
    max_iters = SPSA_parameters['max_iters']
    a0 = SPSA_parameters['a'] * 2       
    c0 = SPSA_parameters['c']
    alpha = 0.602
    gamma = 0.101
    # A = 0.5 * max_iters                     
    A = 500

    avg_len = SPSA_parameters['average_length']
    avg_tol = SPSA_parameters['average_tol']

    # --- Natural gradient parameters ---
    beta = 0.9                             
    F_diag = np.ones(n_paras)               
    lambda_reg = 1e-4                       

    stop_reason = None
    iteration = 1

    def stopping_criterion(costs, l, tol):
        if len(costs) < l + 1:
            return False
        old_avg = np.mean(costs[-l-1:-1])
        new_avg = np.mean(costs[-l:])
        return abs(old_avg - new_avg) < tol
        
    if SPSA_parameters['diagnostics']:
        print("SPSA_parameters:",SPSA_parameters)
        print("a0:",a0)
        print("c0:",c0)
        print("alpha:",alpha)
        print("gamma:",gamma)
        print("A:",A)
        print("beta:",beta)
        print("lambda_reg:",lambda_reg)

    while iteration <= max_iters:
        # --- SPSA gradient ---
        tok_full = time.perf_counter()
        delta = np.random.choice([-1.0, 1.0], size=n_paras)
        ck = c0 / (iteration ** gamma)
        
        tok_both_calls = time.perf_counter()
        cost_plus = cost_function(paras + ck * delta)
        cost_minus = cost_function(paras - ck * delta)
        tik_both_calls = time.perf_counter()
        
        grad = (cost_plus - cost_minus) / (2 * ck * delta)

        # --- Natural metric update (diagonal Fisher) ---
        F_diag = beta * F_diag + (1 - beta) * (grad ** 2)
        F_eff = np.maximum(F_diag, lambda_reg)

        # --- Step size ---
        ak = a0 / ((A + iteration) ** alpha)

        # --- Natural gradient step ---
        paras -= ak * grad / F_eff

        # --- Bookkeeping ---
        _paras.append(paras.copy())
        _grads.append(grad.copy())
        _costs.append(cost_function(paras))

        if stopping_criterion(_costs, avg_len, avg_tol):
            stop_reason = 'average_tol'
            break

        iteration += 1
        if SPSA_parameters['prints'] and iteration % 50 == 0:
            print("-"*10)
            print("Iteration:",iteration)
            print("Current energy:", _costs[-1])
        
        tik_full = time.perf_counter()
        time_taken_full = tik_full - tok_full
        time_taken_both_calls = tik_both_calls - tok_both_calls
        time_taken_classical = time_taken_full - time_taken_both_calls
        
        time_taken_full = smart_round(time_taken_full, 6)
        time_taken_both_calls = smart_round(time_taken_both_calls, 6)
        time_taken_classical = smart_round(time_taken_classical, 6)
            
        if SPSA_parameters['diagnostics'] and (iteration % 50 == 0) and (iteration > 1):
            parameter_velocity = _paras[-1] - _paras[-2]
            delta_E = _costs[-1] - _costs[-2]
            delta_Es.append(delta_E)
            delta_Es = delta_Es[-20:]
            R = np.mean(delta_Es) / (np.std(delta_Es) + 0.00001)
            print("-"*10)
            print("Iteration:",iteration)
            print("Energy:",_costs[-1])
            print("Grad norm (|grad|):",np.linalg.norm(grad.copy()))
            print("Parameter velocity norm (|_paras[-1] - _paras[-2]|):", np.linalg.norm(parameter_velocity))
            print("min(F_diag), max(F_diag), np.std(F_diag):", min(F_diag), max(F_diag), np.std(F_diag))
            print("delta_E:", delta_E)
            print("R:", R)
            print("E_plus - E_minus:",cost_plus - cost_minus)
            print("time_taken_full:", time_taken_full)
            print("time_taken_both_calls:", time_taken_both_calls)
            print("time_taken_classical:", time_taken_classical)
        

    if stop_reason is None:
        stop_reason = 'max_iters'

    return {
        'paras': _paras,
        'costs': _costs,
        'grads': _grads,
        'final_paras': _paras[-1],
        'final_cost': _costs[-1],
        'final_grads': _grads[-1],
        'stop_reason': stop_reason
    }

def SPSA_parameter_shift(cost_function, guess, SPSA_parameters):
    _costs, _grads, _paras = [], [], []

    paras = np.array(guess, dtype=float)
    n_paras = len(paras)

    A = 0.1 * SPSA_parameters['max_iters']
    alpha = 0.602
    a = SPSA_parameters['a']
    max_iters = SPSA_parameters['max_iters']
    grad_tol = SPSA_parameters['grad_tol']
    avg_len = SPSA_parameters['average_length']
    avg_tol = SPSA_parameters['average_tol']

    stop_reason = None
    pi_over_2 = np.pi / 2
    iteration = 1

    def stopping_criterion(costs, l, tol):
        if len(costs) < l + 1:
            return False
        old_avg = np.mean(costs[-l-1:-1])
        new_avg = np.mean(costs[-l:])
        return abs(old_avg - new_avg) < tol

    while iteration <= max_iters:
        delta = np.random.choice([-1.0, 1.0], size=n_paras)
        cost_plus = cost_function(paras + pi_over_2 * delta)
        cost_minus = cost_function(paras - pi_over_2 * delta)

        grad_vector = (cost_plus - cost_minus) / (2 * delta)
        step_size = a / ((A + iteration) ** alpha)
        paras -= step_size * grad_vector

        _paras.append(paras.copy())
        _grads.append(grad_vector.copy())
        _costs.append(cost_function(paras))

        if np.linalg.norm(grad_vector) < grad_tol:
            stop_reason = 'grad_tol'
            break
        if stopping_criterion(_costs, avg_len, avg_tol):
            stop_reason = 'average_tol'
            break

        iteration += 1
        
        if SPSA_parameters['prints']: print("Current energy:",_costs[-1])

    if stop_reason is None:
        stop_reason = 'max_iters'

    results = {
        'paras': _paras,
        'costs': _costs,
        'grads': _grads,
        'final_paras': _paras[-1],
        'final_cost': _costs[-1],
        'final_grads': _grads[-1],
        'stop_reason': stop_reason
    }
    return results