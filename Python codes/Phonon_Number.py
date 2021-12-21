from qutip import *
import numpy as np
qutip.settings.has_mkl=False

def convert_to_GHz():
    '''
    Returns conversion factor for meV -> GHz
    '''
    return convert_unit(1, orig='meV', to='GHz') # conversion ratio from meV to GHz



def calculate_nth(omega_m, temperature):
    '''
    Calculates the average occupation of a mode with given frequency and temperature
    
    Parameters
    ----------
    omega_m: float
        frequency of mode in GHz
    temperature: float
        temperature of mode in K
        
    Returns
    -------
    nth: float
        nth = 1/(exp(omega_m/(k_B*temperature)) - 1)
    '''
    
    kb = 20.864950997206705 # Boltzmann constant in GHz/K
    nth = 1/(np.exp(omega_m/(kb*temperature)) - 1)
    return nth

    

def define_three_level_basis(N):
    '''
    Calculates the basis states for the three-level system in the combined system of three-level system + mechanical resonator
    
    Parameters
    ----------
    N: int
        size of the fock state basis
        
    Returns
    ----------
    sig11, sig22, sig01, sig02, sig12: Qobj
        three-level system transition operators sig_xy = |x><y|
    '''
    
    # basis states
    psi0 = basis(3,0)
    psi1 = basis(3,1)
    psi2 = basis(3,2)

    # Population opertors
    sig11 = tensor(psi1*psi1.dag(), qeye(N))
    sig22 = tensor(psi2*psi2.dag(), qeye(N))

    # Transition operators sig_xy = |x><y|:
    sig01 = tensor(psi0*psi1.dag(), qeye(N))
    sig02 = tensor(psi0*psi2.dag(), qeye(N))
    sig12 = tensor(psi1*psi2.dag(), qeye(N))

    return sig11, sig22, sig01, sig02, sig12



def destruction_operator(N):
    '''
    Calculates the destruction operator for the mechanical mode
    
    Parameters
    ----------
    N: int
        size of the fock state basis
        
    Returns
    ----------
    a: Qobj
        destruction operator
    '''
    
    a = tensor(qeye(3), destroy(N))
    return a
    


def steady_state_phonon(Hamiltonian, collapse_ops, N):
    '''
    Calculates the steady state occupation of the mechanical mode
    
    Parameters
    ----------
    Hamiltonian: Qobj
        Hamiltonian of the system
    collapse_ops: dictionaty of Qobj
        collapse operators for the combined system
    N: int
        size of the fock state basis
        
    Returns
    ----------
    nss: float
        steady state phonon number
    '''

    # Calculate the steady state using the Hamiltonian and the collapse operators
    steady_state = steadystate(Hamiltonian, collapse_ops)
    
    # Define destruction operator
    a = destruction_operator(N)
    
    # Calculate steady state phonon number
    nss = expect(a.dag()*a, steady_state)
    
    return nss

    
    
def three_level_system(delta_1, delta_2, g, pump, T2, T1, gamma, nth, N):
    '''
    Calculates the Hamiltonian and collapse operators for the three-level system coupled to the mechanical resonator mode
    
    Parameters
    ----------
    delta_1: float
        omega_1 - omega_p; in GHz
    delta_2: float
        omega_2 - omega_1 - omega_m; in GHz
    g: float
        coupling strength; in GHz
    pump: float
        pumping strength; in GHz
    T2: float
        decay rate for the second excited state: |2>; in GHz
    T1: float
        decay rate for the first excited state: |1>; in GHz
    gamma: float
        decay rate for the mechanical mode; in GHz
    nth: float
        average occupation of the mechanical mode
    N: int
        size of the fock state basis
        
    Returns
    ----------
    Hamiltonian: Qobj
        Hamiltonian of the system
    collapse_ops: dictionaty of Qobj
        collapse operators for the combined system
    '''
    
    sig11, sig22, sig01, sig02, sig12 = define_three_level_basis(N)
    a = destruction_operator(N)
    
    Hamiltonian = delta_1*sig11 + (delta_2 + delta_1)*sig22 + g*(sig12*a.dag() + sig12.dag()*a) + pump*(sig01 + sig01.dag())
    collapse_ops = [np.sqrt(T1)*sig01, np.sqrt(T2)*sig02, np.sqrt(gamma*(nth + 1))*a, np.sqrt(gamma*nth)*a.dag()]
    
    return Hamiltonian, collapse_ops
    
    

def detuning(delta_1, delta_2, omega_21, temperature, g, pump, T2, T1, gamma, N):
    '''
    Calculate steady state phonon number for a three level system
    
    Parameters
    ----------
    delta_1: float
        omega_1 - omega_p; in GHz
    delta_2: float
        omega_2 - omega_1 - omega_m; in GHz
    omega_21: float
        omega_2 - omega_1; in GHz
    temperature: float
        initial temperature of system; in K
    g: float
        coupling strength; in GHz
    pump: float
        pumping strength; in GHz
    T2: float
        decay rate for the second excited state: |2>; in GHz
    T1: float
        decay rate for the first excited state: |1>; in GHz
    gamma: float
        decay rate for the mechanical mode; in GHz
    N: int
        size of the fock state basis
        
    Returns
    ----------
    nss: float
        steady state phonon number
    '''

    # nth changes with delta_2
    # omega_21 = omega_2 - omega_1; delta2 = omega_2 - omega_1 - omega_m
    # => omega_m = omega_21 - delta_2
    nth = calculate_nth((omega_21 - delta_2), temperature)
    
    Hamiltonian, collapse_ops = three_level_system(delta_1, delta_2, g, pump, T2, T1, gamma, nth, N)

    nss = steady_state_phonon(Hamiltonian, collapse_ops, N)

    return nss/nth



def QD_theory(g, pump, T2, T1, gamma, nth): 
    '''
    Calculates the nss/nth for QD case as per the theoretical expression derived in the mathematica notebook
    
    Parameters
    ----------
    g: float
        coupling strength; in GHz
    pump: float
        pumping strength; in GHz
    T2: float
        decay rate for the second excited state: |2>; in GHz
    T1: float
        decay rate for the first excited state: |1>; in GHz
    gamma: float
        decay rate for the mechanical mode; in GHz
    nth: float
        average occupation of the mechanical mode
        
    Returns
    ----------
    nss: float
        steady state phonon number
    '''
    
    # numerator already divided by nth
    numerator = (gamma*(2*T2*(g**4 + 4*pump**4 + pump**2*T2**2) + gamma*(4*g**2*(g**2 + 2*pump**2)*(1 + nth) + T2**2*(2*(g**2 + 2*pump**2) + (g**2 + 4*pump**2)*nth))))
    denominator = (2*T2*(2*g**2*pump**2*T2 + gamma*(2*(g**4 + g**2*pump**2 + 2*pump**4) + (3*g**4 + 4*g**2*pump**2 + 8*pump**4)*nth + pump**2*(1 + 2*nth)*T2**2)))
    
    return numerator/denominator



def QD_system(g, pump, T2, T1, gamma, dephasing, nth, N):
    '''
    Calculates the Hamiltonian and collapse operators for QD + confined phonon mode system
    
    Parameters
    ----------
    g: float
        coupling strength; in GHz
    pump: float
        pumping strength; in GHz
    T2: float
        decay rate for the second excited state: |2>; in GHz
    T1: float
        decay rate for the first excited state: |1>; in GHz
    gamma: float
        decay rate for the mechanical mode; in GHz
    dephasing: float
        dephasing rate. assumed to be the same for the excited states |1> and |2>
    nth: float
        average occupation of the mechanical mode
    N: int
        size of the fock state basis
        
    Returns
    ----------
    Hamiltonian: Qobj
        Hamiltonian of the system
    collapse_ops: dictionaty of Qobj
        collapse operators for the combined system
    '''
    
    sig11, sig22, sig01, sig02, sig12 = define_three_level_basis(N)
    a = destruction_operator(N)
    
    Hamiltonian = g*(sig12*a.dag() + sig12.dag()*a) + pump*(sig01 + sig01.dag())
    collapse_ops = [np.sqrt(T1)*sig01, np.sqrt(T2)*sig02, np.sqrt(gamma*(nth + 1))*a, np.sqrt(gamma*nth)*a.dag(), np.sqrt(dephasing)*sig11, np.sqrt(dephasing)*sig22]
    
    return Hamiltonian, collapse_ops



def QD_simulation(g, pump, T2, T1, gamma, dephasing, nth, N):
    '''
    Calculate steady state phonon number for the QD model
    
    Parameters
    ----------
    g: float
        coupling strength; in GHz
    pump: float
        pumping strength; in GHz
    T2: float
        decay rate for the second excited state: |2>; in GHz
    T1: float
        decay rate for the first excited state: |1>; in GHz
    gamma: float
        decay rate for the mechanical mode; in GHz
    dephasing: float
        dephasing rate. assumed to be the same for the excited states |1> and |2>
    nth: float
        average occupation of the mechanical mode
    N: int
        size of the fock state basis
        
    Returns
    ----------
    nss: float
        steady state phonon number
    '''

    # Define the QD system
    Hamiltonian, collapse_ops = QD_system(g, pump, T2, T1, gamma, dephasing, nth, N)
    
    # Calculate the steady state phonon number
    nss = steady_state_phonon(Hamiltonian, collapse_ops, N)
    
    return nss/nth



# def incoherent_theory(g, pump, T2, gamma, T1, nth):
#     num1 = gamma*(8*g**2 + T2**2)
#     num2 = 8*g**2*T2*gamma**2*nth
#     dem1 = 4*g**2*T2 + gamma*(T2**2*(1 + 2*nth) + 8*g**2*(1 + 3*nth))
#     dem2 = dem1*pump**2
#     return num1/dem1 + num2/dem2



def incoherent_system(g, pump, T2, T1, gamma, nth, N):
    '''
    Calculates the Hamiltonian and collapse operators for the three-level system that is incoherently pumped
    
    Parameters
    ----------
    g: float
        coupling strength; in GHz
    pump: float
        pumping strength; in GHz
    T2: float
        decay rate for the second excited state: |2>; in GHz
    T1: float
        decay rate for the first excited state: |1>; in GHz
    gamma: float
        decay rate for the mechanical mode; in GHz
    nth: float
        average occupation of the mechanical mode
    N: int
        size of the fock state basis
        
    Returns
    ----------
    Hamiltonian: Qobj
        Hamiltonian of the system
    collapse_ops: dictionaty of Qobj
        collapse operators for the combined system
    '''
    
    sig11, sig22, sig01, sig02, sig12 = define_three_level_basis(N)
    a = destruction_operator(N)

    Hamiltonian = g*(sig12*a.dag() + sig12.dag()*a)
    collapse_ops = [np.sqrt(T1)*sig01, np.sqrt(T2)*sig02, np.sqrt(gamma*(nth + 1))*a, np.sqrt(gamma*nth)*a.dag(), np.sqrt(pump)*sig01.dag()]
    
    return Hamiltonian, collapse_ops
    
    

def incoherent_simulation(g, pump, T2, T1, gamma, nth, N):
    '''
    Calculate steady state phonon number for the QD model
    
    Parameters
    ----------
    g: float
        coupling strength; in GHz
    pump: float
        pumping strength; in GHz
    T2: float
        decay rate for the second excited state: |2>; in GHz
    T1: float
        decay rate for the first excited state: |1>; in GHz
    gamma: float
        decay rate for the mechanical mode; in GHz
    nth: float
        average occupation of the mechanical mode
    N: int
        size of the fock state basis
        
    Returns
    ----------
    nss: float
        steady state phonon number
    '''
    # Define the incoherent system
    Hamiltonian, collapse_ops = incoherent_system(g, pump, T2, T1, gamma, nth, N)
    
    # Calculate the steady state phonon number
    nss = steady_state_phonon(Hamiltonian, collapse_ops, N)
    
    return nss/nth



def polariton_system(g, pump, T2, gamma, dephasing, incoherent_pump, nth, N):
    '''
    Calculates the Hamiltonian and collapse operators for polariton + nanomechanical resonator system
    
    Parameters
    ----------
    g: float
        coupling strength; in MHz
    pump: float
        pumping strength; in MHz
    T2: float
        decay rate for the second excited state: |2>; in MHz
    gamma: float
        decay rate for the mechanical mode; in MHz
    dephasing: float
        dephasing rate; in MHz. assumed to be the same for the excited states |1> and |2>
    incoherent_pump: float
        incoherent pumpind rate; in MHz. assumed to be the same for the excited states |1> and |2>
    nth: float
        average occupation of the mechanical mode
    N: int
        size of the fock state basis
        
    Returns
    ----------
    Hamiltonian: Qobj
        Hamiltonian of the system
    collapse_ops: dictionaty of Qobj
        collapse operators for the combined system
    '''
    sig11, sig22, sig01, sig02, sig12 = define_three_level_basis(N)
    a = destruction_operator(N)
    
    Hamiltonian = g*(sig12*a.dag() + sig12.dag()*a)/2 - pump*(sig01 + sig01.dag())/np.sqrt(2)
    collapse_ops = [np.sqrt(T2)*sig01, np.sqrt(T2)*sig02, np.sqrt(gamma*(nth + 1))*a, np.sqrt(gamma*nth)*a.dag(), \
            np.sqrt(dephasing)*sig11, np.sqrt(dephasing)*sig22, np.sqrt(incoherent_pump)*sig01.dag(), np.sqrt(incoherent_pump)*sig02.dag()]
    
    return Hamiltonian, collapse_ops
    
    
    
def polariton_simulation(g, pump, T2, gamma, dephasing, incoherent_pump, nth, N):
    '''
    Calculate steady state phonon number for the QD model
    
    Parameters
    ----------
    g: float
        coupling strength; in MHz
    pump: float
        pumping strength; in MHz
    T2: float
        decay rate for the second excited state: |2>; in MHz
    gamma: float
        decay rate for the mechanical mode; in MHz
    dephasing: float
        dephasing rate; in MHz. assumed to be the same for the excited states |1> and |2>
    incoherent_pump: float
        incoherent pumpind rate; in MHz. assumed to be the same for the excited states |1> and |2>
    nth: float
        average occupation of the mechanical mode
    N: int
        size of the fock state basis
        
    Returns
    ----------
    nss: float
        steady state phonon number
    '''
    # Define the incoherent system
    Hamiltonian, collapse_ops = polariton_system(g, pump, T2, gamma, dephasing, incoherent_pump, nth, N)
    
    # Calculate the steady state phonon number
    nss = steady_state_phonon(Hamiltonian, collapse_ops, N)
    
    return nss/nth



def polariton_theory(g, pump, T2, gamma, nth): 
    '''
    Calculates the nss/nth for polariton case as per the theoretical expression derived in the mathematica notebook
    
    Parameters
    ----------
    g: float
        coupling strength; in GHz
    pump: float
        pumping strength; in GHz
    T2: float
        decay rate for the second excited state: |2>; in GHz
    gamma: float
        decay rate for the mechanical mode; in GHz
    nth: float
        average occupation of the mechanical mode
        
    Returns
    ----------
    nss: float
        steady state phonon number
    '''
    numerator = (gamma*(2*T2*(3*g**4*pump**2 + 16*pump**6 + (g**4 + 2*g**2*pump**2 + 24*pump**4)*T2**2 + \
    (2*g**2 + 9*pump**2)*T2**4 + T2**6) + gamma*(4*g**2*pump**2*(g**2 + 4*pump**2)*(1 + nth) + 3*T2**6*(3 + 5*nth) + \
    2*T2**4*(5*g**2 + 27*pump**2 + (11*g**2 + 42*pump**2)*nth) + T2**2*(g**4 + 30*g**2*pump**2 + 72*pump**4 + \
    (7*g**4 + 16*g**2*pump**2 + 96*pump**4)*nth))))

    denominator = (2*T2*(2*g**2*pump**2*T2*(4*pump**2 + T2**2) + gamma*((4*pump**2 + T2**2)*(g**4 + g**2*pump**2 + 4*pump**4 + \
    (2*g**2 + 5*pump**2)*T2**2 + T2**4) + (7*g**4*pump**2 + 12*g**2*pump**4 + 32*pump**6 + T2**2*(2*g**4 + 19*g**2*pump**2 + 48*pump**4 + \
    2*T2**2*(2*g**2 + 9*pump**2 + T2**2)))*nth)))

    return numerator/denominator


def check_ss(phonon, phonon_ss):
    '''
    Checks if the time dynamic simulation is in steady state using the corresponding phonon number array and steady state phonon occupation
    
    Parameters
    -----------
    phonon: numpy array
        phonon number array from simulations
    phonon_ss: float
        steady state phonon number
    
    Returns
    ----------
    
    True: if the in steady state
    False: otherwise
    '''
    
    # set tolerance for steady state condition
    tolerance = 10**-2
    
    # If average of last 40 values is wihitn tolerance% of the steady state phonon number value
    if ((np.abs(phonon_ss - np.sum(phonon[-40:])/len(phonon[-40:])))/phonon_ss  < tolerance):
        return True
    else:
        return False



def stretched_exponential(time, decay_rate, beta, phonon_ss, nth):
    '''
    Calculate and return the stretched exponential fit function
    
    Parameters
    ----------
    time: numpy array
        array of time values of the simulation
    decay_rate: float
        effective decay rate of the phonon number simulation
    beta: float
        stretch parameter. 0 < beta < 1 
    phonon_ss: float
        steady state phonon number
    nth: float
        average occupation of the mechanical mode
        
    Returns
    ----------
    
    stretched exponential fit array
    '''
    
    # Fit function as in Eq 8 in manuscript
    return (nth-phonon_ss)*np.exp(-(decay_rate*time)**beta) + phonon_ss



def find_index(phonon, phonon_ss):
    '''
    Locate the index where phonon(t) becomes <= 99% of the steady state phonon.
    
    Parameters
    ----------
    phonon: numpy array
        phonon number array from simulations
    phonon_ss: float
        steady state phonon number
    
    Returns
    ----------
    index: int
        the required index
    '''
    
    index = 0
    
    # Loop and increase the value of index until phonon(t) becomes <= 99%
    while(phonon[index]*0.99 > phonon_ss):
        index = index + 1
        
    return index


def calculate_initial_state(N, nth):
    '''
    Calculate the combined inital state of the system: 
    Ground state for three-level system and thermal state for mechanical resonator mode
    
    Parameters
    -----------
    N: int
        size of the fock state basis
    nth: float
        average occupation of the mechanical mode
        
    Returns
    ----------
    initial_state: Qobj
        initial state for the combined system
    '''
    
    # Initial state for mechanical mode
    thermal_state = thermal_dm(N, nth)
    
    # Initial state for three-level system
    ground_state = ket2dm(basis(3,1))
    
    # Combined initial state: tensor product of the density matrices
    initial_state = tensor(ground_state, thermal_state)
    
    return initial_state

def calculate_phonon_number(state, N):
    '''
    Calculate the expectation value of phonon number in the given state
    
    Parameters
    -----------
    state: Qobj
        state of the system
    N: int
        size of the fock state basis in the system
        
    Returns
    ----------
    phonon_number: float
        expectation value of phonon number
    '''
    
    # Define the destruction operator
    a = destruction_operator(N)
    
    phonon_number = expect(a.dag()*a, state)
    
    return phonon_number

def master_eq_solver(Hamiltonian, initial_state, time, collapse_ops, N):
    '''
    Calculate phonon(t) using the master equation
    
    Parameters
    ----------
    Hamiltonian: Qobj
        Hamiltonian of the system
    initial_state: Qobj
        initial state for the combined system
    time: numpy array
        array of time values of the simulation
    collapse_ops: dictionaty of Qobj
        collapse operators for the combined system
    N: int
        size of the fock state basis in the system
        
    Returns
    ----------
    phonon: numpy array
        phonon(t) array
    '''
    
    # define destruction operator
    a = destruction_operator(N)
    
    # run time dynamics
    result = mesolve(Hamiltonian, initial_state, time, collapse_ops, [a.dag()*a])
    phonon = result.expect[0]    # store phonon number time dynamics
    
    return phonon


def calculate_average_error(best_fit, phonon):
    '''
    Calculate average error between best fit and phonon arrays
    
    Parameters
    ----------
    best_fit: numpy array
        fit obtained from scipy.minimize
    phonon: numpy array
        phonon(t) array
    
    Return
    ----------
    average_error: float
        average error between best fit and phonon arrays
    '''
    
    # Diviving by 5000 as both the arrays have 5000 elements
    average_error = np.sum(np.abs((best_fit - phonon))/phonon)/5000
    
    return average_error