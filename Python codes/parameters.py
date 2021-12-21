import Phonon_Number as pn

def three_level_system_params():
    '''
    Returns parameters for three-level system
    
    Returns
    ----------
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
    '''
    
    omega_21 = 0.5*pn.convert_to_GHz()  # in meV -> GHz
    temperature = 50                    # in K
    g = 5                               # in GHz
    pump = g/2                          # in GHz
    T2 = 0.5*g                          # in GHz
    T1 = 10**-1*g                       # in GHz
    gamma = 10**-4*g                    # in GHz
    
    return omega_21, temperature, g, pump, T2, T1, gamma


def QD_params():
    '''
    Returns paramaters for the QD model
    
    Reference: M. Khosla, S. Rao, and S. Gupta: Scientific Reports, vol. 8, no. 1, may 2018. 
    
    Returns
    ----------
    omega_m: float
        mechanical resonator frequency; in GHz
    g: float
        coupling strength; in GHz
    T1: float
        decay rate for the first excited state: |1>; in GHz
    gamma: float
        decay rate for the mechanical mode; in GHz
    nth: float
        average occupation of the mechanical mode
    '''
    omega_m = 1*pn.convert_to_GHz()                 # phonon frequency; in meV -> GHz
    g = 20                                          # in GHz
    T1 = 10**-6                                     # in GHz
    gamma = 10**-3                                  # in GHz
    temperature = 17                                # in K
    nth = pn.calculate_nth(omega_m, temperature)
    
    return omega_m, g, T1, gamma, temperature, nth

def polariton_params():
    '''
    Returns parameters for the polariton model
    
    References: 
    1) AppliedPhysics Letters, vol. 103, no. 24, p. 241112, 2013.
    2) Opt. Express, vol. 25, no. 20, pp. 24 639-24 649, Oct 2017
    3) https://iopscience.iop.org/article/10.1088/0268-1242/23/12/123001
    4) Phys. Rev. Lett., vol. 95, p. 067401, Aug 2005
    
    Returns
    ----------
    G: float
        coupling strength between TLS and cavity; in MHz
    omega_m: float
        mechanical resonator frequency; in MHz
    g: float
        coupling strength; in MHz
    gamma: float
        decay rate for the mechanical mode; in MHz
    dephasing: float
        dephasing rate; in MHz. assumed to be the same for the excited states |1> and |2>
    incoherent_pump: float
        incoherent pumpind rate; in MHz. assumed to be the same for the excited states |1> and |2>
    '''
    
    G = 5*10**3                               # coupling strength between TLS and cavity; in MHz
    omega_m = 2*G                             # mechanical resonator frequency; in MHz
    g = 0.002*G                               # coupling strength between mechanical resonator and polariton; in MHz
    gamma = omega_m*10**-7                    # in MHz
    temperature = 2.63                        # in K
    nth = pn.calculate_nth(omega_m/10**3, temperature) # converting omega_m to GHz

    return G, omega_m, g, gamma, temperature, nth


def incoherent_system_params():
    '''
    Returns parameters for incoherent pumping model
    
    Returns
    ----------
    g: float
        coupling strength; in GHz
    pump: float
        pumping strength; in GHz
    T1: float
        decay rate for the first excited state: |1>; in GHz
    nth: float
        average occupation of the mechanical mode
    '''
    g = 1                                       # in GHz
    pump = 10*g                                 # in GHz
    T1 = g*10**-5                               # in GHz
    nth = 200                                   # value taken from the reference; see Appendix G

    return g, pump, T1, nth