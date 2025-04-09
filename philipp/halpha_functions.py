import numpy as np

def j_Halpha_R(T, ne, nHp):
    """
    Computes the recombination H-alpha emissivity.

    Parameters:
    T  : float - Temperature in Kelvin
    ne : float - Electron number density in cm^-3
    nHp: float - Proton number density in cm^-3

    Returns:
    float - H-alpha emissivity in erg cm^-3 s^-1 sr^-1
    """
    T4 = T / 1e4
    return 2.82e-26 * T4**(-0.942 - 0.031 * np.log(T4)) * ne * nHp

def Gamma_13(T):
    """
    Computes the effective collision strength Gamma_13(T).

    Parameters:
    T : float - Temperature in Kelvin

    Returns:
    float - Effective collision strength
    """
    if 4000 <= T <= 25000:
        return (0.35 - 2.62e-7 * T - 8.15e-11 * T**2 + 6.19e-15 * T**3)
    elif 25000 < T <= 500000:
        return (0.276 + 4.99e-6 * T - 8.85e-12 * T**2 + 7.18e-18 * T**3)
    else:
        raise ValueError("Temperature out of valid range (4000 K to 500000 K)")

def j_Halpha_C(T, ne, nH):
    """
    Computes the collisional excitation H-alpha emissivity.

    Parameters:
    T  : float - Temperature in Kelvin
    ne : float - Electron number density in cm^-3
    nH : float - Neutral hydrogen number density in cm^-3

    Returns:
    float - H-alpha emissivity in erg cm^-3 s^-1 sr^-1
    """
    kB = 8.617333262145e-5  # Boltzmann constant in eV/K
    Gamma13_T = Gamma_13(T)
    prefactor = 1.30e-17 / (4 * np.pi)
    exponent = np.exp(-12.1 / (kB * T))
    
    return prefactor * (Gamma13_T / np.sqrt(T)) * exponent * ne * nH
