import numpy as np

def _nuclei_density(field, data):
    return data[("gas", "number_density")] * data[("flash", "ihp ")]

def _electron_density(field, data):
    return data[("gas", "density")] * data[("flash", "ihp ")]
