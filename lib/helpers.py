import numpy as np

def complex_to_vec(c):
  return np.array([np.imag(c), np.real(c)])

def sign_no_zero(x):
  return np.where(x < 0.0, -1.0, 1.0)