import random
def check_delta_sat(rho_value: float, delta: float) -> bool:
    threshold = 5.0*delta
    if rho_value > threshold: return True
    if rho_value < 0: return False
    p = rho_value / max(threshold, 1e-6)
    return random.random() < p*0.2
