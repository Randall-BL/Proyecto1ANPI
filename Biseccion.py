import math
import matplotlib.pyplot as plt
import numpy as np

# ==============================
# CONSTANTES FÍSICAS
# ==============================

h = 6.626e-34       # Constante de Planck (J·s)
c = 3.00e8          # Velocidad de la luz en el vacío (m/s)
k = 1.381e-23       # Constante de Boltzmann (J/K)

# ==============================
# LEY DE PLANCK
# ==============================

def planck_law(lambda_, T):
    """
    Calcula la intensidad espectral B(λ, T) según la ley de Planck.
    """
    numerator = 2 * h * c**2 / (lambda_**5)
    exponent = (h * c) / (lambda_ * k * T)
    denominator = math.exp(exponent) - 1
    return numerator / denominator

# ==============================
# FUNCIÓN PARA EL MÉTODO DE BISECCIÓN
# ==============================

def f(lambda_, T, I0):
    """
    f(λ) = B(λ, T) - I₀
    """
    return planck_law(lambda_, T) - I0

# ==============================
# MÉTODO DE BISECCIÓN
# ==============================

def bisection(T, I0, a, b, tol=1e-9, max_iter=1000):
    """
    Método numérico para encontrar λ tal que B(λ, T) ≈ I₀.
    """
    fa = f(a, T, I0)
    fb = f(b, T, I0)

    if fa * fb > 0:
        raise ValueError("La función no cambia de signo en el intervalo dado.")

    for _ in range(max_iter):
        c = (a + b) / 2
        fc = f(c, T, I0)

        if abs(fc) < tol or (b - a) / 2 < tol:
            return c

        if fa * fc < 0:
            b, fb = c, fc
        else:
            a, fa = c, fc

    raise RuntimeError("No se encontró la raíz dentro del número máximo de iteraciones.")

# ==============================
# VISUALIZACIÓN DEL ESPECTRO
# ==============================

def graficar_planck(T, a, b):
    """
    Grafica B(λ, T) y devuelve la intensidad máxima y el pico λ.
    """
    lambdas = np.linspace(a, b, 1000)
    intensities = [planck_law(l, T) for l in lambdas]
    max_I = max(intensities)
    lambda_peak = lambdas[np.argmax(intensities)]

    # Graficar
    plt.plot([l * 1e9 for l in lambdas], intensities)
    plt.title(f"Radiación de Planck a T = {T} K")
    plt.xlabel("Longitud de onda (nm)")
    plt.ylabel("Intensidad (W·sr⁻¹·m⁻³)")
    plt.grid(True)
    plt.show()

    return max_I, lambda_peak

# ==============================
# PROGRAMA PRINCIPAL
# ==============================

if __name__ == "__main__":
    T = 5000           # Temperatura en Kelvin
    a = 100e-9         # Límite inferior (100 nm)
    b = 3000e-9        # Límite superior (3000 nm)

    # Paso 1: Graficar y obtener intensidad máxima y pico
    max_I, lambda_peak = graficar_planck(T, a, b)
    print(f"Intensidad máxima: {max_I:.2e} W·sr⁻¹·m⁻³")
    print(f"Longitud de onda pico: {lambda_peak * 1e9:.2f} nm")

    # Paso 2: Elegir intensidad objetivo
    I0 = max_I * 0.5
    print(f"Usando I0 = {I0:.2e} W·sr⁻¹·m⁻³")

    # Paso 3: Intentar encontrar soluciones en ambos lados del pico
    try:
        lambda_left = bisection(T, I0, a, lambda_peak)
        print(f"λ (antes del pico): {lambda_left * 1e9:.2f} nm")
    except Exception as e:
        print("No se encontró raíz en el lado izquierdo del pico.")

    try:
        lambda_right = bisection(T, I0, lambda_peak, b)
        print(f"λ (después del pico): {lambda_right * 1e9:.2f} nm")
    except Exception as e:
        print("No se encontró raíz en el lado derecho del pico.")
