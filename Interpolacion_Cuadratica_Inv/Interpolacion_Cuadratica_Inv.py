import numpy as np
import matplotlib.pyplot as plt

# ==============================
# CONSTANTES FÍSICAS
# ==============================

h = 6.626e-34       # Constante de Planck (J·s)
c = 3.00e8          # Velocidad de la luz en el vacío (m/s)
k = 1.381e-23       # Constante de Boltzmann (J/K)

# ==============================
# FUNCIÓN DE LA LEY DE PLANCK
# ==============================

def planck_law(lambd, T):
    """
    Calcula la intensidad espectral de la radiación emitida por un cuerpo negro 
    a una temperatura T según la ley de Planck.
    
    """
    a = 2.0 * h * c**2
    b = h * c / (lambd * k * T)
    return a / (lambd**5 * (np.exp(b) - 1.0))

# ==============================
# INTERPOLACIÓN CUADRÁTICA INVERSA (IQI)
# ==============================

def buscar_solucion_IQI(T, I0, x0, x1, x2, tol=1e-9, max_iter=100):
    historial = []

    """
    Aplica el método de Interpolación Cuadrática Inversa (IQI) para encontrar 
    una longitud de onda λ tal que la ley de Planck a temperatura T sea igual a una intensidad I0.
    Este método es útil para encontrar raíces de funciones no lineales sin derivadas,
    utilizando tres estimaciones iniciales y ajustando mediante interpolación racional inversa.

    Parámetros
    ----------
    T :          Temperatura en kelvin (K).
    I0 :         Intensidad objetivo a alcanzar mediante la ley de Planck.
    x0, x1, x2 : Tres estimaciones iniciales para la longitud de onda λ (en metros).
    tol :        Tolerancia absoluta para la convergencia. Por defecto establecido en 1e-9.
    max_iter :   Número máximo de iteraciones permitidas.

    """
    
    def f(lambd): return planck_law(lambd, T) - I0
    
    for k in range(max_iter):
        f0, f1, f2 = f(x0), f(x1), f(x2)
        
        if f0 == f1 or f1 == f2 or f0 == f2:
            raise ValueError("Valores f repetidos; IQI no puede continuar.")
        
        # Interpolación cuadrática inversa
        L0 = f1 * f2 / ((f0 - f1) * (f0 - f2))
        L1 = f0 * f2 / ((f1 - f0) * (f1 - f2))
        L2 = f0 * f1 / ((f2 - f0) * (f2 - f1))
        x3 = x0 * L0 + x1 * L1 + x2 * L2

        historial.append((k, x3, f(x3)))

        if abs(f(x3)) < tol:
            return x3, k + 1, abs(f(x3)), historial

        x0, x1, x2 = x1, x2, x3

    raise Exception("No se alcanzó la convergencia.")

# ==============================
# GRAFICAR PLANCK + CONVERGENCIA
# ==============================

def graficar_planck(T_val, a, b, I0=None, soluciones=None, historial=None):
    lambdas = np.linspace(a, b, 1000)
    intensities = np.array([planck_law(l, T_val) for l in lambdas])
    max_I = np.max(intensities)
    lambda_peak = lambdas[np.argmax(intensities)]
    
    plt.figure(figsize=(12, 6))
    plt.plot(lambdas * 1e9, intensities, label=f'Curva de Planck (T = {T_val} K)', linewidth=2)
    plt.axhline(y=max_I, color='gray', linestyle=':', alpha=0.5, label='Intensidad máxima')

    if I0 is not None:
        plt.axhline(y=I0, color='r', linestyle='--', label=f'I0 = {I0:.2e}')
    
    if soluciones:
        for i, sol in enumerate(soluciones):
            if sol is not None:
                plt.axvline(x=sol*1e9, linestyle='--', label=f'λ solución {i+1} = {sol*1e9:.1f} nm')

    if historial:
        for i, (iter_num, lamb, f_val) in enumerate(historial):
            if i % 2 == 0 or i == len(historial)-1:
                plt.plot(lamb*1e9, planck_law(lamb, T_val), 'ro', markersize=4)
                plt.text(lamb*1e9, planck_law(lamb, T_val), f'{iter_num}', fontsize=8)

    plt.title(f"Radiación de Planck a T = {T_val} K")
    plt.xlabel("Longitud de onda (nm)")
    plt.ylabel("Intensidad espectral (W·sr⁻¹·m⁻³)")
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()
    
    return max_I, lambda_peak

def find_peak_wavelength(T_val):
    return 2.897e-3 / T_val if T_val > 0 else 500e-9

# ==============================
# PROGRAMA PRINCIPAL PARA IQI
# ==============================

if __name__ == "__main__":
    T_val = 5000
    a = 100e-9
    b = 3000e-9
    I0_percent = 0.5

    print("="*50)
    print(f" ANÁLISIS CON INTERPOLACIÓN CUADRÁTICA INVERSA (IQI)")
    print("="*50)

    max_I, lambda_peak = graficar_planck(T_val, a, b)
    print(f"\nIntensidad máxima: {max_I:.3e}")
    print(f"Longitud de onda pico (calculada): {lambda_peak*1e9:.2f} nm")
    print(f"Longitud de onda pico (teórica - Wien): {find_peak_wavelength(T_val)*1e9:.2f} nm")

    I0 = max_I * I0_percent
    print(f"\nBuscando soluciones para I0 = {I0:.3e} ({I0_percent:.0%} del máximo)\n")

    soluciones = []

    # Solución izquierda (antes del pico)
    try:
        print("Buscando λ izquierda (IQI)...")
        left_guess = [a, (a + lambda_peak) / 2, lambda_peak * 0.95]
        lambda_left, it_left, err_left, hist_left = buscar_solucion_IQI(T_val, I0, *left_guess)
        print(f"λ izquierda: {lambda_left*1e9:.2f} nm - Iteraciones: {it_left}, Error: {err_left:.2e}")
        soluciones.append(lambda_left)
        graficar_planck(T_val, a, b, I0, [lambda_left], hist_left)
    except Exception as e:
        print(f"Error IQI izquierda: {e}")
        soluciones.append(None)

    # Solución derecha (después del pico)
    try:
        print("\nBuscando λ derecha (IQI)...")
        right_guess = [lambda_peak * 1.05, lambda_peak * 1.2, lambda_peak * 1.5]
        lambda_right, it_right, err_right, hist_right = buscar_solucion_IQI(T_val, I0, *right_guess)
        print(f"λ derecha: {lambda_right*1e9:.2f} nm - Iteraciones: {it_right}, Error: {err_right:.2e}")
        soluciones.append(lambda_right)
        graficar_planck(T_val, a, b, I0, [lambda_right], hist_right)
    except Exception as e:
        print(f"Error IQI derecha: {e}")
        soluciones.append(None)

    print("\nMostrando gráfico final con ambas soluciones...")
    graficar_planck(T_val, a, b, I0, soluciones)
