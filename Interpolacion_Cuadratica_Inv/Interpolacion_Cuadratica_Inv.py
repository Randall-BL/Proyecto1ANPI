import numpy as np
import matplotlib.pyplot as plt
import time

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
    """
    Método: Interpolación Cuadrática Inversa (IQI)
    
    Parámetros
    ----------
    T     :     Temperatura en kelvin (K).
    I0    :     Intensidad objetivo a alcanzar.
    x0,x1,x2 :  Tres valores iniciales para longitud de onda (λ).
    tol   :     Tolerancia para criterio de parada.
    max_iter :  Número máximo de iteraciones.
    
    Retorna
    -------
    xk :    Aproximación final.
    k  :    Número de iteraciones realizadas.
    ek :    Error en la solución.
    """
    # Registrar tiempo del IQI
    import time
    inicio = time.time()

    historial = []
    def f(lambd): return planck_law(lambd, T) - I0

    print("\n>> MÉTODO: Interpolación Cuadrática Inversa (IQI)")
    print(f"   Valores iniciales: x0 = {x0:.2e}, x1 = {x1:.2e}, x2 = {x2:.2e}")

    # Bucle principal para el IQI
    for k in range(max_iter):

        # Asignación de Valores del método
        f0, f1, f2 = f(x0), f(x1), f(x2)

        # Condición de parada 1.
        if f0 == f1 or f1 == f2 or f0 == f2:
            raise ValueError("Valores f repetidos; IQI no puede continuar.")

        # Definición de la Interpolación de Lagrange inversa de segundo orden
        L0 = f1 * f2 / ((f0 - f1) * (f0 - f2))
        L1 = f0 * f2 / ((f1 - f0) * (f1 - f2))
        L2 = f0 * f1 / ((f2 - f0) * (f2 - f1))

        # Aproximación de la raíz obtenida en cada iteración
        x3 = x0 * L0 + x1 * L1 + x2 * L2
        historial.append((k, x3, f(x3)))

        # Definicion y cálculo del Error 
        ek = max(abs(f(x3)), abs(x3 - x2))

        # Condición de parada 2.
        if ek < tol:
            tiempo = time.time() - inicio
            print(f"   Iteraciones realizadas: {k + 1}")
            print(f"   Aproximación final xk: {x3:.6e}")
            print(f"   Error final ek: {ek:.2e}")
            print(f"   Tiempo de ejecución: {tiempo:.6f} segundos")
            return x3, k + 1, ek, historial

        # Actualizar los valores del método en cada iteración
        x0, x1, x2 = x1, x2, x3

    # Condición de parada 3.
    raise Exception("No se alcanzó la convergencia.")

# ==============================
# GRAFICAR PLANCK + CONVERGENCIA
# ==============================

# Funcion principal para graficar el IQI

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

    # Parámetros tomados por el IQI
    T_val = 1337  
    a = 400e-9    
    b = 5000e-9
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

    # Solución del método por la izquierda
    try:
        print("Buscando λ izquierda (IQI)...")
        left_guess = [a, (a + lambda_peak) / 2, lambda_peak * 0.95]
        lambda_left, it_left, err_left, hist_left = buscar_solucion_IQI(T_val, I0, *left_guess)
        soluciones.append(lambda_left)
        graficar_planck(T_val, a, b, I0, [lambda_left], hist_left)
    except Exception as e:
        print(f"Error IQI izquierda: {e}")
        soluciones.append(None)

    # Solución del método por la derecha
    try:
        print("\nBuscando λ derecha (IQI)...")
        right_guess = [lambda_peak * 1.05, lambda_peak * 1.3, lambda_peak * 1.7]
        lambda_right, it_right, err_right, hist_right = buscar_solucion_IQI(T_val, I0, *right_guess)
        soluciones.append(lambda_right)
        graficar_planck(T_val, a, b, I0, [lambda_right], hist_right)
    except Exception as e:
        print(f"Error IQI derecha: {e}")
        soluciones.append(None)

    print("\nMostrando gráfico final con ambas soluciones...")
    graficar_planck(T_val, a, b, I0, soluciones)