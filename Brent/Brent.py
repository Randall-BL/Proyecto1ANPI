import math
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
import time

# ==============================
# CONSTANTES FÍSICAS (Simbólicas)
# ==============================
λ, T = sp.symbols('λ T', positive=True, real=True)
h, c, k = 6.626e-34, 3.0e8, 1.381e-23

x = (h * c) / (λ * k * T)
B_sym = (2 * h * c**2) / (λ**5 * (sp.exp(x) - 1))
dB_dλ_sym = sp.diff(B_sym, λ)

f_B = sp.lambdify((λ, T), B_sym, modules='numpy')
f_dB = sp.lambdify((λ, T), dB_dλ_sym, modules='numpy')

def planck_law(lambda_, T_val):
    """
    Calcula la intensidad espectral B(λ, T) usando la expresión simbólica,
    evaluada numéricamente.
    """
    if lambda_ <= 0:
        return 0.0
    return f_B(lambda_, T_val)

# ==============================
# MÉTODO BRENT
# ==============================
def metodo_brent(f, a, b, I0, tol=1e-12, max_iter=100):
    """
    Método de Brent para encontrar raíces de funciones no lineales.

    Parámetros:
        f        : función a evaluar (que sea continua en [a, b])
        a        : límite inferior del intervalo
        b        : límite superior del intervalo
        I0       : valor objetivo de f (se busca f(x) = I0)
        tol      : tolerancia aceptable para el error en f(λ)
        max_iter : número máximo de iteraciones permitidas

    Retorna:
        b        : valor aproximado de la raíz 
        iteration: número de iteraciones realizadas
        error_rel: error relativo final |f(b) - I0| / |I0|
    """
    # Comienzo con bisección
    fa, fb = f(a), f(b)
    if fa * fb >= 0:
        raise ValueError("El intervalo no cumple la condición de signo opuesto.")

    c = a
    fc = fa
    d = e = b - a

    for iteration in range(1, max_iter):
        # Si no hay cambio de signo entre b y c, se reinicia el bracket (parte de bisección)
        if fb * fc > 0:
            c = a
            fc = fa
            d = e = b - a

        # Se reorganizan puntos para que |fb| < |fc|
        if abs(fc) < abs(fb):
            a, b = b, c
            fa, fb = fb, fc
            c = a
            fc = fa

        xm = (c - b) / 2  # Punto medio se usa si se necesita bisección

        # Criterio de parada
        if abs(xm) <= tol or abs(fb) < tol:
            error_rel = abs(fb) / abs(I0) if I0 != 0 else abs(fb)
            return b, iteration, error_rel

        # Si hay condiciones, se intenta interpolación cuadrática inversa
        if abs(e) > tol and abs(fa) > abs(fb):
            s = fb / fa
            if a == c:
                # Secante: si solo hay dos puntos válidos (a == c)
                p = 2 * xm * s
                q = 1 - s
            else:
                # Interpolación cuadrática inversa
                q = fa / fc
                r = fb / fc
                p = s * (2 * xm * q * (q - r) - (b - a) * (r - 1))
                q = (q - 1) * (r - 1) * (s - 1)

            if p > 0:
                q = -q
            p = abs(p)

            min1 = 3 * xm * q
            min2 = abs(e * q)

            # Decisión inteligente: usar interpolación o no?
            if 2 * p < min(min1, min2):
                # Se acepta el paso por interpolación (secante o cuadrática)
                e = d
                d = p / q
            else:
                # Fallback a bisección si el salto no es confiable
                d = xm
                e = d
        else:
            # No se cumplen condiciones para interpolar, por lo que se hace bisección
            d = xm
            e = d

        # Actualización de los puntos
        a = b
        fa = fb
        b = b + d if abs(d) > tol else b + (tol if xm >= 0 else -tol)
        fb = f(b)
    raise RuntimeError("El método de Brent no convergió.")

def buscar_solucion_brent(T_val, I0, a, b):
    """
    Busca una λ tal que B(λ, T) - I0 = 0 usando el método de Brent.
    """
    f_objetivo = lambda l: planck_law(l, T_val) - I0
    return metodo_brent(f_objetivo, a, b, I0)

def graficar_planck(T_val, a, b, I0=None, soluciones=None):
    """
    Grafica B(λ, T) muestra:
      - Intensidad objetivo (I0)
      - Soluciones encontradas
      - Proceso de convergencia
    """
    lambdas = np.linspace(a, b, 1000)
    intensities = np.array([planck_law(l, T_val) for l in lambdas])
    max_I = np.max(intensities)
    lambda_peak = lambdas[np.argmax(intensities)]

    plt.figure(figsize=(12, 6))
    plt.plot(lambdas * 1e9, intensities, label=f'Curva de Planck (T = {T_val} K)', linewidth=2)
    plt.axhline(y=max_I, color='gray', linestyle=':', alpha=0.5, label='Intensidad máxima')
    if I0 is not None:
        plt.axhline(y=I0, color='r', linestyle='--', label=f'I0 = {I0:.2e} W·sr⁻¹·m⁻³')
    if soluciones:
        colors = ['green', 'blue']
        for i, sol in enumerate(soluciones):
            if sol is not None:
                plt.axvline(x=sol * 1e9, color=colors[i], linestyle='-.', label=f'λ solución {i+1} = {sol*1e9:.1f} nm')
    plt.title(f"Radiación de Planck a T = {T_val} K")
    plt.xlabel("Longitud de onda (nm)")
    plt.ylabel("Intensidad espectral (W·sr⁻¹·m⁻³)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
    return max_I, lambda_peak

def find_peak_wavelength(T_val):
    """
    Estimación inicial usando la ley de Wien.
    """
    return 2.897e-3 / T_val if T_val > 0 else 500e-9

# ==============================
# PROGRAMA PRINCIPAL
# ==============================
if __name__ == "__main__":
    T_val = 1337.58 
    a = 900e-9
    b = 10000e-9
    I0_percent = 0.5

    print("\n" + "="*50)
    print(f" ANÁLISIS DE RADIACIÓN DE PLANCK - T = {T_val} K ")
    print("="*50)

    max_I, lambda_peak = graficar_planck(T_val, a, b)
    print(f"\nIntensidad máxima: {max_I:.3e} W·sr⁻¹·m⁻³")
    print(f"Longitud de onda pico (calculada): {lambda_peak*1e9:.2f} nm")
    print(f"Longitud de onda pico (teórica - Wien): {find_peak_wavelength(T_val)*1e9:.2f} nm")

    I0 = max_I * I0_percent
    print(f"\nBuscando λ para I0 = {I0_percent:.0%} del máximo ({I0:.3e} W·sr⁻¹·m⁻³)")

    soluciones = []

    # Lado izquierdo
    try:
        print("\nBuscando solución antes del pico (Método de Brent):")
        x0_a = a
        x0_b = lambda_peak
        start = time.time()
        lambda_left, k_left, error_left = buscar_solucion_brent(T_val, I0, x0_a, x0_b)
        end = time.time()
        soluciones.append(lambda_left)
        print("  Método: Brent")
        print(f"  Intervalo inicial: a = {x0_a*1e9:.2f} nm, b = {x0_b*1e9:.2f} nm")
        print(f"  λ encontrado: {lambda_left*1e9:.2f} nm")
        print(f"  Iteraciones: {k_left}")
        print(f"  Error absoluto: {error_left:.2e}")
        print(f"  Tiempo de ejecución: {end - start:.2e} segundos")
    except Exception as e:
        soluciones.append(None)
        print(f"Error en lado izquierdo: {str(e)}")

    # Lado derecho
    try:
        print("\nBuscando solución después del pico (Método de Brent):")
        x0_a = lambda_peak
        x0_b = b
        start = time.time()
        lambda_right, k_right, error_right = buscar_solucion_brent(T_val, I0, x0_a, x0_b)
        end = time.time()
        soluciones.append(lambda_right)
        print("  Método: Brent")
        print(f"  Intervalo inicial: a = {x0_a*1e9:.2f} nm, b = {x0_b*1e9:.2f} nm")
        print(f"  λ encontrado: {lambda_right*1e9:.2f} nm")
        print(f"  Iteraciones: {k_right}")
        print(f"  Error absoluto: {error_right:.2e}")
        print(f"  Tiempo de ejecución: {end - start:.2e} segundos")
    except Exception as e:
        soluciones.append(None)
        print(f"Error en lado derecho: {str(e)}")

    # Mostrar gráfico con soluciones
    print("\nMostrando gráfico con soluciones...")
    graficar_planck(T_val, a, b, I0, soluciones)
    
    # Mostrar convergencia para una de las soluciones
    if soluciones[0] is not None:
        print("\nProceso de convergencia (lado izquierdo):")
        graficar_planck(T_val, a, b, I0, [soluciones[0]])
    
    if soluciones[1] is not None:
        print("\nProceso de convergencia (lado derecho):")
        graficar_planck(T_val, a, b, I0, [soluciones[1]])