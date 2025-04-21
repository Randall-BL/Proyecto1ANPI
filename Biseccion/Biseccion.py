import math
import matplotlib.pyplot as plt
import numpy as np
import time

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
    if lambda_ <= 0:
        return 0.0
    x = (h * c) / (lambda_ * k * T)
    if x > 700:
        return 0.0
    exp_x = math.exp(x)
    numerator = 2 * h * c**2
    denominator = (lambda_**5) * (exp_x - 1)
    return numerator / denominator

def f(lambda_, T, I0):
    return planck_law(lambda_, T) - I0

# ==============================
# BÚSQUEDA AUTOMÁTICA DE INTERVALOS
# ==============================

def buscar_intervalo(T, I0, lam_min=1e-9, lam_max=3e-6, pasos=10000):
    lams = np.linspace(lam_min, lam_max, pasos)
    f_vals = [f(lam, T, I0) for lam in lams]
    for i in range(len(lams) - 1):
        if f_vals[i] * f_vals[i + 1] < 0:
            return lams[i], lams[i + 1]
    raise ValueError("No se encontró un intervalo con cambio de signo.")

# ==============================
# MÉTODO DE BISECCIÓN
# ==============================

def bisection(T, I0, a, b, tol=1e-8, max_iter=100):
    a_inicial, b_inicial = a, b  # Guardamos los valores originales

    fa = f(a, T, I0)
    fb = f(b, T, I0)
    if fa * fb > 0:
        raise ValueError(f"No hay cambio de signo en [{a:.2e}, {b:.2e}]")

    history = []
    start_time = time.time()

    for i in range(max_iter):
        c = (a + b) / 2
        fc = f(c, T, I0)
        history.append((i, c, fc))

        if abs(fc) < tol:
            break

        if fa * fc < 0:
            b, fb = c, fc
        else:
            a, fa = c, fc

    end_time = time.time()
    tiempo = end_time - start_time

    xk = c
    xk_1 = history[-2][1] if len(history) > 1 else xk
    ek = max(abs(fc), abs(xk - xk_1))
    iteraciones = len(history)

    resultados = {
        "xk": xk,
        "error": ek,
        "iteraciones": iteraciones,
        "tiempo": tiempo,
        "a": a,
        "b": b,
        "a_inicial": a_inicial,
        "b_inicial": b_inicial,
        "intensidad_en_xk": planck_law(xk, T)
    }

    return xk, history, resultados

# ==============================
# GRAFICAR CURVA DE PLANCK
# ==============================

def graficar_planck(T, a, b, I0=None, soluciones=None, historial=None):
    lambdas = np.linspace(a, b, 1000)
    intensities = np.array([planck_law(l, T) for l in lambdas])
    max_I = np.max(intensities)
    lambda_peak = lambdas[np.argmax(intensities)]

    plt.figure(figsize=(12, 6))
    plt.plot(lambdas * 1e9, intensities, label=f'Curva de Planck (T = {T} K)', linewidth=2)
    plt.axhline(y=max_I, color='gray', linestyle=':', alpha=0.5, label='Intensidad máxima')

    if I0 is not None:
        plt.axhline(y=I0, color='r', linestyle='--', label=f'I₀ = {I0:.2e}')

    if soluciones:
        for i, sol in enumerate(soluciones):
            if sol is not None:
                plt.axvline(x=sol*1e9, linestyle='--', label=f'λ solución {i+1} = {sol*1e9:.1f} nm')

    if historial:
        for i, (iter_num, lambda_, f_val) in enumerate(historial):
            if i % 5 == 0 or i == len(historial) - 1:
                plt.plot(lambda_ * 1e9, planck_law(lambda_, T), 'ro', markersize=4, alpha=0.5)
                plt.text(lambda_ * 1e9, planck_law(lambda_, T), f'{iter_num}', fontsize=8)

    plt.xlabel("Longitud de onda (nm)")
    plt.ylabel("Intensidad espectral (W·sr⁻¹·m⁻³)")
    plt.title(f"Distribución espectral para T = {T} K")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return max_I, lambda_peak

# ==============================
# FUNCIÓN PRINCIPAL
# ==============================

if __name__ == "__main__":
    T = 1337
    I0_percent = 0.5

    print("\n" + "=" * 50)
    print(f" ANÁLISIS DE RADIACIÓN DE PLANCK usando Biseccion - T = {T} K ")
    print("=" * 50)

    a_vis = 100e-9
    b_vis = 3000e-9

    max_I, lambda_peak = graficar_planck(T, a_vis, b_vis)
    print(f"\nIntensidad máxima: {max_I:.3e} W·sr⁻¹·m⁻³")
    print(f"Longitud de onda pico (calculada): {lambda_peak*1e9:.2f} nm")

    I0 = max_I * I0_percent
    print(f"\nBuscando longitud de onda para I₀ = {I0_percent:.0%} del máximo ({I0:.2e})")

    soluciones = []
    historiales = []
    resultados_finales = []

    for lado, desc in [("izquierdo", (1e-9, lambda_peak)), ("derecho", (lambda_peak, 3e-6))]:
        try:
            print(f"\nBuscando en el lado {lado}...")
            a, b = buscar_intervalo(T, I0, *desc)
            print(f"Intervalo inicial: a = {a*1e9:.2f} nm, b = {b*1e9:.2f} nm")

            raiz, historial, resultado = bisection(T, I0, a, b)
            soluciones.append(raiz)
            historiales.append(historial)
            resultados_finales.append(resultado)

            print(f"λ encontrado: {raiz*1e9:.2f} nm")
            print(f"Iteraciones: {resultado['iteraciones']}")
            print(f"Error final: {resultado['error']:.2e}")
            print(f"Tiempo de ejecución: {resultado['tiempo']:.6f} s")

        except Exception as e:
            soluciones.append(None)
            historiales.append(None)
            resultados_finales.append(None)
            print(f"Error: {str(e)}")

    print("\nGraficando resultados encontrados...")
    graficar_planck(T, a_vis, b_vis, I0, soluciones)

    for i, res in enumerate(resultados_finales):
        if res:
            print(f"\n Resultados método de bisección (solución {i+1}):")
            print(f"Valores iniciales: a = {res['a_inicial']*1e9:.2f} nm, b = {res['b_inicial']*1e9:.2f} nm")
            print(f"Aproximación xk: {res['xk']*1e9:.6f} nm")
            print(f"Error ek: {res['error']:.2e}")
            print(f"Iteraciones: {res['iteraciones']}")
            print(f"Tiempo de ejecución: {res['tiempo']:.6f} s")
            print(f"Intensidad en xk: {res['intensidad_en_xk']:.2e} vs I₀ = {I0:.2e}")
