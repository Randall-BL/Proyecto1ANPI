import math
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

# ==============================
# CONSTANTES FÍSICAS (Simbólicas)
# ==============================

# Definimos la variable simbólica para lambda (se la nombra λ) y para T
λ, T = sp.symbols('λ T', positive=True, real=True)
h, c, k = 6.626e-34, 3.0e8, 1.381e-23

# ==============================
# EXPRESIÓN SIMBÓLICA DE LA LEY DE PLANCK
# ==============================

# Definimos la variable intermedia x (simbólica)
x = (h * c) / (λ * k * T)
# Expresión simbólica de B(λ, T)
B_sym = (2 * h * c**2) / (λ**5 * (sp.exp(x) - 1))
# Derivada simbólica de B respecto a λ
dB_dλ_sym = sp.diff(B_sym, λ)

# Convertimos las expresiones simbólicas a funciones numéricas
# Esto se hace SOLO cuando es necesario evaluar la función
f_B = sp.lambdify((λ, T), B_sym, modules='numpy')
f_dB = sp.lambdify((λ, T), dB_dλ_sym, modules='numpy')

# ==============================
# LEY DE PLANCK (FUNCIÓN)
# ==============================

def planck_law(lambda_, T_val):
    """
    Calcula la intensidad espectral B(λ, T) usando la expresión simbólica,
    evaluada numéricamente.
    """
    if lambda_ <= 0:
        return 0.0
    # Se evalúa la función simbólica ya convertida
    return f_B(lambda_, T_val)

def df_planck(lambda_, T_val):
    """
    Evalúa la derivada simbólica de la Ley de Planck en un punto,
    usando la función creada con lambdify.
    """
    return f_dB(lambda_, T_val)

# ==============================
# MÉTODO NEWTON-RAPHSON
# ==============================

def newton_raphson(f, df, x0, tol, iter_max):
    """
    Método de Newton-Raphson para encontrar raíces de funciones no lineales.
    
    Parámetros:
        f        : función original
        df       : derivada de la función
        x0       : valor inicial
        tol      : tolerancia (criterio de parada)
        iter_max : número máximo de iteraciones

    Retorna:
        xk : valor aproximado de la raíz
        k  : número de iteraciones realizadas
        er : error final (|f(xk)|)
    """
    xk = x0
    for k in range(1, iter_max):
        if xk <= 0:
            raise ValueError("Valor de λ inválido (negativo o cero) durante el proceso.")
        dfxk = df(xk)
        if dfxk == 0:
            raise ZeroDivisionError("Derivada cero. No se puede continuar con Newton-Raphson.")
        
        xk = xk - f(xk) / dfxk
        er = abs(f(xk))
        if er < tol:
            break
    return xk, k, er

def buscar_solucion_newton(T_val, I0, x0, tol=1e-12, iter_max=100):
    """
    Busca una λ tal que B(λ, T) = I0 usando el método de Newton-Raphson.
    """
    f_nr = lambda l: planck_law(l, T_val) - I0
    df_nr = lambda l: df_planck(l, T_val)
    
    lambda_sol, k, err = newton_raphson(f_nr, df_nr, x0, tol, iter_max)
    return lambda_sol, k, err

# ==============================
# MÉTODO PARA EL GRÁFICO Y VISUALIZACIÓN
# ==============================

def graficar_planck(T_val, a, b, I0=None, soluciones=None, historial=None):
    """
    Grafica B(λ, T) con opciones para mostrar:
      - Intensidad objetivo (I0)
      - Soluciones encontradas
      - Proceso de convergencia
    """
    lambdas = np.linspace(a, b, 1000)
    intensities = np.array([planck_law(l, T_val) for l in lambdas])
    max_I = np.max(intensities)
    lambda_peak = lambdas[np.argmax(intensities)]
    
    plt.figure(figsize=(12, 6))
    # Curva de Planck
    plt.plot(lambdas * 1e9, intensities, label=f'Curva de Planck (T = {T_val} K)', linewidth=2)
    # Línea de intensidad máxima
    plt.axhline(y=max_I, color='gray', linestyle=':', alpha=0.5, label='Intensidad máxima')
    # Intensidad objetivo, si se define
    if I0 is not None:
        plt.axhline(y=I0, color='r', linestyle='--', label=f'Intensidad objetivo (I0 = {I0:.2e})')
    # Dibujar las soluciones (líneas verticales)
    if soluciones:
        colors = ['green', 'blue', 'purple']
        for i, sol in enumerate(soluciones):
            if sol is not None:
                plt.axvline(x=sol*1e9, color=colors[i], linestyle='-.', 
                            label=f'λ solución {i+1} = {sol*1e9:.1f} nm')
    # Historial de convergencia (opcional)
    if historial and len(historial) > 0:
        for i, (iter_num, lambda_, f_val) in enumerate(historial):
            if i % 5 == 0 or i == len(historial)-1:
                plt.plot(lambda_*1e9, planck_law(lambda_, T_val), 'ro', markersize=4, alpha=0.5)
                plt.text(lambda_*1e9, planck_law(lambda_, T_val), f'{iter_num}', fontsize=8)
    plt.title(f"Radiación de Planck a T = {T_val} K con Soluciones", pad=20)
    plt.xlabel("Longitud de onda (nm)")
    plt.ylabel("Intensidad espectral (W·sr⁻¹·m⁻³)")
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()
    
    return max_I, lambda_peak

def find_peak_wavelength(T_val):
    """Estimación inicial usando la ley de Wien."""
    return 2.897e-3 / T_val if T_val > 0 else 500e-9

# ==============================
# PROGRAMA PRINCIPAL
# ==============================
if __name__ == "__main__":
    # Configuración
    T_val = 5000            # Temperatura en Kelvin
    a = 100e-9              # Límite inferior (100 nm)
    b = 3000e-9             # Límite superior (3000 nm)
    I0_percent = 0.5        # Porcentaje de la intensidad máxima a buscar
    
    print("\n" + "="*50)
    print(f" ANÁLISIS DE RADIACIÓN DE PLANCK - T = {T_val} K ")
    print("="*50)
    
    # Paso 1: Graficar y obtener la intensidad máxima
    max_I, lambda_peak = graficar_planck(T_val, a, b)
    print(f"\nIntensidad máxima: {max_I:.3e} W·sr⁻¹·m⁻³")
    print(f"Longitud de onda pico (calculada): {lambda_peak*1e9:.2f} nm")
    print(f"Longitud de onda pico (teórica - Wien): {find_peak_wavelength(T_val)*1e9:.2f} nm")
    
    # Paso 2: Definir intensidad objetivo
    I0 = max_I * I0_percent
    print(f"\nBuscando λ para I0 = {I0_percent:.0%} del máximo ({I0:.3e} W·sr⁻¹·m⁻³)")
    
    # Paso 3: Buscar soluciones usando Newton-Raphson
    soluciones = []
    iteraciones = []
    
    # Lado izquierdo del pico: usamos un x0 razonable, por ejemplo 'a'
    try:
        print("\nBuscando solución antes del pico (Newton-Raphson):")
        lambda_left, k_left, err_left = buscar_solucion_newton(T_val, I0, x0=400e-9)
        soluciones.append(lambda_left)
        iteraciones.append((k_left, err_left))
        print(f"λ (antes del pico): {lambda_left*1e9:.2f} nm")
        print(f"Iteraciones: {k_left}, Error: {err_left:.2e}")
    except Exception as e:
        soluciones.append(None)
        iteraciones.append(None)
        print(f"Error en lado izquierdo: {str(e)}")
    
    # Lado derecho del pico: usamos un x0 razonable, por ejemplo 'b'
    try:
        print("\nBuscando solución después del pico (Newton-Raphson):")
        lambda_right, k_right, err_right = buscar_solucion_newton(T_val, I0, x0=800e-9)
        soluciones.append(lambda_right)
        iteraciones.append((k_right, err_right))
        print(f"λ (después del pico): {lambda_right*1e9:.2f} nm")
        print(f"Iteraciones: {k_right}, Error: {err_right:.2e}")
    except Exception as e:
        soluciones.append(None)
        iteraciones.append(None)
        print(f"Error en lado derecho: {str(e)}")
    
    # Paso 4: Graficar con las soluciones encontradas
    print("\nMostrando gráfico con soluciones...")
    graficar_planck(T_val, a, b, I0, soluciones)
    
    # Opcional: Mostrar convergencia para una de las soluciones
    if soluciones[0] is not None:
        print("\nProceso de convergencia (lado izquierdo):")
        graficar_planck(T_val, a, b, I0, [soluciones[0]])
    
    if soluciones[1] is not None:
        print("\nProceso de convergencia (lado derecho):")
        graficar_planck(T_val, a, b, I0, [soluciones[1]])
