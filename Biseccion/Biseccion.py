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
# LEY DE PLANCK (MEJORADA)
# ==============================

def planck_law(lambda_, T):
    """Calcula la intensidad espectral B(λ, T) según la ley de Planck."""
    if lambda_ <= 0:
        return 0.0
    
    x = (h * c) / (lambda_ * k * T)
    if x > 700:  # Evitar overflow
        return 0.0
    
    exp_x = math.exp(x)
    numerator = 2 * h * c**2
    denominator = (lambda_**5) * (exp_x - 1)
    return numerator / denominator

# ==============================
# FUNCIÓN PARA EL MÉTODO DE BISECCIÓN
# ==============================

def f(lambda_, T, I0):
    """f(λ) = B(λ, T) - I₀"""
    return planck_law(lambda_, T) - I0

# ==============================
# MÉTODO DE BISECCIÓN MEJORADO
# ==============================

def bisection(T, I0, a, b, tol=1e-8, max_iter=100):
    """
    Método de bisección mejorado con:
    - Manejo de errores mejorado
    - Verificación de intervalo
    - Mensajes de progreso
    """
    fa = f(a, T, I0)
    fb = f(b, T, I0)

    if fa * fb > 0:
        raise ValueError(f"No hay cambio de signo en [{a*1e9:.1f} nm, {b*1e9:.1f} nm]. f(a)={fa:.2e}, f(b)={fb:.2e}")

    history = []  # Para guardar el historial de aproximaciones
    
    for i in range(max_iter):
        c = (a + b) / 2
        fc = f(c, T, I0)
        history.append((i, c, fc))
        
        if abs(fc) < tol:
            print(f"Convergencia alcanzada en {i} iteraciones")
            return c, history

        if fa * fc < 0:
            b, fb = c, fc
        else:
            a, fa = c, fc

    print(f"Advertencia: Máximo de iteraciones ({max_iter}) alcanzado")
    return c, history

# ==============================
# VISUALIZACIÓN MEJORADA
# ==============================

def graficar_planck(T, a, b, I0=None, soluciones=None, historial=None):
    """
    Grafica B(λ, T) con opciones para mostrar:
    - Intensidad objetivo (I0)
    - Soluciones encontradas
    - Proceso de convergencia
    """
    lambdas = np.linspace(a, b, 1000)
    intensities = np.array([planck_law(l, T) for l in lambdas])
    max_I = np.max(intensities)
    lambda_peak = lambdas[np.argmax(intensities)]

    plt.figure(figsize=(12, 6))
    
    # Curva de Planck
    plt.plot(lambdas * 1e9, intensities, label=f'Curva de Planck (T = {T} K)', linewidth=2)
    
    # Línea de intensidad máxima
    plt.axhline(y=max_I, color='gray', linestyle=':', alpha=0.5, label='Intensidad máxima')
    
    # Intensidad objetivo
    if I0 is not None:
        plt.axhline(y=I0, color='r', linestyle='--', label=f'Intensidad objetivo (I0 = {I0:.2e})')
    
    # Soluciones encontradas
    if soluciones:
        colors = ['green', 'blue', 'purple']
        for i, sol in enumerate(soluciones):
            if sol is not None:
                plt.axvline(x=sol*1e9, color=colors[i], linestyle='-.', 
                           label=f'λ solución {i+1} = {sol*1e9:.1f} nm')
    
    # Historial de convergencia (opcional)
    if historial and len(historial) > 0:
        for i, (iter_num, lambda_, f_val) in enumerate(historial):
            if i % 5 == 0 or i == len(historial)-1:  # Mostrar cada 5 iteraciones y la última
                plt.plot(lambda_*1e9, planck_law(lambda_, T), 'ro', markersize=4, alpha=0.5)
                plt.text(lambda_*1e9, planck_law(lambda_, T), f'{iter_num}', fontsize=8)
    
    plt.title(f"Radiación de Planck a T = {T} K con Soluciones", pad=20)
    plt.xlabel("Longitud de onda (nm)")
    plt.ylabel("Intensidad espectral (W·sr⁻¹·m⁻³)")
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

    return max_I, lambda_peak

# ==============================
# FUNCIÓN PARA ENCONTRAR PICO
# ==============================

def find_peak_wavelength(T):
    """Estimación inicial usando la ley de Wien."""
    return 2.897e-3 / T if T > 0 else 500e-9

# ==============================
# PROGRAMA PRINCIPAL MEJORADO
# ==============================

if __name__ == "__main__":
    # Configuración
    T = 5000           # Temperatura en Kelvin
    a = 100e-9         # Límite inferior (100 nm)
    b = 3000e-9        # Límite superior (3000 nm)
    I0_percent = 0.5   # Porcentaje de la intensidad máxima a buscar
    
    print("\n" + "="*50)
    print(f" ANÁLISIS DE RADIACIÓN DE PLANCK - T = {T} K ")
    print("="*50)
    
    # Paso 1: Graficar y obtener intensidad máxima
    max_I, lambda_peak = graficar_planck(T, a, b)
    print(f"\nIntensidad máxima: {max_I:.3e} W·sr⁻¹·m⁻³")
    print(f"Longitud de onda pico (calculada): {lambda_peak*1e9:.2f} nm")
    print(f"Longitud de onda pico (teórica - Wien): {find_peak_wavelength(T)*1e9:.2f} nm")
    
    # Paso 2: Definir intensidad objetivo
    I0 = max_I * I0_percent
    print(f"\nBuscando λ para I0 = {I0_percent:.0%} del máximo ({I0:.3e} W·sr⁻¹·m⁻³)")
    
    # Paso 3: Buscar soluciones en ambos lados del pico
    soluciones = []
    historiales = []
    
    # Lado izquierdo del pico
    try:
        print("\nBuscando solución antes del pico:")
        lambda_left, hist_left = bisection(T, I0, a, lambda_peak)
        soluciones.append(lambda_left)
        historiales.append(hist_left)
        print(f"λ (antes del pico): {lambda_left*1e9:.2f} nm")
        print(f"Intensidad en este punto: {planck_law(lambda_left, T):.3e} (error: {abs(planck_law(lambda_left, T)-I0):.2e})")
    except Exception as e:
        soluciones.append(None)
        historiales.append(None)
        print(f"Error en lado izquierdo: {str(e)}")
    
    # Lado derecho del pico
    try:
        print("\nBuscando solución después del pico:")
        lambda_right, hist_right = bisection(T, I0, lambda_peak, b)
        soluciones.append(lambda_right)
        historiales.append(hist_right)
        print(f"λ (después del pico): {lambda_right*1e9:.2f} nm")
        print(f"Intensidad en este punto: {planck_law(lambda_right, T):.3e} (error: {abs(planck_law(lambda_right, T)-I0):.2e})")
    except Exception as e:
        soluciones.append(None)
        historiales.append(None)
        print(f"Error en lado derecho: {str(e)}")
    
    # Paso 4: Graficar con las soluciones encontradas
    print("\nMostrando gráfico con soluciones...")
    graficar_planck(T, a, b, I0, soluciones)
    
    # Opcional: Mostrar convergencia para una de las soluciones
    if soluciones[0] is not None:
        print("\nProceso de convergencia (lado izquierdo):")
        graficar_planck(T, a, b, I0, [soluciones[0]], historiales[0])
    
    if soluciones[1] is not None:
        print("\nProceso de convergencia (lado derecho):")
        graficar_planck(T, a, b, I0, [soluciones[1]], historiales[1])