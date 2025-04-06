import math
import matplotlib.pyplot as plt
import numpy as np

# ==============================
# CONSTANTES FÍSICAS
# ==============================

h = 6.626e-34       # Constante de Planck (J·s)
c = 3.00e8          # Velocidad de la luz (m/s)
k = 1.381e-23       # Constante de Boltzmann (J/K)

# ==============================
# FUNCIONES DE PLANCK (MEJORADAS)
# ==============================

def planck_law(lambda_, T):
    """Ley de radiación de Planck para una longitud de onda lambda (m) y temperatura T (K)."""
    # Versión numéricamente más estable
    if lambda_ <= 0:
        return 0.0
    
    x = (h * c) / (lambda_ * k * T)
    if x > 700:  # Para evitar overflow en exp(x)
        return 0.0
    
    exp_x = math.exp(x)
    numerator = 2 * h * c**2
    denominator = (lambda_**5) * (exp_x - 1)
    return numerator / denominator

def planck_law_derivative(lambda_, T):
    """Derivada de la ley de Planck con respecto a lambda (versión estable)."""
    if lambda_ <= 0:
        return 0.0
    
    x = (h * c) / (lambda_ * k * T)
    if x > 700:
        return 0.0
    
    exp_x = math.exp(x)
    term1 = -5 / lambda_ + (x / lambda_) * (exp_x / (exp_x - 1))
    return planck_law(lambda_, T) * term1

def f(lambda_, T, I0):
    """Función objetivo para encontrar raíces."""
    return planck_law(lambda_, T) - I0

def df(lambda_, T):
    """Derivada de la función objetivo."""
    return planck_law_derivative(lambda_, T)

# ==============================
# MÉTODO DE KUNG-TRAUB (MEJORADO)
# ==============================

def kung_traub(T, I0, lambda0, tol=1e-8, max_iter=100):
    """
    Implementación mejorada del método de Kung-Traub con:
    - Mejor manejo de casos especiales
    - Verificación de convergencia dual
    - Punto de partida inteligente
    """
    x = lambda0
    best_x = x
    best_residual = abs(f(x, T, I0))
    
    for i in range(max_iter):
        try:
            fx = f(x, T, I0)
            dfx = df(x, T)
            
            # Verificar convergencia
            if abs(fx) < tol:
                return x
                
            if dfx == 0:
                # Intento recuperar con pequeño paso
                x = x * (1 + 1e-4)
                continue
            
            # Paso 1: Newton step
            y = x - fx / dfx
            fy = f(y, T, I0)
            
            # Paso 2: Segundo paso
            denom_y = fx - 2 * fy
            if denom_y == 0:
                # Si falla, usar solo paso de Newton
                x_next = y
            else:
                z = y - (fy / fx) * (fx / denom_y)
                fz = f(z, T, I0)
                
                # Paso 3: Tercer paso
                denom_z = (fx - 2 * fy)**2
                if denom_z == 0:
                    x_next = z
                else:
                    x_next = z - (fz / fx) * ((fx**2) / denom_z)
            
            # Actualizar mejor solución encontrada
            current_residual = abs(f(x_next, T, I0))
            if current_residual < best_residual:
                best_residual = current_residual
                best_x = x_next
            
            # Criterios de parada
            if abs(x_next - x) < tol or current_residual < tol:
                return x_next
                
            x = x_next
            
        except (OverflowError, ValueError):
            # Si hay problemas numéricos, ajustar el paso
            x = x * (1 + 1e-4)
    
    # Si no converge, devolver la mejor solución encontrada
    print(f"Advertencia: No se alcanzó la tolerancia en {max_iter} iteraciones. Mejor residual: {best_residual:.2e}")
    return best_x

# ==============================
# FUNCIONES AUXILIARES
# ==============================

def find_peak_wavelength(T):
    """Ley del desplazamiento de Wien para estimar el pico."""
    return 2.897e-3 / T if T > 0 else 1e-9

def graficar_planck(T, a, b):
    """Grafica la curva de Planck y devuelve el máximo."""
    lambdas = np.linspace(a, b, 1000)
    intensities = np.array([planck_law(l, T) for l in lambdas])
    max_I = np.max(intensities)
    lambda_peak = lambdas[np.argmax(intensities)]

    plt.figure(figsize=(10, 6))
    plt.plot(lambdas * 1e9, intensities, label=f'T = {T} K')
    plt.title(f"Primera grafica de Radiación de Planck a T = {T} K")
    plt.xlabel("Longitud de onda (nm)")
    plt.ylabel("Intensidad espectral (W·sr⁻¹·m⁻³)")
    plt.grid(True)
    plt.legend()
    plt.show()

    return max_I, lambda_peak

# ==============================
# PROGRAMA PRINCIPAL
# ==============================

if __name__ == "__main__":
    # Configuración
    T = 5000  # Temperatura en Kelvin
    a, b = 100e-9, 3000e-9  # Rango de longitudes de onda (100 nm a 3000 nm)
    I0_percent = 0.5  # Porcentaje de la intensidad máxima a buscar
    
    # Paso 1: Graficar y encontrar máximo
    print(f"\nAnalizando radiación a T = {T} K en rango {a*1e9:.0f}-{b*1e9:.0f} nm")
    max_I, lambda_peak = graficar_planck(T, a, b)
    print(f"Intensidad máxima: {max_I:.3e} W·sr⁻¹·m⁻³")
    print(f"Longitud de onda pico (teórica): {find_peak_wavelength(T)*1e9:.2f} nm")
    print(f"Longitud de onda pico (calculada): {lambda_peak*1e9:.2f} nm")
    
    # Paso 2: Buscar longitudes de onda para intensidad dada
    I0 = max_I * I0_percent
    print(f"\nBuscando λ para I0 = {I0_percent:.0%} del máximo ({I0:.3e} W·sr⁻¹·m⁻³)")
    
    # Puntos iniciales inteligentes (izquierda y derecha del pico)
    initial_left = lambda_peak * 0.6
    initial_right = lambda_peak * 1.5
    
    # Búsqueda de soluciones
    try:
        lambda_left = kung_traub(T, I0, initial_left)
        print(f"λ (antes del pico): {lambda_left*1e9:.2f} nm → I = {planck_law(lambda_left, T):.3e}")
    except Exception as e:
        print(f"Error en lado izquierdo: {str(e)}")
        lambda_left = None
    
    try:
        lambda_right = kung_traub(T, I0, initial_right)
        print(f"λ (después del pico): {lambda_right*1e9:.2f} nm → I = {planck_law(lambda_right, T):.3e}")
    except Exception as e:
        print(f"Error en lado derecho: {str(e)}")
        lambda_right = None
    
    # Graficar resultados
    lambdas = np.linspace(a, b, 1000)
    intensities = np.array([planck_law(l, T) for l in lambdas])
    
    plt.figure(figsize=(12, 6))
    plt.plot(lambdas * 1e9, intensities, label=f'Curva de Planck (T = {T} K)')
    plt.axhline(y=I0, color='r', linestyle='--', label=f'Intensidad objetivo ({I0_percent:.0%} del máximo)')
    
    if lambda_left:
        plt.axvline(x=lambda_left*1e9, color='g', linestyle=':', label=f'λ izquierda = {lambda_left*1e9:.1f} nm')
    if lambda_right:
        plt.axvline(x=lambda_right*1e9, color='b', linestyle=':', label=f'λ derecha = {lambda_right*1e9:.1f} nm')
    
    plt.title(f"Solución para I0 = {I0_percent:.0%} de la intensidad máxima")
    plt.xlabel("Longitud de onda (nm)")
    plt.ylabel("Intensidad espectral (W·sr⁻¹·m⁻³)")
    plt.grid(True)
    plt.legend()
    plt.show()