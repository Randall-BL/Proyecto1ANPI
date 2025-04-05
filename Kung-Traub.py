import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import h, c, k

def planck_law(lambda_, T):
    """Versión numéricamente estable de la ley de Planck"""
    lambda_m = lambda_  # Asegurarnos que está en metros
    term = h * c / (lambda_m * k * T)
    
    # Manejo numérico para valores extremos
    if term > 700:  # Para evitar overflow en exp
        return 0.0
    exp_term = np.exp(term)
    
    numerator = 2 * h * c**2 / (lambda_m**5)
    denominator = exp_term - 1
    
    # Evitar división por cero
    if denominator == 0:
        return 0.0
        
    return numerator / denominator

def kung_traub(T, I0, lambda_guess, tol=1e-6, max_iter=100):
    """Método de Kung-Traub con estabilidad numérica mejorada"""
    x = lambda_guess
    last_valid = x
    
    for _ in range(max_iter):
        try:
            fx = planck_law(x, T) - I0
            if abs(fx) < tol:
                return x
                
            # Derivada numérica para mayor estabilidad
            dx = max(1e-10 * x, 1e-12)
            dfx = (planck_law(x + dx, T) - planck_law(x - dx, T)) / (2 * dx)
            
            if dfx == 0:
                dfx = 1e-15 if dfx >= 0 else -1e-15
                
            y = x - fx / dfx
            fy = planck_law(y, T) - I0
            
            denom_y = fx - 2 * fy
            if abs(denom_y) < 1e-15:
                denom_y = 1e-15 if denom_y >= 0 else -1e-15
                
            z = y - (fy / fx) * (fx / denom_y)
            fz = planck_law(z, T) - I0
            
            denom_z = (fx - 2 * fy)**2
            if abs(denom_z) < 1e-15:
                denom_z = 1e-15
                
            x_next = z - (fz / fx) * (fx**2 / denom_z)
            
            # Asegurar que x_next sea físicamente razonable
            if x_next <= 0 or x_next > 10 * lambda_guess:
                x_next = x * 0.9
                
            if abs(x_next - x) < tol:
                return x_next
                
            x = x_next
            last_valid = x
            
        except Exception as e:
            x = last_valid * 0.9 if last_valid > 0 else 1e-9
            
    return last_valid

def graficar_planck(T, a, b):
    """Grafica la curva de Planck y encuentra el máximo"""
    lambdas = np.linspace(a, b, 1000)
    intensities = np.array([planck_law(l, T) for l in lambdas])
    max_I = np.max(intensities)
    lambda_peak = lambdas[np.argmax(intensities)]
    
    plt.figure(figsize=(10, 6))
    plt.plot(lambdas * 1e9, intensities)
    plt.title(f"Radiación de Planck a T = {T} K")
    plt.xlabel("Longitud de onda (nm)")
    plt.ylabel("Intensidad (W·sr⁻¹·m⁻³)")
    plt.grid(True)
    plt.show()
    
    return max_I, lambda_peak

if __name__ == "__main__":
    T = 5000  # Temperatura en Kelvin
    a = 100e-9  # 100 nm
    b = 3000e-9  # 3000 nm
    
    # Paso 1: Graficar y encontrar el pico
    max_I, lambda_peak = graficar_planck(T, a, b)
    print(f"Intensidad máxima: {max_I:.2e} W·sr⁻¹·m⁻³")
    print(f"Longitud de onda pico: {lambda_peak * 1e9:.2f} nm")
    
    # Paso 2: Definir intensidad objetivo (50% del máximo)
    I0 = max_I * 0.5
    print(f"\nBuscando soluciones para I0 = {I0:.2e} W·sr⁻¹·m⁻³")
    
    # Paso 3: Encontrar soluciones con estimaciones iniciales adecuadas
    print("\nCalculando solución del lado izquierdo...")
    lambda_left = kung_traub(T, I0, lambda_peak * 0.5)  # Estimación más conservadora
    
    print("Calculando solución del lado derecho...")
    lambda_right = kung_traub(T, I0, lambda_peak * 1.5)  # Estimación más conservadora
    
    # Paso 4: Mostrar resultados
    print(f"\nλ (lado izquierdo): {lambda_left * 1e9:.2f} nm")
    print(f"Intensidad calculada: {planck_law(lambda_left, T):.2e} (objetivo: {I0:.2e})")
    
    print(f"\nλ (lado derecho): {lambda_right * 1e9:.2f} nm")
    print(f"Intensidad calculada: {planck_law(lambda_right, T):.2e} (objetivo: {I0:.2e})")
    
    # Paso 5: Graficar resultados finales
    lambdas = np.linspace(a, b, 1000)
    intensities = np.array([planck_law(l, T) for l in lambdas])
    
    plt.figure(figsize=(12, 7))
    plt.plot(lambdas * 1e9, intensities, 'b-', label=f'Ley de Planck (T = {T} K)')
    plt.axhline(I0, color='gray', linestyle='--', label=f'I0 = {I0:.2e}')
    
    # Solo graficar puntos si son válidos
    if a <= lambda_left <= b:
        plt.plot(lambda_left * 1e9, planck_law(lambda_left, T), 'ro', markersize=8, 
                label=f'Solución izquierda: {lambda_left*1e9:.1f} nm')
    
    if a <= lambda_right <= b:
        plt.plot(lambda_right * 1e9, planck_law(lambda_right, T), 'go', markersize=8,
                label=f'Solución derecha: {lambda_right*1e9:.1f} nm')
    
    plt.xlabel('Longitud de onda (nm)')
    plt.ylabel('Intensidad (W·sr⁻¹·m⁻³)')
    plt.title('Solución de la Ley de Planck usando Kung-Traub')
    plt.grid(True)
    plt.legend()
    plt.show()