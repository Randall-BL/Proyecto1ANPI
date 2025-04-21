import numpy as np
import matplotlib.pyplot as plt
import time

# Constantes físicas
h = 6.62607015e-34      # constante de Planck [J·s]
c = 299792458           # velocidad de la luz [m/s]
k = 1.380649e-23        # constante de Boltzmann [J/K]

# Ley de Planck
# Función que implementa la Ley de Planck para una longitud de onda y temperatura dadas.
# Devuelve la intensidad espectral de radiación (W·sr⁻¹·m⁻³)
def planck(wavelength, T):
    a = 2.0*h*c**2
    b = h*c / (wavelength*k*T)
    intensity = a / (wavelength**5 * (np.exp(b) - 1.0)) # Fórmula de Planck
    return intensity

# Derivada numérica de primer orden usando diferencias centradas
def first_derivative(f, x, h=1e-10, *args):
    return (f(x + h, *args) - f(x - h, *args)) / (2 * h)

# Derivada numérica de segundo orden usando diferencias centradas
def second_derivative(f, x, h=1e-10, *args):
    return (f(x + h, *args) - 2 * f(x, *args) + f(x - h, *args)) / (h**2)

# Implementación del método de Kung-Traub para encontrar raíces de una función f
def kung_traub(f, x0, tol=1e-8, max_iter=100, *args):

    """
    Método de Kung-Traub para encontrar raíces de funciones no lineales.

    Este método combina evaluaciones sucesivas de la función y sus derivadas 
    para aproximarse rápidamente a una raíz. Utiliza una fórmula de tercer orden 
    para mejorar la convergencia.

    Parámetros:
        f         : función objetivo, de la forma f(x, *args)
        x0        : valor inicial (λ₀) para la búsqueda de la raíz
        tol       : tolerancia aceptable para el error (criterio de parada)
        max_iter  : número máximo de iteraciones permitidas
        *args     : parámetros adicionales que requiere f (por ejemplo, temperatura T, intensidad I0)

    Retorna:
        dict con:
            xk         : valor aproximado de la raíz (λ tal que f(λ) ≈ 0)
            error      : error en la última iteración (máximo entre |f(xk)| y |xk - x_{k-1}|)
            iteraciones: número de iteraciones realizadas
            tiempo     : tiempo de ejecución total del método en segundos
            lambda0    : valor inicial utilizado para iniciar el método
    """
    x_values = [x0] # Lista para almacenar aproximaciones sucesivas
    start_time = time.time() # Tiempo inicial para medir duración del método

    for k in range(1, max_iter + 1):
        # Evaluaciones de la función y sus derivadas numéricas
        f_xk = f(x_values[-1], *args)
        f1_xk = first_derivative(f, x_values[-1], 1e-10, *args)
        f2_xk = second_derivative(f, x_values[-1], 1e-10, *args)

        # Verificación de división por cero
        if f1_xk == 0:
            break
        
        # Paso 1: calcular z usando el método de Newton
        z = x_values[-1] - f_xk / f1_xk
        f_z = f(z, *args)

        # Paso 2: calcular y usando la fórmula de mejora sobre z
        y = x_values[-1] - (f_xk**2) / (f1_xk * (f_xk - 2*f_z))
        f_y = f(y, *args)
        denominator = f1_xk * (f_xk - 2*f_z)

        if denominator == 0:
            break
                
        # Paso 3: aplicar la fórmula de Kung-Traub para obtener la siguiente aproximación
        x_next = x_values[-1] - (f_xk**3) / (denominator * (f_xk - 2*f_y))
        x_values.append(x_next)

        # Cálculo del error como el máximo entre |f(xk)| y |xk - x_{k-1}|
        error = max(abs(f(x_next, *args)), abs(x_next - x_values[-2]))
                
        # Criterio de parada: si el error es menor que la tolerancia
        if error < tol:
            end_time = time.time()
            return {
                'xk': x_next,
                'error': error,
                'iteraciones': k,
                'tiempo': end_time - start_time,
                'lambda0': x0
            }
        
    # Si no se alcanzó la tolerancia en las iteraciones permitidas
    end_time = time.time()
    return {
        'xk': x_values[-1],
        'error': max(abs(f(x_values[-1], *args)), abs(x_values[-1] - x_values[-2])),
        'iteraciones': max_iter,
        'tiempo': end_time - start_time,
        'lambda0': x0
    }

# Función objetivo para encontrar longitud de onda tal que I(λ) = I0
def f(lmbda, T, I0):
    return planck(lmbda, T) - I0

# Rango de análisis
T = 1337  # temperatura en Kelvin
wavelengths = np.linspace(100e-9, 3000e-9, 1000)  # 100 a 3000 nm

print(f"\n Usando Kung-Traub para analizar radiación a T = {T} K en rango 100-3000 nm")

# Calcular intensidad de Planck
intensities = planck(wavelengths, T)

# Longitud de onda del máximo teórico (Ley de Wien)
lambda_max_theoretical = 2.898e-3 / T  # metros
print(f"Longitud de onda pico (teórica): {lambda_max_theoretical*1e9:.2f} nm")

# Longitud de onda del pico (calculada)
idx_max = np.argmax(intensities)
lambda_max = wavelengths[idx_max]
I_max = intensities[idx_max]
print(f"Intensidad máxima: {I_max:.3e} W·sr⁻¹·m⁻³")
print(f"Longitud de onda pico (calculada): {lambda_max*1e9:.2f} nm")

# Buscar longitudes de onda para una intensidad I0
I0 = 0.5 * I_max
print(f"\nBuscando λ para I0 = 50% del máximo ({I0:.3e} W·sr⁻¹·m⁻³)")

# Valores iniciales a la izquierda y derecha del pico
initial_left = lambda_max * 0.9
initial_right = lambda_max * 1.1

# Lado izquierdo del pico
result_left = kung_traub(f, initial_left, 1e-8, 100, T, I0)
I_left = planck(result_left['xk'], T)
print(f"\nλ (antes del pico): {result_left['xk']*1e9:.2f} nm → I = {I_left:.3e}")
print(f"V.I. = {result_left['lambda0']*1e9:.2f} nm")
print(f"xk = {result_left['xk']*1e9:.5f} nm")
print(f"error ek = {result_left['error']:.2e}")
print(f"iteraciones = {result_left['iteraciones']}")
print(f"tiempo de ejecución = {result_left['tiempo']:.4f} s")

# Lado derecho del pico
result_right = kung_traub(f, initial_right, 1e-8, 100, T, I0)
I_right = planck(result_right['xk'], T)
print(f"\nλ (después del pico): {result_right['xk']*1e9:.2f} nm → I = {I_right:.3e}")
print(f"V.I. = {result_right['lambda0']*1e9:.2f} nm")
print(f"xk = {result_right['xk']*1e9:.5f} nm")
print(f"error ek = {result_right['error']:.2e}")
print(f"iteraciones = {result_right['iteraciones']}")
print(f"tiempo de ejecución = {result_right['tiempo']:.4f} s")

# Gráfico de intensidad vs longitud de onda con puntos clave
plt.figure(figsize=(10, 6))
plt.plot(wavelengths * 1e9, intensities, label=f'Planck T={T}K', color='blue')

# Marcar el pico
plt.plot(lambda_max * 1e9, I_max, 'ro', label='Pico (máx)')

# Marcar los puntos donde I = 50% I_max
plt.plot(result_left['xk'] * 1e9, I_left, 'go', label='λ izquierda (50%)')
plt.plot(result_right['xk'] * 1e9, I_right, 'mo', label='λ derecha (50%)')

# Líneas de referencia
plt.axhline(I0, color='gray', linestyle='--', linewidth=1, label='50% I_max')
plt.axvline(result_left['xk'] * 1e9, color='green', linestyle='--', linewidth=1)
plt.axvline(result_right['xk'] * 1e9, color='purple', linestyle='--', linewidth=1)
plt.axvline(lambda_max * 1e9, color='red', linestyle='--', linewidth=1)

# Detalles de la gráfica
plt.title(f"Distribución espectral de la radiación a {T} K")
plt.xlabel("Longitud de onda (nm)")
plt.ylabel("Intensidad espectral (W·sr⁻¹·m⁻³)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

