import numpy as np
import matplotlib.pyplot as plt
import time

# Constantes físicas
h = 6.62607015e-34      # constante de Planck [J·s]
c = 299792458           # velocidad de la luz [m/s]
k = 1.380649e-23        # constante de Boltzmann [J/K]

# Ley de Planck
def planck(wavelength, T):
    a = 2.0*h*c**2
    b = h*c / (wavelength*k*T)
    intensity = a / (wavelength**5 * (np.exp(b) - 1.0))
    return intensity

# Derivada numérica de orden 1 y 2
def first_derivative(f, x, h=1e-10, *args):
    return (f(x + h, *args) - f(x - h, *args)) / (2 * h)

def second_derivative(f, x, h=1e-10, *args):
    return (f(x + h, *args) - 2 * f(x, *args) + f(x - h, *args)) / (h**2)

# Método de Kung-Traub
def kung_traub(f, x0, tol=1e-8, max_iter=100, *args):
    x_values = [x0]
    start_time = time.time()

    for k in range(1, max_iter + 1):
        f_xk = f(x_values[-1], *args)
        f1_xk = first_derivative(f, x_values[-1], 1e-10, *args)
        f2_xk = second_derivative(f, x_values[-1], 1e-10, *args)

        if f1_xk == 0:
            break

        z = x_values[-1] - f_xk / f1_xk
        f_z = f(z, *args)
        y = x_values[-1] - (f_xk**2) / (f1_xk * (f_xk - 2*f_z))
        f_y = f(y, *args)
        denominator = f1_xk * (f_xk - 2*f_z)

        if denominator == 0:
            break

        x_next = x_values[-1] - (f_xk**3) / (denominator * (f_xk - 2*f_y))
        x_values.append(x_next)

        error = max(abs(f(x_next, *args)), abs(x_next - x_values[-2]))
        if error < tol:
            end_time = time.time()
            return {
                'xk': x_next,
                'error': error,
                'iteraciones': k,
                'tiempo': end_time - start_time,
                'lambda0': x0
            }

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

