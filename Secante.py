import numpy as np
import matplotlib.pyplot as plt
import time

# Constantes físicas universales utilizadas en la Ley de Planck
h = 6.626e-34       # Constante de Planck (Julios por segundo)
c = 3.0e8           # Velocidad de la luz en el vacío (metros por segundo)
k = 1.38e-23        # Constante de Boltzmann (Julios por Kelvin)

# -----------------------------------------------
# Ley de Planck: calcula la intensidad espectral de la radiación emitida
# por un cuerpo negro a una temperatura T para una longitud de onda dada.
# -----------------------------------------------
def planck_law(wavelength_m, T):
    a = 2.0 * h * c**2
    b = h * c / (wavelength_m * k * T)
    intensity = a / ((wavelength_m**5) * (np.exp(b) - 1.0))
    return intensity

# -----------------------------------------------
# Función que se anula cuando la intensidad B(λ,T) es igual a I0.
# Es la función f(λ) = B(λ,T) - I0, usada para encontrar raíces.
# -----------------------------------------------
def f(wavelength_m, T, I0):
    return planck_law(wavelength_m, T) - I0

# -----------------------------------------------
# Método de la secante: encuentra una raíz de la función f
# a partir de dos valores iniciales x0 y x1.
# También imprime información relevante del proceso.
# -----------------------------------------------
def metodo_secante(f, x0, x1, T, I0, tol=1e-9, max_iter=100):
    print("\n--- Método de la secante ---")
    print(f"Valores iniciales: x0 = {x0:.2e} m, x1 = {x1:.2e} m")

    start_time = time.time()
    historial = []        # Lista para guardar las aproximaciones (xk, f(xk))
    iteracion = 0         # Contador de iteraciones
    x_prev = x0           # Para cálculo de error

    for _ in range(max_iter):
        f0 = f(x0, T, I0)
        f1 = f(x1, T, I0)

        # Evitar división por cero si f0 y f1 son muy similares
        if abs(f1 - f0) < 1e-12:
            print("División por cero evitable. Terminando método.")
            return None, historial

        # Fórmula de la secante para calcular siguiente aproximación
        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        historial.append((x2, planck_law(x2, T)))  # Guardar el punto actual

        # Calcular error como el máximo entre |f(xk)| y |xk - xk-1|
        error = max(abs(f(x2, T, I0)), abs(x2 - x1))
        iteracion += 1

        # Verificar si se alcanzó la tolerancia deseada
        if abs(x2 - x1) < tol:
            end_time = time.time()
            print(f"Iteraciones: k = {iteracion}")
            print(f"Aproximación xk = {x2:.5e} m ({x2 * 1e9:.2f} nm)")
            print(f"Error ek = {error:.2e}")
            print(f"Tiempo de ejecución: {end_time - start_time:.5f} segundos")
            return x2, historial

        # Actualizar valores para siguiente iteración
        x0, x1 = x1, x2

    # Si se alcanza el número máximo de iteraciones sin converger
    end_time = time.time()
    print("No se encontró la raíz dentro del número máximo de iteraciones.")
    return None, historial

# -----------------------------------------------
# Función que genera una gráfica de la ley de Planck y
# muestra las raíces encontradas con el método de la secante.
# También incluye las iteraciones realizadas.
# -----------------------------------------------
def graficar_planck_secante(T, a, b, I0, soluciones, historiales):
    lambdas = np.linspace(a, b, 1000)  # Rango de longitudes de onda
    intensities = planck_law(lambdas, T)  # Intensidad espectral
    max_I = np.max(intensities)
    lambda_peak = lambdas[np.argmax(intensities)]

    # Crear figura
    plt.figure(figsize=(12, 6))
    plt.plot(lambdas * 1e9, intensities, label=f'Curva de Planck (T = {T} K)', linewidth=2)
    plt.axhline(y=I0, color='r', linestyle='--', label=f'I0 = {I0:.2e}')

    # Marcar las raíces encontradas
    for i, sol in enumerate(soluciones):
        if sol is not None:
            plt.axvline(x=sol * 1e9, linestyle='--', label=f'λ solución {i+1} = {sol * 1e9:.1f} nm', alpha=0.7)

    # Marcar las iteraciones realizadas durante el método
    for i, hist in enumerate(historiales):
        for j, (x, y) in enumerate(hist):
            if j % 2 == 0 or j == len(hist)-1:
                plt.plot(x * 1e9, y, 'ro', markersize=4)
                plt.text(x * 1e9, y, f'{j}', fontsize=8)

    plt.title(f"Radiación de Planck con método de la secante (T = {T} K)")
    plt.xlabel("Longitud de onda (nm)")
    plt.ylabel("Intensidad espectral (W·sr⁻¹·m⁻³)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

# ------------------- MAIN -------------------

# Temperatura del cuerpo negro
T = 1337  # Kelvin

# Intervalo de longitudes de onda de interés (en metros)
a = 400e-9   # 400 nm
b = 5000e-9  # 5000 nm

# Obtener la intensidad máxima y definir I0 como el 50%
lambdas = np.linspace(a, b, 1000)
intensidades = planck_law(lambdas, T)
I_max = np.max(intensidades)
I0 = 0.5 * I_max

# Valores iniciales para buscar raíces a la izquierda y derecha del pico
left_guess = [a, (a + lambdas[np.argmax(intensidades)]) / 2]
right_guess = [lambdas[np.argmax(intensidades)] * 1.05, lambdas[np.argmax(intensidades)] * 1.7]

# Ejecutar el método de la secante para ambos lados
start_time = time.time()
raiz_izq, hist_izq = metodo_secante(f, *left_guess, T, I0)
raiz_der, hist_der = metodo_secante(f, *right_guess, T, I0)
tiempo_total = time.time() - start_time

# Imprimir resultados finales
print(f"\nRaíz izquierda encontrada: {raiz_izq * 1e9:.2f} nm")
print(f"Raíz derecha encontrada: {raiz_der * 1e9:.2f} nm")
print(f"Tiempo de ejecución total: {tiempo_total:.5f} segundos")

# Graficar resultados
graficar_planck_secante(T, a, b, I0, [raiz_izq, raiz_der], [hist_izq, hist_der])





