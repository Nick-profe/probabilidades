import numpy as np

# TALLER DE PROBABILIDAD APLICADA A MACHINE LEARNING
# ============================================================

# Nombre: Wilson
# Apellido 1: Navia
# Apellido 2: Valencia
# Rama: navia_valencia


# 1. CARGA DE DATOS
# ------------------------------------------------------------

data = np.loadtxt(
    "data/customer_campaign.csv",
    delimiter=",",
    skiprows=1
)

customer_id = data[:, 0]
channel = data[:, 1]
purchased = data[:, 2]
order_value = data[:, 3]

print("Número de clientes:", len(customer_id))


# 2. VARIABLE ALEATORIA
# ------------------------------------------------------------

# X = 1 si el cliente realizo una compra
# X = 0 si el cliente no realizo una compra

X = purchased

print("Primeros valores de X:")
print(X[:10])


# 3. PROBABILIDAD EMPÍRICA
# ------------------------------------------------------------

def empirical_probability(event):
    """
    Calcula la proporción de observaciones
    para las cuales un evento es verdadero.
    """
    return np.mean(event)


p_purchase = empirical_probability(purchased == 1)
p_no_purchase = empirical_probability(purchased == 0)

print("P(compra):", p_purchase)
print("P(no compra):", p_no_purchase)


# 4. PMF
# Probability Mass Function
# Función de Masa de Probabilidad
# ------------------------------------------------------------

def empirical_pmf(values):
    """
    Calcula los valores posibles de una variable
    discreta y sus probabilidades empíricas.
    """
    unique_values, counts = np.unique(values, return_counts=True)
    probabilities = counts / len(values)
    return unique_values, probabilities


pmf_values, pmf_probabilities = empirical_pmf(X)

print("Valores de la variable:", pmf_values)
print("Probabilidades:", pmf_probabilities)


# 5. VERIFICACIÓN DE LA PMF
# ------------------------------------------------------------

pmf_sum = np.sum(pmf_probabilities)

print("Suma de probabilidades:", pmf_sum)


# 6. VALOR ESPERADO
# ------------------------------------------------------------

def expected_value(values, probabilities):
    """
    Calcula E[X].
    """
    return np.sum(values * probabilities)


expected_purchase = expected_value(pmf_values, pmf_probabilities)

print("Valor esperado de X:", expected_purchase)


# 7. VARIANZA
# ------------------------------------------------------------

def variance(values, probabilities):
    """
    Calcula Var(X).
    """
    mean = expected_value(values, probabilities)
    return np.sum((values - mean) ** 2 * probabilities)


purchase_variance = variance(pmf_values, pmf_probabilities)

print("Varianza de X:", purchase_variance)


# 8. MUESTREO
# ------------------------------------------------------------

np.random.seed(42)

sample = np.random.choice(X, size=10, replace=False)
sample_probability = empirical_probability(sample == 1)

print("Probabilidad de compra en la población:", p_purchase)
print("Probabilidad de compra en la muestra:", sample_probability)


# 9. PROBABILIDAD CONDICIONAL
# ------------------------------------------------------------

def conditional_probability(event, condition):
    """
    Calcula P(evento | condición).
    """
    return np.mean(event[condition])


email_condition = channel == 0
social_condition = channel == 1

p_purchase_email = conditional_probability(purchased == 1, email_condition)
p_purchase_social = conditional_probability(purchased == 1, social_condition)

print("P(compra | email):", p_purchase_email)
print("P(compra | social):", p_purchase_social)


# 10. CONEXIÓN CON MACHINE LEARNING
# ------------------------------------------------------------

# En un problema de clasificación binaria, la variable objetivo
# es purchased. Lo que estamos haciendo aquí es estimar
# la probabilidad de que un cliente compre
# segun ciertas caracteristicas, lo cual es la base de
# muchos modelos de clasificación.

# 11. PREGUNTAS DE INTERPRETACIÓN
# ------------------------------------------------------------

# 1. ¿Por qué purchased puede considerarse una variable aleatoria Bernoulli?
# Respuesta:
# Porque solo puede tomar dos valores: 0 o 1.
# Eso representa un exito o fracaso, es decir, compra o no compra.

# 2. ¿Qué representa P(compra)?
# Respuesta:
# Es la proporcion de clientes que realizaron una compra sobre el total
# de clientes analizados.

# 3. ¿Por qué las probabilidades de la PMF deben sumar 1?
# Respuesta:
# Porque cubren todos los resultados posibles de la variable,
# y la suma total de probabilidades siempre debe ser 1.

# 4. ¿Qué representa E[X]?
# Respuesta:
# Es el valor esperado, o la probabilidad promedio de compra.

# 5. ¿Qué representa la varianza?
# Respuesta:
# Mide qué tan dispersos están los valores alrededor de la media.

# 6. ¿Por qué la muestra puede diferir de la población?
# Respuesta:
# Porque la muestra es solo un subconjunto aleatorio
# y puede fluctuar por azar.

# 7. ¿Cuál es la diferencia entre P(compra) y P(compra | email)?
# Respuesta:
# P(compra) es la probabilidad global.
# P(compra | email) es la probabilidad condicionada
# a que el cliente haya sido contactado por email.

# 8. ¿Cómo se relaciona con Machine Learning?
# Respuesta:
# Es la base de la clasificación binaria:
# estimar la probabilidad de que la clase positiva ocurra
# dado un conjunto de variables de entrada.