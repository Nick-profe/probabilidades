import numpy as np

# TALLER DE PROBABILIDAD APLICADA A MACHINE LEARNING
# ============================================================

# Nombre:
# Apellido 1:
# Apellido 2:
# Rama:


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

# La variable purchased representa:
#
# X = 1 si el cliente realizó una compra
# X = 0 si el cliente no realizó una compra
#
# Por tanto, X puede interpretarse como una
# variable aleatoria Bernoulli.

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

    event = np.asarray(event, dtype=float)
    return np.mean(event)


# P(compra)

p_purchase = empirical_probability(X)


# P(no compra)

p_no_purchase = empirical_probability(1 - X)


print(
    "P(compra):",
    p_purchase
)

print(
    "P(no compra):",
    p_no_purchase
)


# 4. PMF
# Probability Mass Function
# Función de Masa de Probabilidad
# ------------------------------------------------------------

def empirical_pmf(values):
    """
    Calcula los valores posibles de una variable
    discreta y sus probabilidades empíricas.
    """

    values = np.asarray(values)
    unique_values, counts = np.unique(values, return_counts=True)
    probabilities = counts / len(values)

    return unique_values, probabilities


pmf_values, pmf_probabilities = empirical_pmf(X)


print(
    "Valores de la variable:",
    pmf_values
)

print(
    "Probabilidades:",
    pmf_probabilities
)


# 5. VERIFICACIÓN DE LA PMF
# ------------------------------------------------------------

# La suma de todas las probabilidades
# de una PMF debe ser igual a 1.

pmf_sum = np.sum(pmf_probabilities)

print(
    "Suma de probabilidades:",
    pmf_sum
)


# 6. VALOR ESPERADO
# ------------------------------------------------------------

def expected_value(
    values,
    probabilities
):
    """
    Calcula E[X].
    """

    values = np.asarray(values, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    return np.sum(values * probabilities)


expected_purchase = expected_value(pmf_values, pmf_probabilities)


print(
    "Valor esperado de X:",
    expected_purchase
)


# 7. VARIANZA
# ------------------------------------------------------------

def variance(
    values,
    probabilities
):
    """
    Calcula Var(X).
    """

    values = np.asarray(values, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    expected = expected_value(values, probabilities)
    return np.sum(((values - expected) ** 2) * probabilities)


purchase_variance = variance(pmf_values, pmf_probabilities)


print(
    "Varianza de X:",
    purchase_variance
)


# 8. MUESTREO
# ------------------------------------------------------------

# Para que los resultados sean reproducibles
np.random.seed(42)


# seleccionar una muestra aleatoria
# de 10 observaciones de X
# sin reemplazo.

sample = np.random.choice(X, size=10, replace=False)


# calcular la probabilidad de compra
# dentro de la muestra.

sample_probability = empirical_probability(sample)


print(
    "Probabilidad de compra en la población:",
    p_purchase
)

print(
    "Probabilidad de compra en la muestra:",
    sample_probability
)


# 9. PROBABILIDAD CONDICIONAL
# ------------------------------------------------------------

def conditional_probability(
    event,
    condition
):
    """
    Calcula P(evento | condición).
    """

    event = np.asarray(event, dtype=bool)
    condition = np.asarray(condition, dtype=bool)

    if np.sum(condition) == 0:
        return np.nan

    return np.mean(event[condition])


# channel == 0 representa email
email_condition = channel == 0

# channel == 1 representa social
social_condition = channel == 1


# P(compra | email)

p_purchase_email = conditional_probability(X, email_condition)


# P(compra | social)

p_purchase_social = conditional_probability(X, social_condition)


print(
    "P(compra | email):",
    p_purchase_email
)

print(
    "P(compra | social):",
    p_purchase_social
)


# 10. CONEXIÓN CON MACHINE LEARNING
# ------------------------------------------------------------

# En un problema de clasificación binaria,
# podemos interpretar:
#
# Y = purchased
#
# y buscar modelos que estimen:
#
# P(Y = 1 | X)
#
# donde X representa características del cliente.


# 11. PREGUNTAS DE INTERPRETACIÓN
# ------------------------------------------------------------

# 1. ¿Por qué purchased puede considerarse
#    una variable aleatoria Bernoulli?
# Respuesta:
# Porque cada cliente solo puede tomar dos resultados posibles: 1 si compra o 0 si no compra.
# Ese tipo de variable binaria es precisamente una Bernoulli.


# 2. ¿Qué representa P(compra) dentro del problema?
# Respuesta:
# Representa la proporción de clientes que compraron en la población total analizada.


# 3. ¿Por qué las probabilidades de la PMF
#    deben sumar 1?
# Respuesta:
# Porque la PMF describe todas las posibilidades de la variable aleatoria y, en conjunto,
# abarcan todo el espacio de resultados posibles.


# 4. ¿Qué representa E[X] cuando X es
#    la variable purchased?
# Respuesta:
# Representa el valor esperado de compra, que coincide con la probabilidad de compra
# cuando X sigue una Bernoulli.


# 5. ¿Qué representa la varianza de X?
# Respuesta:
# Mide cuánto se alejan los resultados de su valor esperado; en este caso, muestra la dispersión
# de comprar o no comprar alrededor de la media.


# 6. ¿Por qué la probabilidad observada en una
#    muestra puede ser diferente de la probabilidad
#    observada en toda la población?
# Respuesta:
# Porque la muestra es un subconjunto aleatorio y puede tener variabilidad por azar.
# Cuanto mayor sea la muestra, la estimación tiende a acercarse a la probabilidad poblacional.


# 7. ¿Cuál es la diferencia entre
#    P(compra) y P(compra | email)?
# Respuesta:
# P(compra) es la probabilidad general de compra en toda la población.
# P(compra | email) es la probabilidad de compra solo entre los clientes contactados por email.


# 8. ¿Cómo se relaciona P(Y = 1 | X)
#    con un problema de clasificación
#    en Machine Learning?
# Respuesta:
# Es exactamente la probabilidad de que la clase positiva ocurra dado un conjunto de características X.
# En clasificación binaria, un modelo intenta estimar esa probabilidad para decidir si un ejemplo
# pertenece a la clase 1 o a la clase 0.