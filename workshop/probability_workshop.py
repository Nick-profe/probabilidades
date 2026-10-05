import numpy as np

# TALLER DE PROBABILIDAD APLICADA A MACHINE LEARNING
# ============================================================

# Nombre: Violeta Isabella
# Apellido 1: España
# Apellido 2: Bolaños
# Rama: Espana_Bolanos


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

    return np.mean(event)

p_purchase = empirical_probability(X == 1)

p_no_purchase = empirical_probability(X == 0)


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

    unique_values, counts = np.unique(values, return_counts=True)
    probabilities = counts / len(values)

    return unique_values, probabilities


pmf_values,pmf_probabilities = empirical_pmf(X)


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

    expected_value = np.sum(values * probabilities)

    return expected_value

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

    mean = expected_value(values, probabilities)
    return np.sum(probabilities * (values - mean)**2)


purchase_variance = variance(pmf_values, pmf_probabilities)


print(
    "Varianza de X:",
    purchase_variance
)


# 8. MUESTREO
# ------------------------------------------------------------

# Para que los resultados sean reproducibles
np.random.seed(42)


sample = np.random.choice(X, size=10, replace=False)

sample_probability = empirical_probability(sample == 1)


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
    
    return np.mean(event[condition])


# channel == 0 representa email
email_condition = channel == 0

# channel == 1 representa social
social_condition = channel == 1


# TODO:
# calcular P(compra | email)

p_purchase_email = conditional_probability(X == 1, email_condition)

p_purchase_social = conditional_probability(X == 1, social_condition)


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
# Respuesta: Purchased puede considerarse una variable aleatoria Bernoulli porque solo tiene dos posibles resultados: 1 (si el cliente realizó una compra) y 0 (si no realizó una compra).

# 2. ¿Qué representa P(compra) dentro del problema?
# Respuesta: P representa la probabilidad de que un cliente realice una compra sin considerar ninguna condición adicional.


# 3. ¿Por qué las probabilidades de la PMF
#    deben sumar 1?
# Respuesta: La PMF debe sumar 1 (100%) porque representa todas las posibilidades que puede tomar una variable, y alguna de ellas necesariamente debe ocurrir.

# 4. ¿Qué representa E[X] cuando X es
#    la variable purchased?
# Respuesta: E[X] representa la probabilidad esperada de que un cliente realice una compra.


# 5. ¿Qué representa la varianza de X?
# Respuesta: La varianza de X representa la medida de dispersión de los resultados de la variable aleatoria purchased, indicando qué tan lejos están los valores de la media esperada.


# 6. ¿Por qué la probabilidad observada en una
#    muestra puede ser diferente de la probabilidad
#    observada en toda la población?
# Respuesta: La probabilidad observada en una muestra puede ser diferente de la probabilidad observada en toda la población debido a la variabilidad presente en cada muestreo aleatorio.


# 7. ¿Cuál es la diferencia entre
#    P(compra) y P(compra | email)?
# Respuesta: P(compra) es la probabilidad general de que un cliente realice una compra sin considerar ninguna condición, mientras que P(compra | email) es la probabilidad de que un cliente realice una compra dado que recibió un correo electrónico.


# 8. ¿Cómo se relaciona P(Y = 1 | X)
#    con un problema de clasificación
#    en Machine Learning?
# Respuesta: La relación es que P(Y = 1 | X) representa la probabilidad de que un cliente realice una compra dado un conjunto de características (X), lo cual es el objetivo principal en un problema de clasificación binaria en Machine Learning, donde se busca predecir la clase (compra o no compra) basada en las características del cliente.