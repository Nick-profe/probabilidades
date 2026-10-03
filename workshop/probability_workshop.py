import numpy as np

# TALLER DE PROBABILIDAD APLICADA A MACHINE LEARNING
# ============================================================

# Nombre: Jimmy
# Apellido 1: Lopez
# Apellido 2: Yule
# Rama: lopez_yule


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

    # TODO: completar

    return np.mean(event)


# TODO:
# calcular P(compra)

p_purchase = empirical_probability(X == 1)


# TODO:
# calcular P(no compra)

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

    # TODO: completar

    unique_values,counts = np.unique(values, return_counts=True)
    probabilities = counts / len(values)

    return unique_values, probabilities


pmf_values, pmf_probabilities = empirical_pmf(X)


# TODO:
# utilizar empirical_pmf(X)


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

    # TODO: completar
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

    # TODO: completar
    mean = expected_value(values, probabilities)

    return np.sum(probabilities * (values - mean) ** 2)


purchase_variance = variance(pmf_values, pmf_probabilities)


print(
    "Varianza de X:",
    purchase_variance
)


# 8. MUESTREO
# ------------------------------------------------------------

# Para que los resultados sean reproducibles
np.random.seed(42)


# TODO:
# seleccionar una muestra aleatoria
# de 10 observaciones de X
# sin reemplazo.

sample = np.random.choice(X, size=10, replace=False)


# TODO:
# calcular la probabilidad de compra
# dentro de la muestra.

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

    # TODO: completar
    return np.mean(event[condition])


# channel == 0 representa email
email_condition = channel == 0

# channel == 1 representa social
social_condition = channel == 1


# TODO:
# calcular P(compra | email)

p_purchase_email = conditional_probability(X == 1, email_condition)


# TODO:
# calcular P(compra | social)

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
# Respuesta: Solo puede tomar dos valores: 1 si el cliente compra y 0 si no compra, donde p representa la probabilidad de compra.


# 2. ¿Qué representa P(compra) dentro del problema?
# Respuesta: Es la probabilidad de que un cliente realice una compra


# 3. ¿Por qué las probabilidades de la PMF
#    deben sumar 1?
# Respuesta:porque los valores posibles de la variable y la suma de las probabilidades de todos los eventos posibles debe ser 1


# 4. ¿Qué representa E[X] cuando X es
#    la variable purchased?
# Respuesta: Equivale a la probabilidad de que un cliente realice una compra


# 5. ¿Qué representa la varianza de X?
# Respuesta: representa la dispersión de las probabilidades con respecto a purchased respecto a la media, entre la compra y no compra


# 6. ¿Por qué la probabilidad observada en una
#    muestra puede ser diferente de la probabilidad
#    observada en toda la población?
# Respuesta: la muestra contiene solo una parte de la poblacion y no representa la poblacion completa


# 7. ¿Cuál es la diferencia entre
#    P(compra) y P(compra | email)?
# Respuesta: P(compra) es la probabilidad de que un cliente realice una compra sin considerar ningún factor adicional, mientras que P(compra | email) es la probabilidad de que un cliente realice una compra dado que ha recibido un correo electrónico, lo que introduce una condición específica en el cálculo de la probabilidad.


# 8. ¿Cómo se relaciona P(Y = 1 | X)
#    con un problema de clasificación
#    en Machine Learning?
# Respuesta: P(Y = 1 | X) representa la probabilidad de que un cliente realice una compra (Y = 1) dado un conjunto de características del cliente (X). En un problema de clasificación en Machine Learning, el objetivo es construir un modelo que pueda predecir esta probabilidad basándose en los datos de entrenamiento, permitiendo así clasificar nuevos clientes como compradores o no compradores.