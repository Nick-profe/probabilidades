import numpy as np

# TALLER DE PROBABILIDAD APLICADA A MACHINE LEARNING
# ============================================================

# Nombre:juan diego
# Apellido 1:lopez
# Apellido 2:valencia
# Rama:lopez_valencia


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

    unique_values = np.unique(values)

    probabilities = np.array([
        np.mean(values == value)
        for value in unique_values
    ])

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


expected_purchase = expected_value(
    pmf_values,
    pmf_probabilities
)

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

    return np.sum(
        (values - mean) ** 2 * probabilities
    )


purchase_variance = variance(
    pmf_values,
    pmf_probabilities
)



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

sample = np.random.choice(
    X,
    size=10,
    replace=False
)
# TODO:
# calcular la probabilidad de compra
# dentro de la muestra.

sample_probability = np.mean(sample == 1)

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
    return np.mean(event & condition) / np.mean(condition)


# channel == 0 representa email
email_condition = channel == 0

# channel == 1 representa social
social_condition = channel == 1


# TODO:
# calcular P(compra | email)
p_purchase_email = conditional_probability(
    purchased == 1,
    email_condition
)
# TODO:
# calcular P(compra | social)
p_purchase_social = conditional_probability(
    purchased == 1,
    social_condition
)
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
# Porque purchased solo puede tomar dos valores:
# 0 (no compra) o 1 (compra).


# 2. ¿Qué representa P(compra) dentro del problema?
# Respuesta:
# Representa la proporción de clientes que realizaron
# una compra dentro del conjunto de datos.


# 3. ¿Por qué las probabilidades de la PMF
#    deben sumar 1?
# Respuesta:
# Porque la PMF representa todos los posibles resultados
# de la variable aleatoria y alguno de ellos debe ocurrir.


# 4. ¿Qué representa E[X] cuando X es
#    la variable purchased?
# Respuesta:
# Representa el promedio esperado de la variable purchased.
# En una variable Bernoulli también corresponde a la
# probabilidad de compra.


# 5. ¿Qué representa la varianza de X?
# Respuesta:
# Representa qué tanto se dispersan los valores de purchased
# respecto a su valor esperado.


# 6. ¿Por qué la probabilidad observada en una
#    muestra puede ser diferente de la probabilidad
#    observada en toda la población?
# Respuesta:
# Porque una muestra contiene solo una parte de las
# observaciones y puede tener una composición diferente
# debido al azar.


# 7. ¿Cuál es la diferencia entre
#    P(compra) y P(compra | email)?
# Respuesta:
# P(compra) representa la probabilidad de compra
# considerando todos los clientes.
# P(compra | email) representa la probabilidad de compra
# únicamente entre los clientes contactados por email.


# 8. ¿Cómo se relaciona P(Y = 1 | X)
#    con un problema de clasificación
#    en Machine Learning?
# Respuesta:
# Representa la probabilidad de que una observación
# pertenezca a la clase 1 dadas sus características X.
# Los modelos de clasificación pueden utilizar esta
# probabilidad para determinar la clase predicha.