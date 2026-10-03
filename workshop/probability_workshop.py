import numpy as np

# TALLER DE PROBABILIDAD APLICADA A MACHINE LEARNING
# ============================================================

# Nombre: Johan
# Apellido 1: Jurado
# Apellido 2: Lloreda
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

    unique_values, counts = np.unique(
        values,
        return_counts=True
    )
    probabilities = counts / len(values)

    return unique_values, probabilities



# TODO:
# utilizar empirical_pmf(X)
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

    # TODO: completar
    mean = expected_value(values, probabilities)

    return np.sum(
        probabilities * (values - mean) ** 2
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

sample_probability = np.mean(
    sample == 1
)


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
    return np.mean(
        event[condition]
    )


# channel == 0 representa email
email_condition = channel == 0

# channel == 1 representa social
social_condition = channel == 1


# TODO:
# calcular P(compra | email)

p_purchase_email = conditional_probability(
    X == 1,
    email_condition
)


# TODO:
# calcular P(compra | social)

p_purchase_social = conditional_probability(
    X == 1,
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
# Respuesta: porque solo toma dos valores (1 = compra, 0 = no compra)


# 2. ¿Qué representa P(compra) dentro del problema?
# Respuesta: representa la proporción de clientes que compraron.


# 3. ¿Por qué las probabilidades de la PMF
#    deben sumar 1?
# Respuesta: la PMF cubre todos los resultados posibles y alguno debe ocurrir:
# 0.425 + 0.575 = 1.


# 4. ¿Qué representa E[X] cuando X es
#    la variable purchased?
# Respuesta: representa el promedio esperado de compras. En este caso, 0.575.


# 5. ¿Qué representa la varianza de X?
# Respuesta: representa la dispersión de los valores de X respecto a su promedio.


# 6. ¿Por qué la probabilidad observada en una
#    muestra puede ser diferente de la probabilidad
#    observada en toda la población?
# Respuesta: porque una muestra contiene solo una parte de la población y existe variabilidad muestral.


# 7. ¿Cuál es la diferencia entre
#    P(compra) y P(compra | email)?
# Respuesta: P(compra) no considera el canal; P(compra | email) considera que el cliente fue contactado por email.


# 8. ¿Cómo se relaciona P(Y = 1 | X)
#    con un problema de clasificación
#    en Machine Learning?
# Respuesta: Es la probabilidad que estima un clasificador de que la clase sea 1 dadas las características X