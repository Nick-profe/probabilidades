import numpy as np

# TALLER DE PROBABILIDAD APLICADA A MACHINE LEARNING
# ============================================================

# Nombre: Karolain
# Apellido 1: Duque
# Apellido 2: Saavedra
# Rama: duque_saavedra


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

# calcular P(compra)

p_purchase = empirical_probability(X==1)


# calcular P(no compra)

p_no_purchase = empirical_probability(X==0)


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


pmf_values = empirical_pmf(X)
pmf_probabilities = empirical_pmf(X)


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
    return np.sum(values*probabilities)


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
    mean = expected_value(
        values,
        probabilities
    )

    # TODO: completar

    return np.sum(
        (values - mean)** 2 * probabilities
    )


purchase_variance = variance(
    pmf_values,
    pmf_probabilities)


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
    size = 10,
    replace = False
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
    if np.sum(condition) == 0:
        return 0.0
    
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
    X == 0,
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
# Respuesta: purchased puede considerarse una variable aleatoria Bernoulli porque solo tiene dos posibles resultados, 1 si el cliente compra y 0 si no lo hace.


# 2. ¿Qué representa P(compra) dentro del problema?
# Respuesta: Dentro del problema P(compra) representa la proporcion de clientes que realizaron una compra dentro del conjunto de datos.


# 3. ¿Por qué las probabilidades de la PMF
#    deben sumar 1?
# Respuesta: Las probabilidades de la PMF deben sumar 1 porque esta distribuye toda la probabilidad entre los posibles valores de la variable aleatoria. 


# 4. ¿Qué representa E[X] cuando X es
#    la variable purchased?
# Respuesta: Cuando X es la variable purchased, E[X] representa el promedio esperado de compras. Como X toma los valores de 0 y 1 E[X] tambien representa la probabilidad de compra.


# 5. ¿Qué representa la varianza de X?
# Respuesta: La varianza de X representa que tanto se dispersan los resultados de compra y no compra alrededor de su valor esperado.


# 6. ¿Por qué la probabilidad observada en una
#    muestra puede ser diferente de la probabilidad
#    observada en toda la población?
# Respuesta: Porque una muestra contiene una parte de las observaciones de la poblacion. Debido al muestreo aleatorio, sus resultados pueden variar.


# 7. ¿Cuál es la diferencia entre
#    P(compra) y P(compra | email)?
# Respuesta: P(compra) representa la probailidad general de compra mientras P(compra | email) representa la probabilidad de compra cuando sabemos que el cliente pertenece al canal email.


# 8. ¿Cómo se relaciona P(Y = 1 | X)
#    con un problema de clasificación
#    en Machine Learning?
# Respuesta: Representa la probabilidad estimada de que la variable objetivo "Y"" pertenezca a la clase 1 dadas las  caracteristicas X del cliente. Esta probabilidad puede utilizarse para tomar una decision de clasificacion.