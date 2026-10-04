import numpy as np

# TALLER DE PROBABILIDAD APLICADA A MACHINE LEARNING
# ============================================================

# Nombre: David Santiago 
# Apellido 1: Alderete  
# Apellido 2: Patiño
# Rama: alderete_patino


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

    ev = expected_value(values, probabilities)

    return np.sum(((values - ev) ** 2) * probabilities)

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



# calcular P(compra | email)

p_purchase_email = conditional_probability(purchased == 1, email_condition)




# calcular P(compra | social)

p_purchase_social = conditional_probability(purchased == 1, social_condition)


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
# Respuesta: Porque es una variable que solo puede tomar dos valores posibles: 1 (si el cliente realizó una compra) y 0 (si no realizó una compra), lo cual es característico de una distribución Bernoulli.


# 2. ¿Qué representa P(compra) dentro del problema?
# Respuesta: P(compra) representa la probabilidad de que un cliente seleccionado al azar haya realizado una compra, es decir, la proporción de clientes que compraron en relación con el total de clientes.


# 3. ¿Por qué las probabilidades de la PMF
#    deben sumar 1?
# Respuesta: Porque la PMF describe la distribución de probabilidad de una variable aleatoria discreta, y la suma de todas las probabilidades debe ser igual a 1 para que sea una distribución válida.


# 4. ¿Qué representa E[X] cuando X es
#    la variable purchased?
# Respuesta: E[X] representa el valor esperado de la variable purchased, es decir, la probabilidad de que un cliente realice una compra.


# 5. ¿Qué representa la varianza de X?
# Respuesta: La varianza de X representa la dispersión de los valores de la variable purchased alrededor de su valor esperado.

# 6. ¿Por qué la probabilidad observada en una
#    muestra puede ser diferente de la probabilidad
#    observada en toda la población?
# Respuesta: La probabilidad observada en una muestra puede diferir de la probabilidad en toda la población debido a la variabilidad inherente al muestreo. Una muestra puede no ser representativa de la población completa, especialmente si es pequeña o si se selecciona de manera sesgada, lo que puede llevar a estimaciones diferentes de las probabilidades reales.


# 7. ¿Cuál es la diferencia entre
#    P(compra) y P(compra | email)?
# Respuesta: P(compra) representa la probabilidad de que un cliente seleccionado al azar haya realizado una compra, mientras que P(compra | email) representa la probabilidad de que un cliente que utiliza el canal de email haya realizado una compra.


# 8. ¿Cómo se relaciona P(Y = 1 | X)
#    con un problema de clasificación
#    en Machine Learning?
# Respuesta: P(Y = 1 | X) representa la probabilidad de que un cliente realice una compra dado un conjunto de características (X). En un problema de clasificación en Machine Learning, el objetivo es construir un modelo que pueda predecir esta probabilidad para nuevos clientes basándose en sus características, lo que permite tomar decisiones informadas sobre estrategias de marketing y segmentación de clientes.