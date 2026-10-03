import numpy as np

# TALLER DE PROBABILIDAD APLICADA A MACHINE LEARNING
# ============================================================

# Nombre: Sebastián 
# Apellido 1: Moreno 
# Apellido 2: Dorado
# Rama: moreno_dorado


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

    unique_values, counts = np.unique(
        values,
        return_counts=True
    )

    probabilities = counts / len(values)

    return unique_values, probabilities


# Utilizar empirical_pmf(X)

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

    expected_value = np.sum(
    values * probabilities
   
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
    variance = np.sum((values - mean) ** 2 * probabilities)

    return variance


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
# Respuesta: Porque solo puede tomar dos valores, 0 o 1, representando no compra y compra respectivamente.


# 2. ¿Qué representa P(compra) dentro del problema?
# Respuesta: La probabilidad de que un cliente realice una compra, es decir, P(X = 1).

# 3. ¿Por qué las probabilidades de la PMF
#    deben sumar 1?
# Respuesta: Porque la PMF representa la distribución de probabilidad de una variable aleatoria discreta, y la suma de todas las probabilidades posibles debe ser igual a 1 para cumplir con las propiedades de una distribución de probabilidad.


# 4. ¿Qué representa E[X] cuando X es
#    la variable purchased?
# Respuesta: La esperanza de X representa la probabilidad de compra en la población, es decir, E[X] = P(X = 1).


# 5. ¿Qué representa la varianza de X?
# Respuesta: La varianza de X representa la dispersión de los valores de la variable purchased alrededor de su media, indicando qué tan consistente es el comportamiento de compra entre los clientes.


# 6. ¿Por qué la probabilidad observada en una
#    muestra puede ser diferente de la probabilidad
#    observada en toda la población?
# Respuesta: Porque la muestra es un subconjunto de la población y puede no ser representativa de todos los elementos de la población.


# 7. ¿Cuál es la diferencia entre
#    P(compra) y P(compra | email)?
# Respuesta: P(compra) es la probabilidad marginal de que un cliente realice una compra, mientras que P(compra | email) es la probabilidad condicional de que un cliente realice una compra dado que se le ha enviado un correo electrónico.


# 8. ¿Cómo se relaciona P(Y = 1 | X)
#    con un problema de clasificación
#    en Machine Learning?
# Respuesta: P(Y = 1 | X) representa la probabilidad de que un cliente realice una compra dado sus características (X). En un problema de clasificación binaria, el objetivo es construir un modelo que pueda predecir esta probabilidad para nuevos clientes basándose en sus características, lo que permite tomar decisiones informadas sobre estrategias de marketing y segmentación de clientes.