import numpy as np

# TALLER DE PROBABILIDAD APLICADA A MACHINE LEARNING
# ============================================================

# Nombre: jeison  
# Apellido 1: navarro 
# Apellido 2:murillo
# Rama:sarmiento_pradilla


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

p_purchase = empirical_probability(X==1)


# TODO:
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

    unique_values, counts = np.unique(values, return_counts=True)
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
    
    expected_value = np.sum(values * probabilities)
    
    return expected_value


expected_purchase =  expected_value(pmf_values, pmf_probabilities)


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

    mu = expected_value(values, probabilities)
    return np.sum(((values - mu) ** 2) * probabilities)


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

    # TODO: completar
    p_joint = np.mean(event & condition)
    p_condition = np.mean(condition)

    return p_joint / p_condition


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
# Respuesta:
# Porque describe un único experimento (observar a un
# cliente) con solo dos resultados posibles, 1 (compra) y 0 (no
# compra), donde P(X = 1) = p y P(X = 0) = 1 - p. Eso es
# exactamente la definición de una Bernoulli(p).


# 2. ¿Qué representa P(compra) dentro del problema?
# Respuesta:
# La proporción de clientes de la campaña que terminaron
# comprando, es decir, la tasa de conversión global. Es el parámetro
# p de la Bernoulli estimado de forma empírica.


# 3. ¿Por qué las probabilidades de la PMF
#    deben sumar 1?
# Respuesta:
# Porque la PMF reparte toda la probabilidad entre todos
# los resultados posibles y alguno de ellos ocurre con certeza. Aquí,
# cada cliente compra o no compra, así que P(0) + P(1) = 1.


# 4. ¿Qué representa E[X] cuando X es
#    la variable purchased?
# Respuesta:
# El promedio teórico de largo plazo de X. Como X solo
# toma 0 y 1, E[X] = 1*P(1) + 0*P(0) = p; es decir, coincide con la
# probabilidad de compra (la tasa de conversión).


# 5. ¿Qué representa la varianza de X?
# Respuesta:
# La dispersión de X alrededor de su media, o sea la
# incertidumbre sobre si un cliente comprará. En una Bernoulli,
# Var(X) = p(1 - p): es máxima cuando p = 0.5 (máxima
# incertidumbre) y cercana a 0 cuando p está cerca de 0 o de 1.



# 6. ¿Por qué la probabilidad observada en una
#    muestra puede ser diferente de la probabilidad
#    observada en toda la población?
# Respuesta:
# Por la variabilidad del muestreo: una muestra pequeña
# (aquí, 10 clientes) solo ve una parte de la población y puede
# sobre o subrepresentar a los compradores por azar. Según la Ley de
# los Grandes Números, al aumentar el tamaño de la muestra la
# proporción muestral tiende a acercarse a la poblacional.



# 7. ¿Cuál es la diferencia entre
#    P(compra) y P(compra | email)?
# Respuesta:
# P(compra) se calcula sobre todos los clientes, sin
# importar el canal. P(compra | email) restringe el universo solo a
# los clientes que llegaron por email: P(compra ∩ email) / P(email).
# Si ambas difieren, el canal aporta información sobre la compra
# (los eventos no son independientes).


# 8. ¿Cómo se relaciona P(Y = 1 | X)
#    con un problema de clasificación
#    en Machine Learning?
# Respuesta:
# Un clasificador binario estima, para cada cliente con
# características X, la probabilidad de que Y = 1 (compre). Luego
# esa probabilidad se convierte en una decisión comparándola con un
# umbral (por ejemplo, 0.5). P(compra | email) es el caso más simple:
# un "modelo" con una sola característica, el canal.