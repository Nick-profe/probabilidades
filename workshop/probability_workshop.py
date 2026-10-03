import numpy as np

# TALLER DE PROBABILIDAD APLICADA A MACHINE LEARNING
# ============================================================

# Nombre: Juan Camilo
# Apellido 1: Henao 
# Apellido 2: Espinosa
# Rama: Henao_Espinosa


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

    mean = expected_value(values, probabilities)
    return np.sum(((values - mean) ** 2) * probabilities)


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

    condition_count = np.count_nonzero(condition)
    if condition_count == 0:
        return np.nan
    return np.count_nonzero(event & condition) / condition_count


# channel == 0 representa email
email_condition = channel == 0

# channel == 1 representa social
social_condition = channel == 1


p_purchase_email = conditional_probability(
    purchased == 1,
    email_condition
)


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
# Respuesta: Porque solo puede tomar dos valores: 1 si el cliente compra y 0 si no compra.
# Su parámetro p representa la probabilidad de compra.


# 2. ¿Qué representa P(compra) dentro del problema?
# Respuesta: Es la proporción de clientes que compraron en la campaña. En estos datos es 0.575,
# es decir, el 57.5 % de los clientes.


# 3. ¿Por qué las probabilidades de la PMF
#    deben sumar 1?
# Respuesta: Porque la PMF incluye todos los valores posibles de la variable, que son eventos
# mutuamente excluyentes y exhaustivos. Por tanto, sus probabilidades cubren el 100 % de los casos.


# 4. ¿Qué representa E[X] cuando X es
#    la variable purchased?
# Respuesta: Es el promedio esperado del indicador de compra. Como purchased vale 1 al comprar
# y 0 al no comprar, E[X] equivale a la probabilidad de compra: 0.575.


# 5. ¿Qué representa la varianza de X?
# Respuesta: Mide cuánto varía purchased respecto a su media, es decir, cuánta incertidumbre hay
# entre compra y no compra. En estos datos, Var(X) = 0.244375.


# 6. ¿Por qué la probabilidad observada en una
#    muestra puede ser diferente de la probabilidad
#    observada en toda la población?
# Respuesta: Porque una muestra contiene solo parte de los clientes y su composición depende
# del azar. Esta variabilidad de muestreo suele ser mayor cuando la muestra es pequeña.


# 7. ¿Cuál es la diferencia entre
#    P(compra) y P(compra | email)?
# Respuesta: P(compra) considera a todos los clientes y vale 0.575. P(compra | email) considera
# solo a quienes recibieron email y vale 0.65 en estos datos.


# 8. ¿Cómo se relaciona P(Y = 1 | X)
#    con un problema de clasificación
#    en Machine Learning?
# Respuesta: Es la probabilidad estimada de que la clase sea positiva (Y = 1), dadas las
# características X del cliente. Un clasificador puede usarla para asignar una clase, por ejemplo,
# prediciendo compra cuando la probabilidad supera un umbral. Aquí X representa características,
# no la variable purchased usada como X en las secciones anteriores.