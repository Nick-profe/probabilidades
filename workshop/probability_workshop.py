import numpy as np

# TALLER DE PROBABILIDAD APLICADA A MACHINE LEARNING
# ============================================================

# Nombre: Sebastian
# Apellido 1: Giraldo
# Apellido 2: Acosta
# Rama: giraldo_acosta


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

    event = np.asarray(event, dtype=bool)
    return np.mean(event)


# calcular P(compra)
p_purchase = empirical_probability(X == 1)

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

    values = np.asarray(values)
    unique_values, counts = np.unique(values, return_counts=True)
    probabilities = counts / len(values)
    return unique_values, probabilities


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


# muestra aleatoria de 10 observaciones de X sin reemplazo
sample = np.random.choice(X, size=10, replace=False)

# probabilidad de compra dentro de la muestra
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

    event = np.asarray(event, dtype=bool)
    condition = np.asarray(condition, dtype=bool)
    return np.mean(event[condition])


# channel == 0 representa email
email_condition = channel == 0

# channel == 1 representa social
social_condition = channel == 1


# calcular P(compra | email)
p_purchase_email = conditional_probability(X == 1, email_condition)

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

# # 1. ¿Por qué purchased puede considerarse
#    una variable aleatoria Bernoulli?
# Respuesta: Porque describe un experimento con solo dos resultados
# posibles y excluyentes para cada cliente: compra (X = 1) o no compra
# (X = 0). La distribución queda determinada por un único parámetro p,
# la probabilidad de éxito, que aquí estimamos como 0.575.


# 2. ¿Qué representa P(compra) dentro del problema?
# Respuesta: Es la tasa de conversión de la campaña: la proporción de
# clientes que realizaron una compra. Con P(compra) = 0.575, el 57.5 %
# de los clientes compró. Es una estimación empírica de p a partir
# de los datos observados.


# 3. ¿Por qué las probabilidades de la PMF
#    deben sumar 1?
# Respuesta: Porque los valores posibles de X (0 y 1) cubren todos los
# resultados posibles y son mutuamente excluyentes: todo cliente
# compra o no compra. La probabilidad de que ocurra alguno de ellos
# es 1. En nuestro caso 0.425 + 0.575 = 1.0.


# 4. ¿Qué representa E[X] cuando X es
#    la variable purchased?
# Respuesta: Para una Bernoulli, E[X] = 0·(1 − p) + 1·p = p, así que el
# valor esperado coincide con la probabilidad de compra (0.575). Se
# interpreta como la tasa de compra promedio: si se repitiera el
# experimento con muchos clientes, en promedio el 57.5 % compraría.


# 5. ¿Qué representa la varianza de X?
# Respuesta: Mide la dispersión o incertidumbre del resultado de
# compra. Para una Bernoulli, Var(X) = p(1 − p) = 0.575 · 0.425 =
# 0.244375. Es máxima (0.25) cuando p = 0.5, es decir, cuando es más
# difícil predecir si un cliente comprará; nuestro valor está cerca
# de ese máximo, lo que indica alta incertidumbre.


# 6. ¿Por qué la probabilidad observada en una
#    muestra puede ser diferente de la probabilidad
#    observada en toda la población?
# Respuesta: Por la variabilidad muestral. Una muestra pequeña
# (10 clientes) puede, por azar, incluir más o menos compradores que
# la proporción real. Por eso obtuvimos 0.4 en la muestra frente a
# 0.575 en la población. A mayor tamaño de muestra, la estimación
# tiende a acercarse al valor poblacional (ley de los grandes números).


# 7. ¿Cuál es la diferencia entre
#    P(compra) y P(compra | email)?
# Respuesta: P(compra) es la probabilidad de compra considerando a todos
# los clientes (0.575). P(compra | email) restringe el análisis solo a
# los clientes contactados por email (0.65). La condicional incorpora
# información adicional: el canal. Como P(compra | email) = 0.65 es
# mayor que P(compra | social) = 0.5, el canal parece estar asociado
# con la probabilidad de compra.


# 8. ¿Cómo se relaciona P(Y = 1 | X)
#    con un problema de clasificación
#    en Machine Learning?
# Respuesta: En clasificación binaria, un modelo (por ejemplo, regresión
# logística) aprende a estimar P(Y = 1 | X): la probabilidad de que un
# cliente compre dadas sus características (canal, historial, etc.).
# Lo que calculamos en la sección 9 es una versión simple de esto con
# una sola característica (el canal). Luego se aplica un umbral, por
# ejemplo 0.5, para clasificar a cada cliente como comprador o no.