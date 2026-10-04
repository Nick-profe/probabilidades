import numpy as np

# TALLER DE PROBABILIDAD APLICADA A MACHINE LEARNING
# ============================================================

# Nombre: Juan Jose 
# Apellido 1: Melo
# Apellido 2: Montenegro
# Rama: melo_montenegro


# 1. CARGA DE DATOS
# ------------------------------------------------------------
# np.loadtxt lee el CSV como una matriz NumPy:
#   - delimiter="," -> columnas separadas por comas
#   - skiprows=1    -> se salta la fila de encabezados
# Luego se separa cada columna con slicing data[:, j]
# (todas las filas, columna j).

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

    # 'event' es un arreglo booleano (True/False), por ejemplo X == 1.
    # NumPy trata True como 1 y False como 0, así que el promedio
    # es exactamente:
    #
    #   P(A) = (# veces que ocurre A) / (# total de observaciones)
    #
    # Es la misma idea de la guía P1: np.mean(results == "cara").
    return np.mean(event)


# P(compra): proporción de clientes con X == 1
p_purchase = empirical_probability(X == 1)


# P(no compra): proporción de clientes con X == 0.
# Por la regla del complemento también debe cumplirse
# P(no compra) = 1 - P(compra).
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

    # np.unique devuelve:
    #   - unique_values: los valores distintos que aparecen (aquí 0 y 1)
    #   - counts: cuántas veces aparece cada uno
    # Dividir cada conteo entre el total da la frecuencia relativa,
    # que es la estimación empírica de P(X = x).

    unique_values, counts = np.unique(
        values,
        return_counts=True
    )
    probabilities = counts / len(values)

    return unique_values, probabilities


# La función devuelve dos cosas, así que se "desempaquetan"
# en dos variables a la vez.
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
# Igual que en la guía P1 (sección 9): print(np.sum(pmf)).
# Puede salir algo como 0.9999999999 por redondeo de
# punto flotante; eso cuenta como 1.

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

    # Fórmula para una variable discreta:
    #
    #   E[X] = sum_x  x * P(X = x)
    #
    # values * probabilities multiplica elemento a elemento
    # (0*P(0), 1*P(1)) y np.sum suma esos productos.
    # Para una Bernoulli: E[X] = 0*(1-p) + 1*p = p.
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

    # Fórmula:
    #
    #   Var(X) = E[(X - mu)^2] = sum_x (x - mu)^2 * P(X = x)
    #
    # Paso 1: mu = E[X] (se reutiliza la función anterior).
    # Paso 2: se calcula la distancia al cuadrado de cada valor
    #         a la media y se pondera por su probabilidad.
    # Para una Bernoulli el resultado es p * (1 - p).

    mu = expected_value(values, probabilities)

    return np.sum(
        (values - mu) ** 2 * probabilities
    )


purchase_variance = variance(
    pmf_values,
    pmf_probabilities
)


print(
    "Varianza de X:",
    purchase_variance
)

# Comprobaciones: debe coincidir con p(1-p) y con np.var(X)
print("Verificación p(1-p):", p_purchase * (1 - p_purchase))
print("Verificación np.var(X):", np.var(X))


# 8. MUESTREO
# ------------------------------------------------------------

# Para que los resultados sean reproducibles
np.random.seed(42)


# np.random.choice toma 10 elementos al azar de X.
# replace=False -> sin reemplazo: un mismo cliente no puede
# salir dos veces (como en la guía P2, sección 10).

sample = np.random.choice(
    X,
    size=10,
    replace=False
)


# Se reutiliza empirical_probability, pero ahora
# solo sobre los 10 clientes de la muestra.

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

    # Definición (guía P2, sección 14):
    #
    #   P(A | B) = P(A ∩ B) / P(B)
    #
    # En la guía se calcula contando:
    #   probability = email_buyers / email_users
    #
    # 1. event[condition] filtra el arreglo y deja solo las
    #    observaciones que cumplen la condición (mismo filtrado
    #    que sample_space[sample_space % 2 == 0] en la guía P1).
    # 2. len(...) cuenta cuántas cumplen la condición (los "users").
    # 3. np.sum(...) cuenta cuántas de ellas cumplen el evento,
    #    porque True vale 1 y False vale 0 (los "buyers").

    condition_users = len(event[condition])

    event_and_condition = np.sum(event[condition])

    return event_and_condition / condition_users


# channel == 0 representa email
email_condition = channel == 0

# channel == 1 representa social
social_condition = channel == 1


# P(compra | email): entre los clientes que llegaron por email,
# qué proporción compró.

p_purchase_email = conditional_probability(
    X == 1,
    email_condition
)


# P(compra | social): lo mismo para el canal social.

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
#
# El cálculo de P(compra | email) del punto 9 es la versión más
# simple de esa idea: un "modelo" que usa una sola característica
# (el canal). Con un umbral se convierte en predicción:

threshold = 0.5
prediction_email = int(p_purchase_email >= threshold)
prediction_social = int(p_purchase_social >= threshold)

print("Predicción para un cliente de email:", prediction_email)
print("Predicción para un cliente de social:", prediction_social)


# 11. PREGUNTAS DE INTERPRETACIÓN
# ------------------------------------------------------------

# 1. ¿Por qué purchased puede considerarse
#    una variable aleatoria Bernoulli?
# Respuesta:
# Porque representa un único experimento (lo que hace un cliente)
# con solo dos resultados posibles: compra (1) o no compra (0).
# Su distribución queda definida por un solo parámetro
# p = P(X = 1), y P(X = 0) = 1 - p.


# 2. ¿Qué representa P(compra) dentro del problema?
# Respuesta:
# La proporción de clientes de la campaña que realizaron una compra,
# es decir, la tasa de conversión. Es la estimación empírica del
# parámetro p de la Bernoulli.


# 3. ¿Por qué las probabilidades de la PMF
#    deben sumar 1?
# Respuesta:
# Porque la PMF reparte toda la probabilidad entre todos los valores
# posibles de la variable. Siempre ocurre alguno de ellos
# (cada cliente compra o no compra), así que la probabilidad total
# del espacio muestral es 1. Además, P(compra) + P(no compra) = 1.


# 4. ¿Qué representa E[X] cuando X es
#    la variable purchased?
# Respuesta:
# El promedio teórico de largo plazo de X. Como X es 0/1,
# E[X] = 0*(1-p) + 1*p = p: coincide con la probabilidad de compra.
# Se interpreta como el número esperado de compras por cliente.


# 5. ¿Qué representa la varianza de X?
# Respuesta:
# Mide qué tan dispersos están los valores alrededor de la media,
# es decir, la incertidumbre sobre si un cliente comprará o no.
# Para una Bernoulli, Var(X) = p(1-p). Es máxima (0.25) cuando p = 0.5,
# el caso más incierto, y se acerca a 0 cuando casi todos compran
# o casi nadie compra.


# 6. ¿Por qué la probabilidad observada en una
#    muestra puede ser diferente de la probabilidad
#    observada en toda la población?
# Respuesta:
# Por la variabilidad del muestreo: una muestra aleatoria pequeña
# (n = 10) puede, por azar, incluir más o menos compradores que la
# proporción real. Además, con n = 10 la probabilidad solo puede
# tomar valores 0.0, 0.1, ..., 1.0. Según la Ley de los Grandes
# Números, al aumentar el tamaño de la muestra la proporción muestral
# se acerca a la poblacional.


# 7. ¿Cuál es la diferencia entre
#    P(compra) y P(compra | email)?
# Respuesta:
# P(compra) es la probabilidad de compra considerando a todos los
# clientes. P(compra | email) es la probabilidad de compra restringida
# a los clientes que llegaron por email: se calcula como
# P(compra ∩ email) / P(email). Si ambas son distintas, el canal aporta
# información sobre la compra (los eventos no son independientes).
# Si fueran iguales, compra y email serían independientes.


# 8. ¿Cómo se relaciona P(Y = 1 | X)
#    con un problema de clasificación
#    en Machine Learning?
# Respuesta:
# Un clasificador binario (por ejemplo, regresión logística) estima la
# probabilidad de que la clase sea 1 dadas las características X del
# cliente. Luego se aplica un umbral (por ejemplo, 0.5) para decidir
# la clase: si P(Y = 1 | X) >= 0.5, se predice "compra". Es la
# generalización de P(compra | email) a muchas características a la vez.