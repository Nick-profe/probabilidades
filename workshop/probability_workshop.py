from pathlib import Path
import numpy as np

# TALLER DE PROBABILIDAD APLICADA A MACHINE LEARNING
# ============================================================

# Nombre: Leonel Mauricio
# Apellido 1: Reyes
# Apellido 2: Rodriguez
# Rama: reyes_rodriguez


# 1. CARGA DE DATOS
# ------------------------------------------------------------

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "customer_campaign.csv"

data = np.loadtxt(
    DATA_PATH,
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

    # HECHO: completar
    res_ep = np.mean(event)
    return res_ep


# HECHO:
# calcular P(compra)

p_purchase = empirical_probability(X==1)


# HECHO:
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

    # HECHO: completar

    unique_values, counts = np.unique(values, return_counts=True)
    probabilities = counts / len(values)

    return unique_values, probabilities

#pmf_values = None
#pmf_probabilities = None


# HECHO:
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

    # HECHO: completar
    res_ev = np.sum(values * probabilities)
    return res_ev


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
    # HECHO: completar
    mu = expected_value(values, probabilities)
    var = np.sum((values - mu)**2 * probabilities)

    return var


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

sample = np.random.choice(X, size=10, replace=False)
print("Muestra aleatoria de 10 observaciones:", sample)

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
    # P(A|B) = P(A ∩ B) / P(B)
    p_joint = empirical_probability(event & condition)
    p_condition = empirical_probability(condition)

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
# Respuesta: Porque es un único experimento (un cliente) con solo dos
# resultados posibles, compra (1) o no compra (0), y P(X=1) = p.


# 2. ¿Qué representa P(compra) dentro del problema?
# Respuesta: La proporción de clientes de la campaña que compraron
# (23 de 40 = 0.575), es decir, la probabilidad estimada de que un
# cliente cualquiera realice una compra.


# 3. ¿Por qué las probabilidades de la PMF
#    deben sumar 1?
# Respuesta: Porque la PMF cubre todos los valores posibles de X, y
# alguno de ellos tiene que ocurrir. Aquí 0.425 + 0.575 = 1.


# 4. ¿Qué representa E[X] cuando X es
#    la variable purchased?
# Respuesta: El promedio teórico de largo plazo de X. Como X solo toma
# 0 y 1, E[X] coincide con P(compra) = 0.575.


# 5. ¿Qué representa la varianza de X?
# Respuesta: Qué tanto se dispersan los valores de X alrededor de su
# media, es decir, la incertidumbre sobre si un cliente comprará.
# Para una Bernoulli es p(1-p) = 0.575 * 0.425 = 0.244375.


# 6. ¿Por qué la probabilidad observada en una
#    muestra puede ser diferente de la probabilidad
#    observada en toda la población?
# Respuesta: Porque la muestra contiene solo algunas observaciones
# elegidas al azar, y por azar puede tener más o menos compradores que
# la población. Con muestras más grandes la diferencia tiende a
# disminuir (Ley de los Grandes Números).


# 7. ¿Cuál es la diferencia entre
#    P(compra) y P(compra | email)?
# Respuesta: P(compra) considera a todos los clientes. P(compra | email)
# considera solo a los clientes que llegaron por email, es decir, la
# probabilidad de compra una vez que se sabe el canal.


# 8. ¿Cómo se relaciona P(Y = 1 | X)
#    con un problema de clasificación
#    en Machine Learning?
# Respuesta: Un clasificador binario estima la probabilidad de que
# Y = 1 (compra) dadas las características X del cliente. Si esa
# probabilidad supera un umbral (por ejemplo 0.5), predice compra.