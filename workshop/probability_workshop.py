import numpy as np

# TALLER DE PROBABILIDAD APLICADA A MACHINE LEARNING
# ============================================================

# Nombre: Juan Esteban
# Apellido 1: Ruiz
# Apellido 2: Calero
# Rama: ruiz_calero


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
 
    # event es un arreglo booleano: True donde el evento ocurre.
    # La media de un arreglo booleano equivale a
    # (casos favorables) / (casos totales).
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
 
    # np.unique devuelve los valores distintos y, con
    # return_counts, la frecuencia absoluta de cada uno.
    unique_values, counts = np.unique(values, return_counts=True)
 
    # Frecuencia relativa = frecuencia absoluta / total.
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
 
    # E[X] = sumatoria de x_i * P(x_i)
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
 
    # Var(X) = E[(X - E[X])^2] = sumatoria de P(x_i) * (x_i - mu)^2
    mu = expected_value(values, probabilities)
 
    return np.sum(probabilities * (values - mu) ** 2)
 
 
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
 
    # P(A | B) = P(A interseccion B) / P(B)
    # Se cuentan los casos donde ocurren ambos y se divide
    # entre los casos donde se cumple la condicion.
    return np.sum(event & condition) / np.sum(condition)
 
 
# channel == 0 representa email
email_condition = channel == 0
 
# channel == 1 representa social
social_condition = channel == 1
 
 
p_purchase_email = conditional_probability(X == 1, email_condition)
 
 
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
# Respuesta: Porque solo toma dos valores posibles, 0 (no compró) y
# 1 (compró), y cada cliente constituye un ensayo independiente con una
# misma probabilidad p de éxito. Esa es exactamente la definición de una
# variable Bernoulli.


# 2. ¿Qué representa P(compra) dentro del problema?
# Respuesta: Es la proporción de clientes que efectivamente compraron,
# 0.575. Corresponde a la estimación empírica del parámetro p de la
# distribución Bernoulli: la probabilidad de que un cliente cualquiera de
# la campaña realice una compra.


# 3. ¿Por qué las probabilidades de la PMF
#    deben sumar 1?
# Respuesta:


# 4. ¿Qué representa E[X] cuando X es
#    la variable purchased?
# Respuesta:


# 5. ¿Qué representa la varianza de X?
# Respuesta:


# 6. ¿Por qué la probabilidad observada en una
#    muestra puede ser diferente de la probabilidad
#    observada en toda la población?
# Respuesta:


# 7. ¿Cuál es la diferencia entre
#    P(compra) y P(compra | email)?
# Respuesta:


# 8. ¿Cómo se relaciona P(Y = 1 | X)
#    con un problema de clasificación
#    en Machine Learning?
# Respuesta: