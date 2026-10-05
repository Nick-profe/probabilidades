import numpy as np

# TALLER DE PROBABILIDAD APLICADA A MACHINE LEARNING
# ============================================================

# Nombre: Leonardo
# Apellido 1: Monsalve
# Apellido 2: Gomez
# Rama: Monsalve_Gomez


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

    mean = expected_value(values, probabilities)

    return np.sum((values - mean) ** 2 * probabilities)


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

    return np.sum(event & condition) / np.sum(condition)


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

# 1. ¿Por qué purchased puede considerarse
#    una variable aleatoria Bernoulli?
# Respuesta:

"""
Porque solo toma dos valores, 1 (compra) y 0 (no compra),
y cada cliente es un ensayo con una probabilidad de éxito p.

"""

# 2. ¿Qué representa P(compra) dentro del problema?
# Respuesta:
"""
P(compra) representa la probabilidad de que un cliente de la campaña realice una compra. 
Se calcula como la proporción de clientes que compraron sobre el total de clientes: 23 compraron de 40, es decir 0.575 (57.5%). 
En términos de la variable aleatoria Bernoulli, es la probabilidad# de éxito p, o sea P(X = 1).
Dentro del problema nos dice qué tan efectiva fue la campaña en general, sin distinguir el canal por el que se contactó a cada cliente.

"""


# 3. ¿Por qué las probabilidades de la PMF
#    deben sumar 1?
# Respuesta:

"""
 Las probabilidades de una PMF deben sumar 1 porque la PMF recoge todos los valores posibles de la variable aleatoria, y en cada
 observación alguno de ellos tiene que ocurrir. Aquí los valores posibles son 0 y 1, con probabilidades 0.425 y 0.575, y su suma es 1.0. 
 Si la suma fuera distinta de 1, significaría que falta algún resultado posible o que hay un error en el cálculo de las probabilidades.

"""

# 4. ¿Qué representa E[X] cuando X es
#    la variable purchased?
# Respuesta:

"""
Respuesta: E[X] es el valor esperado, es decir, el promedio que se obtendría de X si se repitiera el experimento muchas veces. 
Se calcula como la suma de cada valor por su probabilidad: 0(0.425) + 1(0.575) = 0.575.
Cuando X es purchased, que solo vale 0 o 1, el valor esperado coincide con la probabilidad de compra p. 
Por eso E[X] = 0.575 se interpreta comoque, en promedio, 57.5% de los clientes contactados compra.

"""

# 5. ¿Qué representa la varianza de X?
# Respuesta:

"""
La varianza mide qué tanto se alejan los valores de X de su promedio, es decir, qué tan dispersos o impredecibles son los resultados.
Para una Bernoulli, Var(X) = p(1 - p) = 0.575 x 0.425 = 0.244375. Este valor es cercano al máximo posible (0.25), que ocurre cuando p = 0.5, 
porque la mitad de los clientes compra y la otra mitad no, y por eso es difícil predecir qué hará un cliente individual. 
Si p estuviera cerca de 0 o de 1, la varianza sería pequeña y el resultado sería más predecible.

"""

# 6. ¿Por qué la probabilidad observada en una
#    muestra puede ser diferente de la probabilidad
#    observada en toda la población?
# Respuesta:

"""
La probabilidad de la muestra puede ser distinta a la de la población porque la muestra es solo un subconjunto pequeño y aleatorio
de los datos, y por azar puede contener más o menos compradores de los que le corresponderían. 
En este caso, la población tiene 0.575 de probabilidad de compra, pero la muestra de 10 clientes dio 0.4 (4 compras).
Esta diferencia se llama error de muestreo. Mientras más grande sea la muestra, más se acerca su probabilidad a la de la población.

"""

# 7. ¿Cuál es la diferencia entre
#    P(compra) y P(compra | email)?
# Respuesta:


# 8. ¿Cómo se relaciona P(Y = 1 | X)
#    con un problema de clasificación
#    en Machine Learning?
# Respuesta:python workshop/probability_workshop.py