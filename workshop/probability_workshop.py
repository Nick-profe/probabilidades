import numpy as np

# TALLER DE PROBABILIDAD APLICADA A MACHINE LEARNING
# ============================================================

# Nombre:Edwin Andres
# Apellido 1:Guerrero
# Apellido 2:Diaz
# Rama: guerrero_diaz


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

    unique_values, counts = np.unique(
        values,
        return_counts=True
    )
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

    # Var(X) = E[(X - mu)^2], donde mu = E[X]
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


# 8. MUESTREO
# ------------------------------------------------------------

# Para que los resultados sean reproducibles
np.random.seed(42)


# seleccionar una muestra aleatoria
# de 10 observaciones de X
# sin reemplazo.

sample = np.random.choice(
    X,
    size=10,
    replace=False
)


# calcular la probabilidad de compra
# dentro de la muestra.

sample_probability = empirical_probability(
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

    return np.sum(event & condition) / np.sum(condition)


# channel == 0 representa email
email_condition = channel == 0

# channel == 1 representa social
social_condition = channel == 1


# calcular P(compra | email)

p_purchase_email = conditional_probability(
    X == 1,
    email_condition
)


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
# Respuesta:
# Porque solo hay dos opciones: el cliente compra (1)
# o no compra (0). Eso es justo lo que describe una
# variable Bernoulli.


# 2. ¿Qué representa P(compra) dentro del problema?
# Respuesta:
# Es qué tanto compran los clientes de la campaña.
# Compraron 23 de 40, o sea, un poco más de la
# mitad (57.5 %).


# 3. ¿Por qué las probabilidades de la PMF
#    deben sumar 1?
# Respuesta:
# Porque cada cliente compra o no compra; no hay
# otra opción. Si juntas los dos casos, tienes a
# todos los clientes, o sea, el 100 %.


# 4. ¿Qué representa E[X] cuando X es
#    la variable purchased?
# Respuesta:
# Es el promedio de X. Como X solo vale 0 o 1, ese
# promedio termina siendo la misma probabilidad de
# compra: 0.575.


# 5. ¿Qué representa la varianza de X?
# Respuesta:
# Dice qué tan difícil es adivinar si un cliente va
# a comprar. Aquí sale alta porque más o menos la
# mitad compra y la otra mitad no.


# 6. ¿Por qué la probabilidad observada en una
#    muestra puede ser diferente de la probabilidad
#    observada en toda la población?
# Respuesta:
# Porque 10 clientes son pocos y, por azar, pueden
# salir más o menos compradores. En la muestra salió
# 0.4 y con todos los clientes 0.575. Entre más
# grande es la muestra, más se parecen.


# 7. ¿Cuál es la diferencia entre
#    P(compra) y P(compra | email)?
# Respuesta:
# P(compra) mira a todos los clientes (0.575).
# P(compra | email) solo mira a los que contactamos
# por email (0.65). Por email compran más, así que
# el canal sí importa.


# 8. ¿Cómo se relaciona P(Y = 1 | X)
#    con un problema de clasificación
#    en Machine Learning?
# Respuesta:
# Eso es lo que hace un modelo de clasificación: mira
# los datos de un cliente y calcula la probabilidad
# de que compre. Si sale alta (por ejemplo, 0.5 o
# más), predice que sí va a comprar.