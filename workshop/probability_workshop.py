import numpy as np

# TALLER DE PROBABILIDAD APLICADA A MACHINE LEARNING
# ============================================================

# Nombre: David Alberto
# Apellido 1: Vergara
# Apellido 2: Tabares 
# Rama: vergara_tabares


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

    mean = expected_value(values, probabilities)

    return np.sum(probabilities * (values - mean) ** 2)


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
    conditioned_event = event[condition]
    return np.mean(conditioned_event)

email_condition = channel == 0
social_condition = channel == 1

p_purchase_email = conditional_probability(
    X == 1,
    email_condition
)

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
# Respuesta: Porque purschaed representa un experimento con dos 
# resultados posibles.
#  1 si el cliente realiza una compra y 0 si no la realiza.
# Una variable Bernoulli precisamente representa un experimento con dos posibles resultados.


# 2. ¿Qué representa P(compra) dentro del problema?
# Respuesta: Representa la proporción de clientes del dataset de la población observada
#  que realizaron una compra


# 3. ¿Por qué las probabilidades de la PMF
#    deben sumar 1?
# Respuesta: Porque la PMF cubre todos los valores posibles de la
# variable y cada observación toma exactamente uno de ellos. Algo tiene que ocurrir,
# así que entre todos los valores se reparte el
# 100% de la probabilidad, como en este resultado que se obtuvo 0.425 + 0.575 = 1


# 4. ¿Qué representa E[X] cuando X es
#    la variable purchased?
# Respuesta: Representa el valor promedio esperado de la variable purchased. 
# Como X, solamente puede ser 0 o 1, también puede interpretarse como la probabilidad de compra


# 5. ¿Qué representa la varianza de X?
# Respuesta: Representa la variabilidad de la variable purchased. En este caso, indica
# cuán dispersas están las compras en relación con la media.


# 6. ¿Por qué la probabilidad observada en una
#    muestra puede ser diferente de la probabilidad
#    observada en toda la población?
# Respuesta: Porque una muestra contiene solamente una parte de las observaciones y
#  especialmente cuando es pequeña, puede no representar exactamente las proporciones de toda la población.


# 7. ¿Cuál es la diferencia entre
#    P(compra) y P(compra | email)?
# Respuesta: Para este caso p(compra) respresenta únicamente la probabilidad de compra en toda la población, 
# mientras que P(compra | email) representa la probabilidad de compra con la condición de que el cliente haya sido contactado por email. 

# 8. ¿Cómo se relaciona P(Y = 1 | X)
#    con un problema de clasificación
#    en Machine Learning?
# Respuesta: Representa la probabilidad estimada de que una observación pertenezca a la clase 1 dadas sus características X. 
# En clasificación binaria, esa probabilidad puede utilizarse para determinar la clase predicha mediante un umbral