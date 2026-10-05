import numpy as np

# TALLER DE PROBABILIDAD APLICADA A MACHINE LEARNING
# ============================================================

# Nombre: Diego
# Apellido 1: Berrio
# Apellido 2: Lasso
# Rama: berrio_lasso


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


p_purchase = empirical_probability(purchased == 1)

p_no_purchase = empirical_probability(purchased == 0)

print("P(compra):", p_purchase)
print("P(no compra):", p_no_purchase)


# 4. PMF
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

print("Valores de la variable:", pmf_values)
print("Probabilidades:", pmf_probabilities)


# 5. VERIFICACIÓN DE LA PMF
# ------------------------------------------------------------

pmf_sum = np.sum(pmf_probabilities)

print("Suma de probabilidades:", pmf_sum)


# 6. VALOR ESPERADO
# ------------------------------------------------------------

def expected_value(values, probabilities):
    """
    Calcula E[X].
    """
    return np.sum(values * probabilities)


expected_purchase = expected_value(pmf_values, pmf_probabilities)

print("Valor esperado de X:", expected_purchase)


# 7. VARIANZA
# ------------------------------------------------------------

def variance(values, probabilities):
    """
    Calcula Var(X).
    """
    mean = expected_value(values, probabilities)
    return np.sum(probabilities * (values - mean) ** 2)


purchase_variance = variance(pmf_values, pmf_probabilities)

print("Varianza de X:", purchase_variance)


# 8. MUESTREO
# ------------------------------------------------------------

np.random.seed(42)

sample = np.random.choice(X, size=10, replace=False)

sample_probability = empirical_probability(sample == 1)

print("Probabilidad de compra en la población:", p_purchase)
print("Probabilidad de compra en la muestra:", sample_probability)


# 9. PROBABILIDAD CONDICIONAL
# ------------------------------------------------------------

def conditional_probability(event, condition):
    """
    Calcula P(evento | condición).
    """
    return np.mean(event[condition])


email_condition = channel == 0
social_condition = channel == 1

p_purchase_email = conditional_probability(purchased == 1, email_condition)

p_purchase_social = conditional_probability(purchased == 1, social_condition)

print("P(compra | email):", p_purchase_email)
print("P(compra | social):", p_purchase_social)


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
# Respuesta: Porque solo toma dos valores posibles (0 o 1), representando
# un experimento de "éxito o fracaso" — el cliente compró o no compró —
# que es exactamente la definición de una variable Bernoulli.

# 2. ¿Qué representa P(compra) dentro del problema?
# Respuesta: Es la proporción de clientes de toda la campaña que terminaron
# comprando, es decir, la probabilidad estimada de que un cliente cualquiera
# realice una compra.

# 3. ¿Por qué las probabilidades de la PMF
#    deben sumar 1?
# Respuesta: Porque la PMF cubre todos los valores posibles que puede tomar
# la variable, y la certeza de que ocurra alguno de ellos es del 100%, es
# decir, probabilidad total 1.

# 4. ¿Qué representa E[X] cuando X es
#    la variable purchased?
# Respuesta: Es el promedio esperado de la variable, que para una Bernoulli
# coincide exactamente con P(compra) — el valor esperado "resume" la
# tendencia general de los clientes a comprar.

# 5. ¿Qué representa la varianza de X?
# Respuesta: Mide qué tan dispersos o inciertos son los resultados; en una
# Bernoulli, la varianza es máxima cuando p está cerca de 0.5 (mayor
# incertidumbre) y menor cuando p está cerca de 0 o 1.

# 6. ¿Por qué la probabilidad observada en una
#    muestra puede ser diferente de la probabilidad
#    observada en toda la población?
# Respuesta: Porque una muestra es solo un subconjunto aleatorio de los
# datos, y por azar puede no reflejar exactamente la misma proporción que
# tiene la población completa — a eso se le llama variabilidad muestral.

# 7. ¿Cuál es la diferencia entre
#    P(compra) y P(compra | email)?
# Respuesta: P(compra) es la probabilidad general sin tener en cuenta el
# canal, mientras que P(compra | email) es la probabilidad de compra
# considerando únicamente a los clientes que fueron contactados por email,
# es decir, condicionada a esa información adicional.

# 8. ¿Cómo se relaciona P(Y = 1 | X)
#    con un problema de clasificación
#    en Machine Learning?
# Respuesta: Es exactamente lo que modelos como la regresión logística
# intentan estimar: la probabilidad de que ocurra la clase positiva (Y=1)
# dado un conjunto de características X del cliente, para luego usar esa
# probabilidad y decidir una predicción.