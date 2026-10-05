import numpy as np

# TALLER DE PROBABILIDAD APLICADA A MACHINE LEARNING
# ============================================================

# Nombre: Andres
# Apellido 1: Pinilla
# Apellido 2: Victoria
# Rama: pinilla_victoria


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
# Respuesta: Porque cada cliente solo puede hacer una de dos cosas: comprar (1)
# o no comprar (0). Es un experimento con dos resultados posibles y una
# probabilidad p de que salga "compra", que es justo lo que describe una
# Bernoulli.


# 2. ¿Qué representa P(compra) dentro del problema?
# Respuesta: Es qué tan probable es que un cliente cualquiera termine comprando
# después de la campaña. En la práctica es el porcentaje de clientes que
# compraron, o sea la tasa de conversión.


# 3. ¿Por qué las probabilidades de la PMF
#    deben sumar 1?
# Respuesta: Porque la PMF tiene que cubrir todo lo que puede pasar. Aquí un
# cliente compra o no compra, no hay una tercera opción, así que las dos
# probabilidades juntas deben dar 1 (100%). Si no suman 1, algo está mal en
# el cálculo o en los datos.


# 4. ¿Qué representa E[X] cuando X es
#    la variable purchased?
# Respuesta: Como X solo vale 0 o 1, al calcular E[X] lo único que queda es
# P(X=1), así que E[X] es igual a la probabilidad de compra. Se puede leer
# como la fracción promedio de clientes que compra.


# 5. ¿Qué representa la varianza de X?
# Respuesta: Indica qué tan dispersos o impredecibles son los resultados. En
# una Bernoulli es p(1-p): si p está cerca de 0.5 hay mucha incertidumbre
# porque comprar y no comprar son casi igual de probables, y si p está cerca
# de 0 o de 1 el resultado es mucho más predecible y la varianza baja.


# 6. ¿Por qué la probabilidad observada en una
#    muestra puede ser diferente de la probabilidad
#    observada en toda la población?
# Respuesta: Porque una muestra es solo un pedacito de la población y depende
# de a quiénes les tocó salir. Con solo 10 clientes, por azar pueden salir más
# o menos compradores de los que realmente hay. Entre más grande sea la
# muestra, más se parece a la población (ley de los grandes números).


# 7. ¿Cuál es la diferencia entre
#    P(compra) y P(compra | email)?
# Respuesta: P(compra) mira a todos los clientes sin importar el canal,
# mientras que P(compra | email) solo mira a los que llegaron por email. Si
# los dos valores son distintos, quiere decir que el canal influye en la
# compra y que las dos variables no son independientes.


# 8. ¿Cómo se relaciona P(Y = 1 | X)
#    con un problema de clasificación
#    en Machine Learning?
# Respuesta: Es justo lo que un modelo de clasificación intenta estimar: la
# probabilidad de que un cliente compre (Y=1) dadas sus características X,
# como el canal o el valor de la orden. Modelos como la regresión logística
# lo calculan y después se decide con un umbral, normalmente 0.5: si la
# probabilidad es mayor, se predice que compra.