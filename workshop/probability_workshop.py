import numpy as np

# TALLER DE PROBABILIDAD APLICADA A MACHINE LEARNING
# ============================================================

# Nombre: Juana Gabriela
# Apellido 1: Lopez
# Apellido 2: Trejos
# Rama:lopez_trejos


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

    # TODO: completar
    return np.mean(event)


# TODO:
# calcular P(compra)

p_purchase = empirical_probability(X == 1) #propabilidad empírica de compra (purchased = 1)


# TODO:
# calcular P(no compra)

p_no_purchase = empirical_probability(X == 0) #propabilidad empírica de no compra (purchased = 0)


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

    # TODO: completar

    unique_values, counts = np.unique(values, return_counts=True)
    probabilities = counts / len(values)

    return unique_values, probabilities


pmf_values = None
pmf_probabilities = None


# TODO:
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
    # E[X] = suma de X * P(X=x)

    # TODO: completar
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

    # TODO: completar
    #mean = expected_value(values, probabilities) #opción 2 basandome en la ecuación de la varianza Var(X) = E[(X - E[X])^2]

    return np.var(X) #opción 1 directamente con la función de numpy para varianza
    #return np.sum(probabilities * (values - mean) ** 2) #opción 2


purchase_variance = variance(pmf_values, pmf_probabilities) 


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
    return np.mean(event[condition])  #opción 1
    #return np.sum(event & condition) / np.sum(condition) #opción 2 con respecto a la formula de probabilidad condicional P(A|B) = P(A∩B)/P(B)


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
# Respuesta: porque purchase solo puede tomar dos valores, sinedo 1 cuando el cliente realiza una compra y  cuando no se realiza compra. Una variable Bernoulli representa
# exactamente eso, un experimento con dos posibles resultados.


# 2. ¿Qué representa P(compra) dentro del problema?
# Respuesta: en este caso, P(compra) representa la probabilidad de que un cliente (de toda la población/conjunto de datos) realice una compra


# 3. ¿Por qué las probabilidades de la PMF
#    deben sumar 1?
# Respuesta: en este caso estamos trabajando con dos escenarios posibles 0 y 1, por tanto la suma de las probabilidades de todos los posibles resultados debe ser igual a 1 (100%)
# ya que uno de los dos resultados debe ocurrir.


# 4. ¿Qué representa E[X] cuando X es
#    la variable purchased?
# Respuesta: este representa el valor esperado de la variable aleatoria, en este caso purchased solo toma dos valores (0 y 1), por lo que el valor esperado es equivalente a la probabilidad de compara.
# En este caso en aprticular, hace referencia esa proporción de clientes que se espera que compren


# 5. ¿Qué representa la varianza de X?
# Respuesta: esta representa la dispersión de los datos para la variable aleatoria purchased, es decir, nos indica cuánto varían los resultados entre compra y no compra.


# 6. ¿Por qué la probabilidad observada en una
#    muestra puede ser diferente de la probabilidad
#    observada en toda la población?
# Respuesta: debido a que, al tomar una muestra estamos utilizando solo una parte de las observaciones totales, y en esta caso, su valores se toman al azar, por ende la probabilidad 
#observada en una muestra, por ejemplo de la proporción de clientes que compran, no será exactamente igual a la proporción de clientes que compran en toda la población


# 7. ¿Cuál es la diferencia entre
#    P(compra) y P(compra | email)?
# Respuesta: P(compra) en este caso hace referencia a la probabilidad de que un cliente compre considerando los dos canales, en cambio P(compra | email) hace referencia
# a la probabilidad de que un cliente compre por medio de email


# 8. ¿Cómo se relaciona P(Y = 1 | X)
#    con un problema de clasificación
#    en Machine Learning?
# Respuesta: se relaciona en que P(Y = 1 | X) representa la probabilidad de que la variable obbjetivo Y, tomer un valor (en este caso 1 = compra), dada las características X de un cliente
# y en ML un modelo de clasificación binaria por ejemplo, aprende a estimar esas probabilidades con base a esas X características. 