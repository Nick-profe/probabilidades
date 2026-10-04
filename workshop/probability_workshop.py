import numpy as np

# TALLER DE PROBABILIDAD APLICADA A MACHINE LEARNING
# ============================================================

# Nombre: Jaiber
# Apellido 1: Obando
# Apellido 2: Lopez
# Rama: obando_lopez


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
Y= channel
print("Primeros valores de X:")
print(X[:10])


# 3. PROBABILIDAD EMPÍRICA
# ------------------------------------------------------------

def empirical_probability(event):
    """
    Calcula la proporción de observaciones
    para las cuales un evento es verdadero.
    """

    #TODO : completar
    purchase_array = np.array(event)
    p_purchase=np.mean(purchase_array==1)
    p_no_purchase=np.mean(purchase_array==0)
      
    return p_purchase, p_no_purchase


# TODO:
# calcular P(compra)
# TODO:
# calcular P(no compra)
#p_purchase = None
#p_no_purchase = None

p_purchase, p_no_purchase = empirical_probability(X)

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
    #unique_values = None
    unique_values, counts = np.unique(values, return_counts=True)
    #probabilities = None
    probabilities = counts / len(values)

    return unique_values, probabilities


#pmf_values = None
#pmf_probabilities = None

pmf_values, pmf_probabilities = empirical_pmf(X)

# TODO:
# utilizar empirical_pmf(X)

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

#pmf_sum = None
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

    # TODO: completar
    expected_purchase = np.sum(values * probabilities)
    return expected_purchase


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
    values = np.asarray(values, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)

    expected_val = np.sum(values * probabilities)
    purchase_variance=np.sum(probabilities * (values - expected_val) ** 2)
    # TODO: completar

    return purchase_variance


purchase_variance = variance(pmf_values, pmf_probabilities)
comprobacion = p_purchase * (1 - p_purchase)


print(
    "Varianza de X:",
    purchase_variance
)
print("Comprobación de la varianza:", comprobacion)

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
    # TODO: completar
    event = np.asarray(event)
    condition = np.asarray(condition)

    p_event_condition= np.mean(event[condition]==1)
    
    
    return p_event_condition


# channel == 0 representa email
email_condition = channel == 0

# channel == 1 representa social
social_condition = channel == 1


# TODO:
# calcular P(compra | email)

p_purchase_email = conditional_probability(X, email_condition)


# TODO:
# calcular P(compra | social)

p_purchase_social = conditional_probability(X, social_condition)


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
# Puesto que una variable aleatoria Bernoulli, es una variable discreta que solo puede tomar 2 valores posibles, siendo la forma más simple de
# modelar un experimento binario, purchased puede considerarse una variable de este tipo, puesto que solo puede tomar 2 valores posibles, 1 si el cliente realizó una compra 
# y 0 si no realizó una compra.


# 2. ¿Qué representa P(compra) dentro del problema?
# Respuesta:
# P(compra) representa la probabilidad de que un cliente realice una compra, es decir, la proporción de clientes en la población que han realizado una compra.


# 3. ¿Por qué las probabilidades de la PMF
#    deben sumar 1?
# Respuesta:
# Las probabilidades de la PMF deben sumar 1 porque representan todos los posibles resultados de un experimento, y la probabilidad total debe ser 1.

# 4. ¿Qué representa E[X] cuando X es
#    la variable purchased?
# Respuesta:
# E[X] representa el valor esperado de la variable purchased, es decir, la probabilidad de que un cliente realice una compra.

# 5. ¿Qué representa la varianza de X?
# Respuesta:
# La varianza de X representa la dispersión de los valores de X alrededor de su valor esperado.


# 6. ¿Por qué la probabilidad observada en una
#    muestra puede ser diferente de la probabilidad
#    observada en toda la población?
# Respuesta:
# Porque una muestra solo contiene algunas observaciones de la población, y
# cada muestra aleatoria da una proporción distinta (variabilidad muestral).
# En promedio, la diferencia (entre el calculo en una muestra y en toda la población) 
# es mayor con muestras pequeñas y disminuye al aumentar el tamaño, pero solo se garantiza que
# desaparezca cuando muestra = toda la población. Por eso, la probabilidad
# observada en una muestra puede diferir de la de toda la población.

# 7. ¿Cuál es la diferencia entre
#    P(compra) y P(compra | email)?
# Respuesta:
# P(compra) es la probabilidad de que un cliente realice una compra sin tener en cuenta ninguna condición, 
# mientras que P(compra | email) es la probabilidad de que un cliente realice una compra debido a  que se le ha contactado a través  de email. 
# La diferencia radica en que la segunda probabilidad está condicionada a un evento específico (canal de comunicación=email).


# 8. ¿Cómo se relaciona P(Y = 1 | X)
#    con un problema de clasificación
#    en Machine Learning?
# Respuesta: 
# # P(Y = 1 | X) representa la probabilidad de que un cliente realice una compra (Y = 1) dado un conjunto de características del cliente (X). 
# En un problema de clasificación en Machine Learning, el objetivo es construir un modelo que pueda predecir esta probabilidad para nuevos clientes basándose en sus características, 
# permitiendo así tomar decisiones informadas sobre estrategias de marketing y ventas.