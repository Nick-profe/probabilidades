# Taller de Probabilidad aplicada a Machine Learning

Este repositorio contiene el taller práctico de Probabilidad del curso **Herramientas Matemáticas y Computacionales para la IA**.

El objetivo del taller es aplicar conceptos fundamentales de Probabilidad utilizando Python y NumPy.

## Objetivos

Durante el taller se trabajarán los siguientes conceptos:

- Variable aleatoria;
- Probabilidad empírica;
- PMF (Probability Mass Function — Función de Masa de Probabilidad);
- Valor esperado;
- Varianza;
- Muestreo;
- Probabilidad condicional;
- Conexión con problemas de clasificación en Machine Learning.

## Caso de estudio

Se analizará información de una campaña comercial realizada sobre un conjunto de clientes.

El dataset contiene las siguientes variables:

- `customer_id`: identificador del cliente;
- `channel`: canal mediante el cual fue contactado;
- `purchased`: indica si el cliente realizó una compra;
- `order_value`: valor de la compra.

La variable `channel` está codificada como:

- `0`: email;
- `1`: social.

La variable `purchased` está codificada como:

- `0`: no compró;
- `1`: compró.

La variable `purchased` será interpretada como una variable aleatoria Bernoulli.

## Estructura del repositorio

```text
probabilidades/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── customer_campaign.csv
└── workshop/
    └── probability_workshop.py
```

## Preparación del entorno

Clone el repositorio:

```bash
git clone https://github.com/Nick-profe/probabilidades.git
```

Ingrese al proyecto:

```bash
cd probabilidades
```

Instale las dependencias:

```bash
pip install -r requirements.txt
```

## Flujo de trabajo con Git

Cada estudiante debe desarrollar el taller en una rama individual.

Primero actualice la información del repositorio:

```bash
git fetch origin
```

Cambie a la rama base del taller:

```bash
git switch taller_1
```

Cree su rama personal:

```bash
git switch -c apellido1_apellido2
```

Ejemplo:

```bash
git switch -c lopez_gomez
```

Verifique la rama activa:

```bash
git branch
```

El símbolo `*` debe aparecer junto a su rama personal.

## Desarrollo del taller

El archivo que debe completar es:

```text
workshop/probability_workshop.py
```

Los bloques que deben desarrollarse están identificados mediante comentarios `TODO`.

Ejecute el taller desde la raíz del repositorio:

```bash
python workshop/probability_workshop.py
```

## Guardar avances

Durante el taller puede guardar sus avances utilizando:

```bash
git status
git add workshop/probability_workshop.py
git commit -m "Avance taller de probabilidades"
```

## Entrega

Cuando termine el taller, publique su rama en GitHub:

```bash
git push -u origin apellido1_apellido2
```

No debe hacer merge con `main` ni con `taller_1`.

La entrega será revisada directamente desde la rama individual de cada estudiante.

## Restricciones

Para resolver el taller utilice principalmente:

```python
numpy
```

No utilice librerías de Machine Learning para resolver los cálculos solicitados.

El objetivo es implementar y comprender directamente los conceptos matemáticos y probabilísticos.
