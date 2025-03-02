from utils import Vector, correlation, standard_deviation, mean, de_mean
from typing import Tuple

def predict(alpha: float, beta: float, x_i: float) -> float:
    """Retorna a previsão do valor de y dada por beta * x_i + alpha"""
    return beta * x_i + alpha

def error(alpha: float, beta: float, x_i: float, y_i: float) -> float:
    """Retorna o erro ao prever beta * x_i + alpha, quando o valor real é y_i"""
    return predict(alpha, beta, x_i) - y_i

def sum_of_sqerrors(alpha: float, beta: float, x: Vector, y: Vector) -> float:
    """Calcula a soma dos erros quadráticos"""
    return sum(error(alpha, beta, x_i, y_i) ** 2
               for x_i, y_i in zip(x, y))

def least_squares_fit(x: Vector, y: Vector) -> Tuple[float, float]:
    """
    Dado dois vetores x e y,
    encontra os valores de alpha e beta pelo método dos mínimos quadrados
    """
    beta = correlation(x, y) * standard_deviation(y) / standard_deviation(x)
    alpha = mean(y) - beta * mean(x)
    return alpha, beta

def total_sum_of_squares(y: Vector) -> float:
    """Calcula a soma total dos quadrados das variações dos valores de y em relação à média"""
    return sum(v ** 2 for v in de_mean(y))

def r_squared(alpha: float, beta: float, x: Vector, y: Vector) -> float:
    """
    Retorna a fração da variação de y explicada pelo modelo, que é igual a:
    1 - a fração da variação de y não explicada pelo modelo
    """
    return 1.0 - (sum_of_sqerrors(alpha, beta, x, y) /
                  total_sum_of_squares(y))
