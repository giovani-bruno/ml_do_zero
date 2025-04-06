from typing import List
from utils import Vector, dot, vector_sum
import math

def logistic(x: float) -> float:
    return 1.0 / (1 + math.exp(-x))

def logistic_prime(x: float) -> float:
    y = logistic(x)
    return y * (1 - y)

def _negative_log_likelihood(x: Vector, y: float, beta: Vector) -> float:
    """Calcula a log-verossimilhança negativa para um ponto de dado"""
    if y == 1:
        return -math.log(logistic(dot(x, beta)))
    else:
        return -math.log(1 - logistic(dot(x, beta)))
    
def negative_log_likelihood(xs: List[Vector],
                            ys: List[float],
                            beta: Vector) -> float:
    """Soma da log-verossimilhança negativa para todos os dados"""
    return sum(_negative_log_likelihood(x, y, beta)
               for x, y in zip(xs, ys))

def _negative_log_partial_j(x: Vector, y: float, beta: Vector, j: int) -> float:
    """Derivada parcial em relação ao j-ésimo coeficiente, para um ponto de dado"""
    return -(y - logistic(dot(x, beta))) * x[j]

def _negative_log_gradient(x: Vector, y: float, beta: Vector) -> Vector:
    """Gradiente da log-verossimilhança negativa para um ponto de dado"""
    return [_negative_log_partial_j(x, y, beta, j)
            for j in range(len(beta))]

def negative_log_gradient(xs: List[Vector],
                          ys: List[float],
                          beta: Vector) -> Vector:
    """Gradiente total da log-verossimilhança negativa para todos os dados"""
    return vector_sum([_negative_log_gradient(x, y, beta)
                       for x, y in zip(xs, ys)])