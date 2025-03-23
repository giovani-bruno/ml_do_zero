from utils import Vector, vector_mean, gradient_step, de_mean
from typing import List
import random
import tqdm

def predict(x: Vector, beta: Vector) -> float:
    """Assume que o primeiro elemento de x é 1"""
    return sum(x_i * b for x_i, b in zip(x, beta))

def error(x: Vector, y: float, beta: Vector) -> float:
    """Calcula o erro entre a previsão e o valor real."""
    return predict(x, beta) - y

def squared_error(x: Vector, y: float, beta: Vector) -> float:
    """Retorna o erro quadrático."""
    return error(x, y, beta) ** 2

def sqerror_gradient(x: Vector, y: float, beta: Vector) -> Vector:
    """Calcula o gradiente do erro quadrático em relação a beta."""
    err = error(x, y, beta)
    return [2 * err * x_i for x_i in x]


def least_squares_fit(xs: List[Vector],
    ys: List[float],
    learning_rate: float = 0.001,
    num_steps: int = 1000,
    batch_size: int = 1) -> Vector:
    """
    Encontra os coeficientes beta que minimizam a soma dos erros quadráticos,
    pressupondo que o modelo y = dot(x, beta).
    """
    # Inicializa com uma estimativa aleatória
    guess = [random.random() for _ in xs[0]]

    for _ in tqdm.trange(num_steps, desc="least squares fit"):
        for start in range(0, len(xs), batch_size):
            batch_xs = xs[start:start + batch_size]
            batch_ys = ys[start:start + batch_size]

            gradient = vector_mean([sqerror_gradient(x, y, guess)
                                    for x, y in zip(batch_xs, batch_ys)])
            guess = gradient_step(guess, gradient, -learning_rate)

    return guess

def total_sum_of_squares(y: Vector) -> float:
    """Calcula a soma total dos quadrados das variações dos valores de y em relação à média"""
    return sum(v ** 2 for v in de_mean(y))

def multiple_r_squared(xs: List[Vector], ys: Vector, beta: Vector) -> float:
    """Calcula o coeficiente de determinação (R²) para a regressão linear múltipla."""
    sum_of_squared_errors = sum(error(x, y, beta) ** 2
                                for x, y in zip(xs, ys))
    return 1.0 - sum_of_squared_errors / total_sum_of_squares(ys)


