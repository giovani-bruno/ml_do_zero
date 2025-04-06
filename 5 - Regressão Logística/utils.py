from typing import List, TypeVar, Tuple
import random, math

Vector = List[float]
X = TypeVar('X')
Y = TypeVar('Y')

def vector_sum(vectors: List[Vector]) -> Vector:
    """Soma todos os elementos correspondentes entre os vetores"""
    assert vectors, "nenhum vetor fornecido!"
    num_elements = len(vectors[0])
    assert all(len(v) == num_elements for v in vectors), "vetores de tamanhos diferentes!"
    return [sum(vector[i] for vector in vectors)
            for i in range(num_elements)]

def dot(v: Vector, w: Vector) -> float:
    """Calcula v_1 * w_1 + ... + v_n * w_n"""
    assert len(v) == len(w), "os vetores devem ter o mesmo comprimento"
    return sum(v_i * w_i for v_i, w_i in zip(v, w))

def split_data(data: List[X], prob: float) -> Tuple[List[X], List[X]]:
    data = data[:]
    random.shuffle(data)
    cut = int(len(data) * prob)
    return data[:cut], data[cut:]

def train_test_split(xs: List[X],
                     ys: List[Y],
                     test_pct: float) -> Tuple[List[X], List[X], List[Y], List[Y]]:
    idxs = [i for i in range(len(xs))]
    train_idxs, test_idxs = split_data(idxs, 1 - test_pct)

    return ([xs[i] for i in train_idxs],
            [xs[i] for i in test_idxs], 
            [ys[i] for i in train_idxs],
            [ys[i] for i in test_idxs]) 

def add(v: Vector, w: Vector) -> Vector:
    """Soma os elementos correspondentes"""
    assert len(v) == len(w), "os vetores devem ter o mesmo comprimento"
    return [v_i + w_i for v_i, w_i in zip(v, w)]

def scalar_multiply(c: float, v: Vector) -> Vector:
    """Multiplica cada elemento por c"""
    return [c * v_i for v_i in v]

def gradient_step(v: Vector, gradient: Vector, step_size: float) -> Vector:
    """Move 'step_size' na direção do 'gradient' a partir de 'v'"""
    assert len(v) == len(gradient)
    step = scalar_multiply(step_size, gradient)
    return add(v, step)

def vector_mean(vectors: List[Vector]) -> Vector:
    """Calcula a média elemento a elemento"""
    n = len(vectors)
    return scalar_multiply(1/n, vector_sum(vectors))

def sum_of_squares(v: Vector) -> float:
    """Retorna v_1 * v_1 + ... + v_n * v_n"""
    return dot(v, v)

def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs)

def de_mean(xs: List[float]) -> List[float]:
    """Subtrai a média de xs (para que o resultado tenha média 0)"""
    x_bar = mean(xs)
    return [x - x_bar for x in xs]

def variance(xs: List[float]) -> float:
    """Quase a média dos desvios quadráticos da média"""
    assert len(xs) >= 2, "variância requer pelo menos dois elementos"
    n = len(xs)
    deviations = de_mean(xs)
    return sum_of_squares(deviations) / (n - 1)

def standard_deviation(xs: List[float]) -> float:
    """O desvio padrão é a raiz quadrada da variância"""
    return math.sqrt(variance(xs))

def scale(data: List[Vector]) -> Tuple[Vector, Vector]:
    """Retorna as médias e desvios padrão de cada posição"""
    dim = len(data[0])
    means = vector_mean(data)
    stdevs = [standard_deviation([vector[i] for vector in data])
              for i in range(dim)]
    return means, stdevs

def rescale(data: List[Vector]) -> List[Vector]:
    """
    Reescala os dados de entrada para que cada posição tenha
    média 0 e desvio padrão 1. (Mantém a posição como está se o desvio for 0.)
    """
    dim = len(data[0])
    means, stdevs = scale(data)

    rescaled = [v[:] for v in data]

    for v in rescaled:
        for i in range(dim):
            if stdevs[i] > 0:
                v[i] = (v[i] - means[i]) / stdevs[i]

    return rescaled
