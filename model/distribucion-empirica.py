"""distribucion_empirica.py

Herramientas para generar distribuciones sintéticas de precios al vencimiento.

Aunque el proyecto ahora trabaja con datasets reales (CSV), sigue siendo útil
contar con generadores de distribuciones sintéticas para pruebas o benchmarking
cuántico.  Este módulo ofrece funciones parametrizables en lugar de *scripts*
ad-hoc.  Se incluyen docstrings detallados.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skewnorm
from typing import Sequence

__all__ = [
    "generate_skewed_distribution",
    "plot_distribution",
]


def generate_skewed_distribution(
    n_samples: int = 10000,
    spot: float = 100.0,
    skew: float = -3.0,
    scale: float = 20.0,
    seed: int | None = None,
) -> np.ndarray:
    """Genera precios sintéticos con sesgo y colas pesadas.

    Parameters
    ----------
    n_samples
        Número de muestras a retornar.
    spot
        Precio spot usado como **loc** de la distribución.
    skew
        Parámetro *a* de ``scipy.stats.skewnorm`` (negativo ⇒ cola pesada izq.).
    scale
        Desviación estándar (~volatilidad) de la distribución.
    seed
        Semilla opcional para reproducibilidad.

    Returns
    -------
    samples
        Vector 1-D de tamaño ``n_samples`` con precios > 0.
    """
    rng = np.random.default_rng(seed)
    samples = skewnorm.rvs(a=skew, loc=spot, scale=scale, size=n_samples, random_state=rng)
    return samples[samples > 0]


def plot_distribution(samples: Sequence[float], bins: int = 50) -> None:
    """Dibuja histograma normalizado de las muestras."""
    plt.hist(samples, bins=bins, density=True, alpha=0.6)
    plt.title("Distribución empírica simulada del subyacente")
    plt.xlabel("Precio al vencimiento")
    plt.ylabel("Densidad")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    samps = generate_skewed_distribution()
    plot_distribution(samps)
n_samples = 10000
S0 = 100
skew = -3  # Sesgo negativo: típico en activos de riesgo
scale = 20

samples = skewnorm.rvs(a=skew, loc=S0, scale=scale, size=n_samples)
samples = samples[samples > 0]  # Recortamos para que los precios sean positivos

plt.hist(samples, bins=50, density=True, alpha=0.6)
plt.title("Distribución empírica simulada del subyacente")
plt.xlabel("Precio al vencimiento")
plt.ylabel("Densidad")
plt.grid()
plt.show()
