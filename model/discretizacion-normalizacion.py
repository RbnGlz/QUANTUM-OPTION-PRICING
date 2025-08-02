"""discretizacion_normalizacion.py

Refactor 2025-08-02
-------------------
Este módulo solía ser un *script* ad-hoc que dependía de variables globales (`S0`, `samples`, etc.) y carecía
completamente de docstrings e imports requeridos.  Ahora se ha convertido en un módulo reutilizable con
funciones claras y documentación detallada.

Funciones principales
~~~~~~~~~~~~~~~~~~~~~
1. ``discretize_prices``  – devuelve probabilidades y amplitudes listas para ser codificadas.
2. ``plot_discretization`` – (opcional) muestra un histograma de la distribución discretizada.

El código está pensado para ser usado desde ``empirical_amplitude_encoding.py`` pero también se puede ejecutar
independientemente para probar la discretización.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Sequence

__all__ = ["discretize_prices", "plot_discretization"]


def discretize_prices(
    samples: Sequence[float],
    n_qubits: int,
    price_min: float | None = None,
    price_max: float | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Discretiza precios continuos en *bins* para amplitud-encoding.

    Parameters
    ----------
    samples
        Colección de precios al vencimiento.
    n_qubits
        Número de qubits ⇒ ``n_bins = 2 ** n_qubits``.
    price_min, price_max
        Límites del rango.  Si son *None*, se usan ``min(samples)`` y ``max(samples)``.

    Returns
    -------
    probs
        Probabilidades normalizadas por *bin*.
    amplitudes
        Raíz cuadrada de las probabilidades (vector ya normalizado).
    bins
        Array con límites de los *bins*.
    """
    samples = np.asarray(samples, dtype=float)
    if price_min is None:
        price_min = float(samples.min())
    if price_max is None:
        price_max = float(samples.max())

    n_bins = 2 ** n_qubits
    bins = np.linspace(price_min, price_max, n_bins + 1)
    hist, _ = np.histogram(samples, bins=bins)
    probs = hist / hist.sum()
    probs = np.where(probs < 1e-12, 1e-12, probs)  # evitar ceros
    probs /= probs.sum()
    amplitudes = np.sqrt(probs)
    amplitudes /= np.linalg.norm(amplitudes)
    return probs, amplitudes, bins


def plot_discretization(probs: np.ndarray) -> None:
    """Dibuja un histograma de las probabilidades discretizadas."""
    plt.bar(range(len(probs)), probs)
    plt.xlabel("Bin")
    plt.ylabel("Probabilidad")
    plt.title("Distribución discretizada para amplitud-encoding")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Ejemplo rápido de uso con datos aleatorios
    rng = np.random.default_rng(42)
    dummy_prices = rng.lognormal(mean=4.5, sigma=0.2, size=10_000)
    p, a, b = discretize_prices(dummy_prices, n_qubits=4)
    print("Probabilidades discretizadas:", p)
    plot_discretization(p)

