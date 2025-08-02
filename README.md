# Codificación cuántica de distribuciones empíricas de precios al vencimiento

![Quantum Circuit Example](https://user-images.githubusercontent.com/your-repo/quantum-circuit-example.png)

Este repositorio implementa técnicas avanzadas de computación cuántica para codificar distribuciones empíricas de precios de activos financieros al vencimiento en un estado cuántico, utilizando Qiskit y codificación por amplitud. Es relevante para pricing de opciones y simulaciones financieras en computadoras cuánticas.

---

## Ejemplo visual: Histograma y circuito cuántico

```python
import matplotlib.pyplot as plt
from model.discretizacion_normalizacion import discretize_prices
from model.distribucion_empirica import generate_skewed_distribution

# Generar datos sintéticos
prices = generate_skewed_distribution(n_samples=10000, spot=100, skew=-3, scale=20, seed=42)
probs, amplitudes, bins = discretize_prices(prices, n_qubits=4)
plt.bar(range(len(probs)), probs)
plt.xlabel('Bin'); plt.ylabel('Probabilidad'); plt.title('Distribución discretizada'); plt.show()
```

![Ejemplo de histograma](https://user-images.githubusercontent.com/your-repo/histograma-ejemplo.png)

```python
from model.inicializar_estado import initialize_state
qc, probs_measured = initialize_state(amplitudes)
print(qc.draw('mpl'))  # Visualiza el circuito cuántico
```

---

## Ejemplo de uso

### Instalación
```bash
pip install -r requirements.txt
```

### Ejecución con datos reales
```bash
python model/empirical_amplitude_encoding.py --csv datos_opciones.csv --column price --n-qubits 4 --mitigation bitflip
```

### Ejecución con datos sintéticos y mitigación avanzada
```bash
python model/empirical_amplitude_encoding.py --csv synthetic --n-qubits 4 --mitigation dd
```

### Opciones de mitigación disponibles
- `bitflip`: Modelo simple de bit-flip (por defecto)
- `readout`: Simulación de readout error mitigation (requiere hardware real para calibración)
- `dd`: Dynamical decoupling (transpilación avanzada)
- `none`: Sin mitigación

---

## Cambios recientes
- Unificación de la lógica de discretización en `discretizacion-normalizacion.py`.
- Mejora de la mitigación de errores cuánticos: ahora se soportan técnicas modernas (`bitflip`, `readout`, `dd`).
- Permite generación sintética de datos desde CLI (`--csv synthetic`).
- Ejemplos visuales y de uso añadidos.

---

## Estructura del código
- `model/empirical_amplitude_encoding.py`: Flujo principal y CLI.
- `model/discretizacion-normalizacion.py`: Discretización y normalización de precios.
- `model/distribucion-empirica.py`: Generación de distribuciones sintéticas.
- `model/inicializar-estado.py`: Inicialización y visualización de estados cuánticos.

---

## Referencias
- Stamatopoulos, N., et al. (2019). Option Pricing using Quantum Computers. [arXiv:1905.02666](https://arxiv.org/abs/1905.02666)
- Qiskit Textbook: [Quantum Amplitude Encoding](https://qiskit.org/textbook/ch-applications/finance.html)

---

## Contribución y comunidad
¿Quieres contribuir? Abre un issue o pull request siguiendo las plantillas en `.github/`.

---

## Licencia
Ver archivo LICENSE.

