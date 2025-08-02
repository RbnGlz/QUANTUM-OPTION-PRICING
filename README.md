# Codificación cuántica de distribuciones empíricas de precios al vencimiento

Este repositorio implementa una técnica avanzada para codificar distribuciones empíricas de precios de activos financieros al vencimiento en un estado cuántico, utilizando Qiskit y codificación por amplitud. Esta metodología es relevante para aplicaciones de pricing de opciones y simulaciones financieras en computadoras cuánticas.

## Descripción general

La codificación por amplitud permite representar una distribución de probabilidad discreta en el estado de un registro de qubits, de modo que la probabilidad de medir cada estado base corresponde a la probabilidad empírica de un rango de precios discretizado. Esto es fundamental para algoritmos cuánticos de finanzas, como el pricing de opciones y la estimación de riesgos.

## Estructura del código

El archivo principal es `model/empirical_amplitude_encoding.py`, que incluye:

- **Preprocesamiento de datos:** Discretización y normalización de precios históricos al vencimiento.
- **Codificación cuántica:** Preparación de un circuito cuántico que codifica la distribución empírica mediante la inicialización de amplitudes.
- **Mitigación de errores:** Inclusión de un modelo de ruido simple para simular y mitigar errores típicos de hardware cuántico.
- **Verificación:** Cálculo de la fidelidad entre el estado cuántico preparado y la distribución objetivo.
- **Comparativa clásica:** Comparación entre la distribución cuántica simulada y el histograma clásico.

## Ejecución y pruebas

Puedes ejecutar el archivo pasando un CSV con precios reales o bien usar datos simulados.

### Instalación de dependencias

```bash
pip install -r requirements.txt
```

### Ejecución con datos reales

```bash
python model/empirical_amplitude_encoding.py --csv datos_opciones.csv --column price --n-qubits 4
```

### Ejecución con datos simulados (pruebas rápidas)

```bash
python model/empirical_amplitude_encoding.py --csv data.csv --column price
```

El script simula precios al vencimiento, discretiza y normaliza los datos, prepara el estado cuántico, simula la mitigación de errores y verifica la fidelidad de la codificación.

## Conceptos clave

- **Codificación por amplitud:** Técnica para preparar un estado cuántico \( \sum_i \sqrt{p_i} |i\rangle \) donde \( p_i \) es la probabilidad de cada bin de precios.
- **Discretización:** Los precios se agrupan en bins definidos por el número de qubits, permitiendo mapear los precios a estados computacionales.
- **Fidelidad:** Métrica que mide la similitud entre el estado cuántico preparado y el estado objetivo. Es crucial para validar la calidad de la codificación.
- **Mitigación de errores:** Estrategias para reducir el impacto de errores inherentes al hardware cuántico, aquí ejemplificado con un modelo de bit-flip.

## Justificación matemática

La codificación por amplitud se basa en preparar un estado cuántico donde la probabilidad de medir cada estado base \( |i\rangle \) es igual a la probabilidad empírica del bin correspondiente. Esto se logra inicializando el registro de qubits con amplitudes proporcionales a la raíz cuadrada de las probabilidades normalizadas.

## Referencias

- Stamatopoulos, N., Egger, D. J., Sun, Y., Zoufal, C., Iten, R., Shen, N., & Woerner, S. (2019). Option Pricing using Quantum Computers. Quantum, 4, 291. [arXiv:1905.02666](https://arxiv.org/abs/1905.02666)
- Qiskit Textbook: [Quantum Amplitude Encoding](https://qiskit.org/textbook/ch-applications/finance.html)

## Ampliación y recomendaciones

- Puedes adaptar el número de qubits y los límites de precios según la resolución deseada y la disponibilidad de hardware.
- Para datos reales, asegúrate de limpiar y filtrar outliers antes de la discretización.
- La mitigación de errores puede mejorarse usando técnicas más avanzadas según el backend cuántico utilizado.
- Este enfoque es la base para algoritmos cuánticos de pricing de opciones, estimación de VaR y simulaciones de Monte Carlo cuánticas.

---

Para cualquier duda o ampliación, consulta los comentarios en el código fuente o las referencias académicas incluidas.

