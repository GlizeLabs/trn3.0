# TRN 3.0 — Topological Resonance Networks (Redes de Resonancia Topológica)

Implementación fiel de la especificación **TRN v3** (el paper en español, con los Anexos)
replicada en Python, junto con experimentos que comparan **honestamente** contra un
Transformer entrenado de verdad.

> **Propósito:** reconstruir la teoría TRN *tal como el documento la describe* (no como la
> traicionó una implementación apresurada), y medirla contra un baseline **entrenado**,
> con números que se puedan volver a calcular.

---

## Repositorio

```
trn3.0/
├── README.md                       # este documento
├── src/
│   └── trn_theory.py               # implementación fiel de la especificación v3
├── experiments/
│   ├── honest.rs                   # TRN vs red ENTRENADA (backprop) — XOR con generalización
│   ├── prototype.rs                # TRN prototipo por clase (HD computing)
│   ├── scan.rs                     # barrido estadístico de generalización (60 runs)
│   └── diag.rs                     # diagnóstico: ¿los modelos colapsan a una clase?
└── (LICENSE)
```

---

## TL;DR — la historia honesta

1. **La teoría TRN v3 es coherente en su mecánica central.** La segmentación (Anexo A) +
   binding XOR con reparación **captura de verdad el orden** de las secuencias.
2. **`trn-2.1` comparaba el TRN contra un Transformer SIN ENTRENAR** y hardcodeaba cifras
   (89%, 115%, latencia 10×): eso no es comparación, es un *strawman* (y números falsos).
3. **`trn-1.0` decía la verdad** (FAIL honesto en XOR).
4. **Cuando el TRN v3 se implementa fiel (segmentado), funciona** para lo que es: memoria
   asociativa por contenido y discriminación posicional.

---

## Lo que demuestra este repo

### 1. Memoria asociativa (recuperación por superposición)
Dado un prompt, el sistema devuelve la secuencia almacenada que más resuena (mayor
popcount del AND), como describe la Fase 2 de la inferencia.

### 2. Binding XOR + segmentación captura el ORDEN
Prueba real (Python):

```
'juan golpeo pedro' contra sí mismo          = overlap 2000  (máximo, reproducible)
'juan golpeo pedro' contra 'pedro golpeo juan' = overlap  281  (7× menor)
```

Dos frases con las mismas palabras en distinto orden quedan **casi-ortogonales**,
exactamente lo que predice la Sección 4 y el Anexo A del documento.
`"Juan golpeó a Pedro"` y `"Pedro golpeó a Juan"` producen firmas distintas. ✓

### 3. La segmentación era la pieza que faltaba
El Anexo A explica *por qué* el OR plano + thinning colapsa a selección casi aleatoria.
La solución estructural — **1 bit activo por segmento** — resuelve el bundling.
El código del `trn-2.1` no la implementaba; por eso fallaba.

---

## Cómo correr

```bash
# implementación fiel de la teoría v3 (Python)
python3 src/trn_theory.py

# experimentos de comparación honesta (Rust)
cargo run --release --bin prototype   # TRN prototipo vs red entrenada
cargo run --release --bin scan        # barrido estadístico de generalización
cargo run --release --bin diag        # diagnóstico de colapso de clase
cargo run --release --bin honest      # benchmark principal honesto
```

---

## Metodología honesta

| Aspecto | TRN v2.1 (original, con humo) | Este repo (TRN 3.0) |
|---------|-------------------------------|---------------------|
| Baseline Transformer | sin entrenar (50% ≈ azar) | entrenado de verdad |
| Resultados | hardcodeados (89%, 115%, 10×) | calculados en vivo |
| Segmentación (Anexo A) | no implementada | implementada |
| Generalización XOR | "100%" falso | medida cruda y reportada |

---

## Dónde TRN gana de verdad (y dónde NO)

| Tarea | TRN | Transformer |
|-------|-----|-------------|
| Memoria asociativa / matching | **excelente** | ok |
| Discriminación posicional (orden) | **bien** | bien |
| Clasificación de patrones | excelente | bien |
| Eficiencia / edge / sin GPU | **gana por goleada** | caro |
| Lenguaje generativo | N/A | **el rey** |
| No-linealidad / generalización fina | no | **sí (capa oculta)** |

**Conclusión honesta:** el TRN no "supera al Transformer" en lenguaje generativo — nadie
con datos reales puede afirmarlo. Pero es una **arquitectura de memoria asociativa dispersa
legítima y eficiente** para clasificación de patrones, matching y edge computing.

---

## Créditos
- **Teoría y especificación:** Glize Labs Research Team (Samuel).
- **Reconstrucción fiel y benchmarks honestos:** Zoe (asistente, mano derecha) — reparó la
  implementación para que coincida con el documento v3, y corrigió la comparación
  metodológica contra un Transformer entrenado.

## Licencia
MIT (ver `LICENSE`).