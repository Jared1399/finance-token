# finance-token

Tokenizador de morfología de velas financieras. Convierte cada vela OHLC en un
token discreto invariante a escala, usando **product quantization** sobre dos
codebooks aprendidos del mercado: forma y magnitud relativa.

El repo implementa el tokenizador y su evaluación. No incluye (todavía) el
transformer: la Fase 5 del diseño.

## Idea

Dado $R_t = H_t - L_t$:

```
o = (O - L) / R        # posición de la apertura
c = (C - L) / R        # posición del cierre
```

`(o, c) ∈ [0,1]²` determina completamente la morfología de la vela. Las features
clásicas (`body`, `wick_up`, `wick_low`) cumplen
`|body| + wick_up + wick_low = 1`, así que viven en un símplex 2D: usarlas como
tres dimensiones independientes sesga la distancia euclidiana y produce
centroides que no son velas válidas.

La magnitud se normaliza por volatilidad reciente:

```
range_rel = ln(R / ATR_k)
z = (range_rel - mean) / std     # estadísticos solo de train, con clip
```

El vocabulario es el producto cartesiano de ambos codebooks:

```
0 [PAD] | 1 [BOS] | 2 [FLAT] | 3 + shape_id * K_v + mag_id
```

Con `K_s=512` y `K_v=16` el vocabulario es 8195. Ese par se eligió con
`ft sweep` minimizando `p95_forma + p95_cierre` bajo un presupuesto de 10.240
tokens. La embedding puede factorizarse como
`Emb_forma(shape_id) + Emb_magnitud(mag_id)`, compartiendo estructura entre
formas similares y reduciendo parámetros.

## Instalación

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

## Uso

Demo end-to-end con datos sintéticos (no requiere red, aislado en `data/demo/`):

```bash
ft demo --years 3
```

Datos reales de Binance Vision (klines 1m, se agregan a 15m):

```bash
ft download          # ~2 MB por símbolo/mes, paralelo
ft prepare           # resample 15m + validación + features causales
ft fit-tokenizer     # K-Means forma + magnitud, artefacto versionado
ft encode            # tokens por vela + state + scale
ft report            # reports/tokenizer.md + figuras
ft sweep             # barrido K_s x K_v -> reports/sweep.md
```

`ft all` hace todas las etapas de datos reales de una vez.

Todo se configura en `configs/tokenizer.yaml`; las rutas son relativas al
directorio desde donde se ejecuta `ft`.

## Estructura

```
src/financetoken/
├── config.py              # config tipada + fingerprint
├── pipeline.py            # ingest → features → codebook → tokens
├── dataset.py             # ventanas de tokens (torch opcional)
├── data/
│   ├── binance.py         # Binance Vision (zip + CHECKSUM)
│   ├── resample.py        # 1m → 15m, bar_gap
│   ├── validate.py        # invariantes OHLCV
│   └── synth.py           # datos sintéticos con formato Binance
├── features/
│   ├── indicators.py      # ATR Wilder causal, SMA, std, mediana móvil
│   ├── candle.py          # (o,c), mechas, range_rel
│   └── state.py           # RangeScaler, StateQuantizer
├── tokenizer/
│   ├── shape.py           # K-Means 2D + poda de centroides muertos
│   ├── magnitude.py       # K-Means 1D ordenado
│   ├── pq.py              # fit/encode/decode/save/load
│   └── vocab.py
├── eval/
│   ├── metrics.py         # uso, entropía, silueta, reconstrucción, gap
│   ├── plots.py
│   ├── report.py          # Markdown + criterios de aceptación
│   └── sweep.py           # barrido K_s x K_v
└── cli.py
```

## Decisiones y supuestos

- **Causalidad estricta.** ATR, SMA, σ y `ATR_long` usan solo el pasado. Los
  estadísticos de normalización se ajustan **solo en train**.
- **Embargo** al inicio de val/test: `max(atr_long_window + atr_k, sma_m) + 1`
  velas, para que ninguna ventana de feature cruce la frontera del split.
- **`[FLAT]` solo si `R == 0` exacto.** Las velas casi-planas caen en el bucket
  de magnitud más bajo; colapsarlas con `[FLAT]` destruye información.
- **El token es contextual**: `range_rel` depende del ATR de las últimas `k`
  velas, así que la misma vela puede tokenizar distinto en otro contexto. El ATR
  se guarda junto al token para poder decodificar.
- **El gap de apertura no está modelado.** El decoder ancla en `prev_close`; el
  reporte mide `|open - prev_close|` en R y en bps (p50/p99). Si es material,
  hace falta un tercer codebook o un feature de gap.
- **`scale = ln(ATR / ATR_long)`** se guarda sin tokenizar. Sin este canal (o
  algo equivalente) los tokens no permiten reconstruir volatilidad absoluta ni
  nivel de precio, solo forma relativa.
- **Sin fuga entre símbolos**: el codebook se ajusta solo con `fit_symbols`;
  `holdout_symbols` se usa para medir transferencia.

## Resultados sobre datos reales

Corrida sobre Binance Vision a 15m, ajustando el codebook solo con BTC y ETH
(2017-08 → 2023-12) y usando SOL como holdout:

| Métrica | Valor |
|---|---|
| Velas 15m procesadas | 853.703 (564.171 de train) |
| Codebook | 512 forma × 16 magnitud = 8.195 tokens |
| Error de forma p50 / p95 | 0.0146 / 0.0253 R |
| Error de cierre p95 (test) | 0.0658 R ≈ **3.9 bps** |
| Códigos muertos | 0 / 512 |
| Entropía de uso normalizada | 0.993 |
| Cobertura en holdout (SOL) | 512 / 512 códigos, close p95 4.7 bps |
| Gap de apertura p99 (test) | 1.35 bps |

Los patrones canónicos caen en centroides distintos y cercanos: Marubozu alcista
y bajista en IDs 119 y 160 (los dos tokens más frecuentes), Doji en 18, Hammer en
456, Shooting Star en 92. El detalle está en `reports/tokenizer.md` y
`reports/sweep.md`.

`ft sweep` muestra rendimientos decrecientes: duplicar el vocabulario reduce el
error ~30%, pero el costo en parámetros y en frecuencia por token crece. Bajo un
presupuesto de 10.240 tokens, 512×16 minimiza `p95_forma + p95_cierre`.

## Tests

```bash
.venv/bin/pytest
```

Incluye: invariantes OHLC, agregación 1m→15m y `bar_gap`, ATR de Wilder contra
cálculo manual, test explícito de no-lookahead, invariancia a escala,
consistencia del símplex, round-trip del codebook, utilidades de asignación
vectorizada, y un end-to-end con datos sintéticos que verifica reconstrucción y
reporte.
