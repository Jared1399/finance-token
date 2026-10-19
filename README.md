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

Para el transformer con GPU Pascal (GTX 1060 / sm_61) hace falta un wheel
CUDA 11.8; los builds nuevos de PyTorch ya no incluyen sm_61:

```bash
.venv/bin/pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu118
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

Transformer causal (PoC):

```bash
ft train       # GPT chico sobre tokens; checkpoint en artifacts/transformer
ft eval-gpt    # one-step + calibracion + closed-loop -> reports/transformer.md
```

Stage 0 direccional (sin GPU): ¿hay skill a 1h/4h/12h antes de invertir en un
modelo de secuencia?

```bash
ft direction   # logistic + HistGradientBoosting + backtest neto -> reports/direction_12h.md
```

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
| Error de forma p50 / p95 | 0.0143 / 0.0252 R |
| Error de cierre p95 (test) | 0.0708 R ≈ **4.3 bps** |
| Códigos muertos | 0 / 512 |
| Entropía de uso normalizada | 0.993 |
| Cobertura en holdout (SOL) | 512 / 512 códigos, close p95 5.2 bps |
| Gap de apertura p99 (test) | 1.35 bps |

El error de cierre usa `ATR_{t-1}`. Con el ATR contemporáneo (oráculo) daba 3.9
bps, pero ese valor no es decodificable en generación porque el ATR incluye el
rango de la vela que se quiere predecir.

Los patrones canónicos caen en centroides distintos y cercanos: Marubozu alcista
y bajista en IDs 119 y 160 (los dos tokens más frecuentes), Doji en 18, Hammer en
456, Shooting Star en 92. El detalle está en `reports/tokenizer.md` y
`reports/sweep.md`.

`ft sweep` muestra rendimientos decrecientes: duplicar el vocabulario reduce el
error ~30%, pero el costo en parámetros y en frecuencia por token crece. Bajo un
presupuesto de 10.240 tokens, 512×16 minimiza `p95_forma + p95_cierre`.

## Transformer (PoC)

El decoder es un GPT causal con entrada híbrida:

```
E_t = Emb_token(Token_t) + Emb_state(State_t) + Linear(scale_t) + Emb_gap(bar_gap_t) + PE(t)
```

- `Emb_token` es **factorizada**: `Emb_shape(s) + Emb_magnitude(v)`.
- La cabeza también es factorizada: `kind` (PAD/BOS/FLAT/MORPH) + `shape` (512) +
  `magnitude` (16). La probabilidad conjunta es
  `P(kind) · P(shape) · P(magnitude)`; la cabeza pasa de 8195·d a 532·d. Con
  ~445k tokens de train, una softmax completa de 8195 clases está
  subentrenada (~54 ejemplos por clase).
- Se predicen 256 velas de contexto. El objetivo es el token siguiente; la
  pérdida es la NLL de la distribución factorizada.
- En `ft eval-gpt` se reportan CE/perplejidad contra unigrama y bigrama
  interpolado, calibración de dirección/magnitud, y generación closed-loop
  comparada contra los mismos baselines usando el mismo decoder.

La normalización de magnitud usa **`ATR_{t-1}`**, no `ATR_t`. Con `ATR_t` el
decoder necesitaría resolver un punto fijo (`R_t` aparece en el `TR_t` que
define `ATR_t`), y el token normalizaría por su propio rango. El fix hace que el
token sea decodificable con información ya conocida y que el target sea más
estable. El costo es un error de cierre honesto de ~4.3 bps en vez de los 3.9
bps que reportaba el decoder con ATR oráculo.

### Resultados del PoC

Corrida real: BTC/ETH train (2017-2023), val 2024, test 2025-2026; SOL holdout.
Modelo de 997k parámetros (`d_model=128`, 4 capas, 4 heads, contexto 256).

| Métrica | Valor |
|---|---|
| Test CE (nats/token) | 8.6199 |
| Unigram | 8.6570 |
| SOL holdout CE | 8.6008 |
| Bigram | se apaga a λ=0; no hay estructura de orden 1 |
| Dirección: Brier | 0.2500 (tasa base 0.2500) |
| Dirección: ECE | 0.0072 |
| Magnitud `>0.5 R/ATR`: Brier | 0.2356 vs 0.2384 unigram |
| Closed-loop ACF1 | 0.349 modelo / 0.241 real / 0.440 unigram |
| Closed-loop ACF5 | 0.286 modelo / 0.140 real |
| Closed-loop Hill | 4.18 modelo / 3.06 real / 3.04 unigram |
| Closed-loop kurtosis | 10.7 modelo / 14.3 real / 41.6 unigram |

Lectura: el transformer aprende algo de dinámica (mejor que los baselines en ACF
de corto plazo y Hurst), pero **el cuello de botella son los datos**: empieza a
overfitear en la epoch 3, la mejora sobre el unigram es 0.43%, no hay skill
direccional y subrepresenta las colas. El unigram preserva mejor la cola (Hill
3.04) pero dispara la kurtosis (41.6). El siguiente paso no es una arquitectura
más grande: es ampliar el universo de símbolos y probar decodificación/loss
tail-aware. Detalle completo en `reports/transformer.md`.

## Stage 0 direccional

Objetivo: decidir si hay skill direccional en ventanas de hasta 12h **antes** de
gastar GPU. No es un modelo de secuencia; es un screening barato con features
causales y clasificadores tabulares.

- Targets: signo del retorno forward a 1h, 4h y 12h (4, 16 y 48 velas de 15m).
- Features: ~50 features causales (momentum multi-horizonte normalizado por ATR,
  tendencia/SMA, RSI/MACD/Bollinger, rango/drawdown, volatilidad, volumen,
  calendario).
- Splits: BTC/ETH train 2017-2023, val 2024; test 2025-2026 con SOL como
  holdout. Muestreo no solapado y purga de `h` barras en las fronteras.
- Modelos: logistic L2 y HistGradientBoosting; selección por AUC en validación.
- Backtest: long/short en futuros, 10 bps round trip, sin funding (Stage 0).
  Se reportan también 0/2/4/5/20 bps para ver el break-even.
- Ventanas de alta direccionalidad: se elige en validación el top-q% de
  `|P(subida)-0.5|` y se aplica a test; no se elige el decil mirando test.

Resultado (detalle en `reports/direction_12h.md`):

- **12h: sin evidencia**, ni bruta ni neta. El AUC test es ~0.505 y el Sharpe
  bruto a 0 bps es negativo.
- **4h: candidato cost-sensitive**. El top-10% de confianza tiene Sharpe neto
  val 0.39 y test 0.51 a 10 bps, Sharpe bruto test 1.66, retorno neto +45% y
  57.4% de accuracy. Está traccionado por ETH; BTC aporta negativo.
- **1h: señal bruta** (Sharpe bruto test ~1.3) pero validación neta negativa y
  costos/turnover la eliminan.
- Conclusión: no entrenar un transformer direccional todavía. El candidato 4h
  necesita walk-forward por año, per-símbolo, funding y ejecución maker antes
  de justificar GPU.

## Flujo 4h: taker, funding y open interest

```bash
ft flow-4h   # ingesta/reutiliza flujo, baseline vs flow, calibracion -> reports/flow_4h.md
```

Reingesta spot 1m con `taker_buy_base` y construye:

- Taker Buy Ratio y CVD a 15m, 1h, 4h y 12h (26 features de flujo en total).
- Funding Rate 8h desde Binance Vision futures, alineado as-of.
- Open Interest 5m y taker long/short ratio desde `futures/um/daily/metrics`.

El modelo es HistGradientBoosting, train 2020-2023, val 2024, test 2025-2026;
se compara **baseline** (precio/State/Scale) contra **flow** (baseline + flujo).
La calibración se elige por ECE con 5-fold CV en validación (Isotonic vs Platt,
con margen de estabilidad para Platt). **No se calculó PnL ni Sharpe.**

Resultado:

| Métrica | Baseline | Flow |
|---|---|---|
| Rank IC test BTC/ETH 2025-26 | 0.0423 | 0.0406 |
| Rank IC SOL holdout | 0.0137 | 0.0134 |
| ECE test BTC/ETH (raw → calibrado) | 0.0432 → 0.0207 | 0.0394 → **0.0144** |
| ECE SOL (raw → calibrado) | 0.0450 → 0.0273 | 0.0463 → **0.0219** |

El objetivo de **IC 0.08+ no se alcanzó**. El flujo no agregó información
direccional out-of-sample; en 2026 mejoró y en 2025 empeoró, sin robustez.
La calibración Platt sí reduce el ECE a la mitad o más, pero no cambia el ranking.
El candidato 4h de Stage 0 sigue siendo el de precio/State/Scale, no el de flujo.

## Agotamiento de momentum / reversión a la media (4h)

```bash
ft exhaustion-4h   # features de rachas, ER, VWAP, aceleración -> reports/exhaustion_experiment.md
```

Features nuevas: `streak_len`/`streak_pct` (percentil rolling de rachas),
`z_return_4/16/32`, `efficiency_ratio_16/32` (Kaufman), distancia a SMA100 y
VWAP 24h normalizadas por ATR, y `momentum_accel` (cambio del impulso 4h).
Baseline = features de precio/State/Scale; Exhaustion = baseline + 11 features
nuevas. Train 2020-2023, val 2024, test 2025-2026, SOL holdout; Platt en
validación. **Sin PnL ni Sharpe.**

Resultado:

- Rank IC test BTC/ETH: 0.0423 → 0.0449 (+0.0026); SOL 0.0137 → 0.0093 (peor).
- ECE exhaustion: raw 0.0370 → Platt 0.0189 en test; SOL 0.0480 → 0.0212.
- Hit rate P9: 56.9% test, 51.5% SOL; los deciles no son monótonos.
- Rachas >95%: high_up test +2.5 bps, reversión 51.9%, t=0.24; high_down
  +14.5 bps, reversión 51.9%, t=1.93. En SOL no hay reversión.
- Conclusión: las features de agotamiento no mejoran el IC de forma robusta y
  las rachas extremas no muestran una reversión efectiva. Platt mejora la
  calibración, no el ranking. No justifica GPU.

## Gating de régimen 4h: continuidad vs reversión

```bash
ft regime-4h   # compresión vs expansión, modelos especializados -> reports/regime_experiment.md
```

Features: `bb_keltner_ratio` (BB 20,2 / Keltner 20,1.5·ATR14),
`volatility_regime` (percentil rolling 30d: compresión/normal/expansión),
`trend_slope_norm` (pendiente OLS 32 velas / precio) y `rvol_streak`
(volumen medio de la racha / media 96). Se entrenan modelos especializados:
`model_range` para compresión, `model_trend` para expansión y `model_normal`;
el baseline usa solo precio/State/Scale. **Sin PnL ni Sharpe.**

Resultado:

- Regímenes ~1/3 cada uno en train, val y test.
- Rank IC test BTC/ETH: compresión baseline 0.0707, regime_all 0.0691, gated
  0.0353; expansión 0.0406 / 0.0389 / 0.0460. SOL: compresión 0.0116 / -0.0041
  / 0.0027; expansión -0.0087 / 0.0039 / -0.0128. El gating no mejora de forma
  consistente; en compresión BTC/ETH destruye el IC del baseline.
- Rachas >95% (BTC/ETH): expansión continuación realizada 49.7%, model_hit
  55.7% (t=1.39); compresión reversión realizada 47.9%, model_hit 47.9%
  (t=-0.36). SOL: 53.7%/50.0% y 56.7%/53.3%. **Ningún hit rate supera 58% con
  t>2.**
- Conclusión: el régimen no agrega una regla explotable; no justifica GPU.
  Detalle en `reports/regime_experiment.md`.

## Triples Barreras con Monte Carlo autoregresivo

```bash
ft triple-barrier --config configs/triple_barrier.yaml
# piloto representativo (167 puntos por símbolo y año, M=500):
ft triple-barrier --paths 500 --points-per-symbol-year 167
```

Para cada punto del test se toman 256 velas de contexto y se generan `M=500`
trayectorias de 48 velas (12h) con el Transformer en closed-loop. En cada vela
se recomputan ATR de Wilder, State (SMA/σ de 100) y Scale (log ATR / mediana
30d). Las barreras son `+1.5 ATR` (TP long) y `-1.0 ATR` (SL long); para short,
`-1.5 ATR` (TP) y `+1.0 ATR` (SL). Si una vela toca ambas, se resuelve con la
morfología (alcista: low antes que high; bajista: high antes que low; doji: la
más cercana al open).

El target real es 1 si el TP Long se toca primero, 0 si SL Long o timeout. Se
reportan Rank IC de Spearman, ECE y deciles de `P_hat(Long)`, y Top 10% long /
short con hit rate real de TP y EV neto (TP +1.5 ATR, SL -1.0 ATR, timeout al
cierre, menos 10 bps). **No se predicen signos individuales de velas.**

El grid completo con stride=16 son ~10.935 puntos. El run completo con M=500
cuesta ~12-16 h en la GTX 1060; por eso el reporte usa un piloto aleatorio
estratificado por símbolo y año: 1.002 puntos (167 por símbolo/año).

Resultados del piloto:

- Rank IC BTC/ETH: -0.0301 (2025 -0.0498, 2026 -0.0115). SOL: +0.0707
  (2025 +0.0906, 2026 +0.0406). Ningún IC es significativo; el mejor (SOL
  2025) tiene t≈1.17.
- ECE: 0.0399 BTC/ETH y 0.0173 SOL, pero `P_hat` medio queda ~3 pp debajo de
  la tasa base y los deciles no son monótonos. Calibración global sin
  discriminación.
- Top 10% long BTC/ETH: hit 0.388 vs base 0.424, EV -13.2 bps; short: hit
  0.373, EV -5.8 bps. SOL long: hit 0.500, EV -0.3 bps (el costo anula el
  bruto); short: 0.382, EV -5.2 bps. Sin cola robusta.
- **Veredicto**: las trayectorias no capturan path dependency útil para
  triples barreras; no hay base para un modelo direccional o GPU adicional.
  Detalle en `reports/triple_barrier_eval.md` y
  `reports/triple_barrier_metrics.json`.

## Tests

```bash
.venv/bin/pytest
```

Incluye: invariantes OHLC, agregación 1m→15m y `bar_gap`, ATR de Wilder contra
cálculo manual, test explícito de no-lookahead, invariancia a escala,
consistencia del símplex, round-trip del codebook, utilidades de asignación
vectorizada, un end-to-end con datos sintéticos que verifica reconstrucción y
reporte, y tests del transformer (distribución factorizada suma 1, ventanas que
no cruzan símbolos/huecos, decoder closed-loop, baselines).
