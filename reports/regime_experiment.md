# Gating de régimen 4h: continuidad vs reversion

Generado: `2026-10-19T04:07:54+00:00`  
Train 2020-2023 · val 2024 · test 2025-2026 · SOL holdout.  
**Sin PnL, sin costos y sin Sharpe.**

## 1. Regímenes

Compresión = `bb_keltner_ratio` en el tercio inferior de la ventana rolling de 30d; expansión = tercio superior; normal = medio. El modelo especializado de compresión (`model_range`) y el de expansión (`model_trend`) se entrenan por separado; las hojas se eligen por Rank IC en validación.

| subset | regime | n | share |
|---|---|---|---|
| train | compression | 5,773 | 0.3299 |
| train | normal | 5,798 | 0.3313 |
| train | expansion | 5,929 | 0.3388 |
| val | compression | 1,308 | 0.3247 |
| val | normal | 1,350 | 0.3352 |
| val | expansion | 1,370 | 0.3401 |
| test_fit | compression | 2,407 | 0.3301 |
| test_fit | normal | 2,419 | 0.3317 |
| test_fit | expansion | 2,466 | 0.3382 |
| sol | compression | 1,229 | 0.3383 |
| sol | normal | 1,190 | 0.3276 |
| sol | expansion | 1,214 | 0.3342 |


Hojas: baseline=15, regime_all=7, compression=15, expansion=7, normal=15.


## 2. Rank IC por régimen (OOS 2025-2026)

| subset | regime | n | baseline_ic | regime_all_ic | gated_ic | delta_gated_vs_regall |
|---|---|---|---|---|---|---|
| test_fit | compression | 2,407 | 0.0707 | 0.0691 | 0.0353 | -0.0337 |
| test_fit | expansion | 2,466 | 0.0406 | 0.0389 | 0.0460 | 0.0072 |
| sol | compression | 1,229 | 0.0116 | -0.0041 | 0.0027 | 0.0068 |
| sol | expansion | 1,214 | -0.0087 | 0.0039 | -0.0128 | -0.0167 |


`baseline` = features de precio/State/Scale; `regime_all` = mismas filas con features de agotamiento y régimen; `gated` = modelo especializado por régimen. La columna delta compara gated contra regime_all (efecto de segmentar).


- Gating vs regime_all: test_fit compresión -0.0337, expansión +0.0072; SOL compresión +0.0068, expansión -0.0167. No hay una mejora consistente; en compresión BTC/ETH el gating destruye el IC del baseline.


## 3. Rachas >95%: continuación vs reversión

| subset | regime | hypothesis | n | realized_rate | model_hit | model_cont_rate | t_realized | t_model | pass_realized | pass_model |
|---|---|---|---|---|---|---|---|---|---|---|
| test_fit | expansion | continuation | 149 | 0.4966 | 0.5570 | 0.3221 | -0.0819 | 1.3927 | no | no |
| test_fit | compression | reversal | 71 | 0.4789 | 0.4789 | 0.5070 | -0.3560 | -0.3560 | no | no |
| sol | expansion | continuation | 82 | 0.5366 | 0.5000 | 0.3293 | 0.6626 | 0.0000 | no | no |
| sol | compression | reversal | 30 | 0.5667 | 0.5333 | 0.5000 | 0.7303 | 0.3651 | no | no |


`realized_rate`: en expansión es la tasa de continuación; en compresión la de reversión. `model_hit` es la accuracy del modelo gated clasificando continuación vs reversión (predice continuación si la racha es up y p>=0.5, o si es down y p<0.5). `pass_*` exige tasa > 58% y t > 2.0.


## 4. Conclusión

- **test_fit**: expansión continuación realizada 0.497 (t=-0.08), model_hit 0.557 (t=1.39), pass_realized=False, pass_model=False; compresión reversión realizada 0.479 (t=-0.36), model_hit 0.479 (t=-0.36), pass_realized=False, pass_model=False.

- **sol**: expansión continuación realizada 0.537 (t=0.66), model_hit 0.500 (t=0.00), pass_realized=False, pass_model=False; compresión reversión realizada 0.567 (t=0.73), model_hit 0.533 (t=0.37), pass_realized=False, pass_model=False.

- **Veredicto**: **no**. El gating no mejora el Rank IC de forma consistente (en compresión BTC/ETH cae de 0.0707 baseline a 0.0353 gated; en SOL la expansión empeora -0.0167). Ningún hit rate de continuación/reversión supera 58% con t>2: el mejor es expansión BTC/ETH con 55.7% y t=1.39. El régimen no agrega una regla explotable. Sin PnL ni Sharpe.
