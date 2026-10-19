# Experimento 4h: flujo de ordenes y calibracion

Generado: `2026-10-19T03:46:41+00:00`  
Train: 2020-01-01 a 2023 · val 2024 · test 2025-2026. Baseline = features de precio/State/Scale; Flow = baseline + taker/CVD + funding + OI.  
**Sin PnL, sin costos y sin Sharpe en este experimento.**

## 1. Datos

| item | valor |
|---|---|
| train rows | 17,500 |
| val rows | 4,028 |
| test BTC/ETH rows | 7,292 |
| SOL holdout rows | 3,633 |
| train has_funding | 1.0000 |
| train has_oi | 0.6771 |
| test has_funding | 1.0000 |
| test has_oi | 0.9997 |
| SOL has_funding | 1.0000 |
| SOL has_oi | 1.0000 |


Modelos: baseline leaves=15, flow leaves=7 (elegidos por Rank IC en validacion). Features de flujo: 26.


## 2. Rank IC (Spearman)

| subset | period | n | baseline_ic | flow_ic | delta | flow_p |
|---|---|---|---|---|---|---|
| test_fit | overall | 7,292 | 0.0423 | 0.0406 | -0.0017 | 5.30e-04 |
| test_fit | 2025 | 4,019 | 0.0521 | 0.0344 | -0.0178 | 0.0293 |
| test_fit | 2026 | 3,273 | 0.0288 | 0.0463 | 0.0175 | 0.0081 |
| sol | overall | 3,633 | 0.0137 | 0.0134 | -3.49e-04 | 0.4195 |
| sol | 2025 | 2,005 | 0.0170 | 0.0100 | -0.0070 | 0.6546 |
| sol | 2026 | 1,628 | 0.0106 | 0.0198 | 0.0092 | 0.4247 |


IC entre `P(subida)` cruda y el retorno forward 4h. `test_fit` = BTC/ETH; `sol` = holdout. La calibracion es monotona y no cambia el ranking.


## 3. ECE (calibracion)

| subset | model | raw_ece | isotonic_ece | platt_ece | chosen |
|---|---|---|---|---|---|
| test_fit | baseline | 0.0432 | 0.0157 | 0.0207 | platt |
| sol | baseline | 0.0450 | 0.0282 | 0.0273 | platt |
| test_fit | flow | 0.0394 | 0.0226 | 0.0144 | platt |
| sol | flow | 0.0463 | 0.0320 | 0.0219 | platt |


ECE en 15 bins sobre test. `chosen` es el metodo elegido por ECE con 5-fold CV en validacion y un margen de estabilidad de 0.005 a favor de Platt. Isotonic in-sample puede dar ECE=0 y sobreajustar. La columna raw es el modelo sin calibrar.


### Mapa de calibracion flow - test_fit

| bin | n | mean_p | up_rate | gap |
|---|---|---|---|---|
| 0.30-0.40 | 21 | 0.3904 | 0.4286 | 0.0382 |
| 0.40-0.50 | 2,765 | 0.4687 | 0.4738 | 0.0051 |
| 0.50-0.60 | 4,292 | 0.5400 | 0.5205 | -0.0195 |
| 0.60-0.70 | 214 | 0.6152 | 0.5794 | -0.0358 |


ECE: 0.0144


### Mapa de calibracion flow - sol

| bin | n | mean_p | up_rate | gap |
|---|---|---|---|---|
| 0.30-0.40 | 7 | 0.3937 | 0.4286 | 0.0349 |
| 0.40-0.50 | 1,300 | 0.4727 | 0.4869 | 0.0143 |
| 0.50-0.60 | 2,242 | 0.5392 | 0.5138 | -0.0253 |
| 0.60-0.70 | 84 | 0.6159 | 0.5357 | -0.0802 |


ECE: 0.0219


## 4. Hit Rate por decil (flow calibrado) - test_fit

| dec | n | mean_p | up_rate | hit_rate |
|---|---|---|---|---|
| P0 | 730 | 0.4348 | 0.4712 | 0.5288 |
| P1 | 729 | 0.4651 | 0.4554 | 0.5446 |
| P2 | 729 | 0.4824 | 0.4801 | 0.5199 |
| P3 | 729 | 0.4960 | 0.4979 | 0.5144 |
| P4 | 729 | 0.5087 | 0.5144 | 0.5144 |
| P5 | 730 | 0.5210 | 0.5164 | 0.5164 |
| P6 | 729 | 0.5336 | 0.4829 | 0.4829 |
| P7 | 729 | 0.5477 | 0.5199 | 0.5199 |
| P8 | 729 | 0.5640 | 0.5336 | 0.5336 |
| P9 | 729 | 0.5942 | 0.5706 | 0.5706 |


## 4. Hit Rate por decil (flow calibrado) - sol

| dec | n | mean_p | up_rate | hit_rate |
|---|---|---|---|---|
| P0 | 364 | 0.4429 | 0.4753 | 0.5247 |
| P1 | 363 | 0.4722 | 0.5207 | 0.4793 |
| P2 | 363 | 0.4871 | 0.4986 | 0.5014 |
| P3 | 364 | 0.4990 | 0.4615 | 0.5467 |
| P4 | 363 | 0.5106 | 0.4821 | 0.4821 |
| P5 | 363 | 0.5220 | 0.5344 | 0.5344 |
| P6 | 364 | 0.5336 | 0.5412 | 0.5412 |
| P7 | 363 | 0.5468 | 0.5096 | 0.5096 |
| P8 | 363 | 0.5630 | 0.5069 | 0.5069 |
| P9 | 363 | 0.5917 | 0.5152 | 0.5152 |


Deciles por `P(subida)` calibrada. `hit_rate` = acierto de la direccion implicita (`p >= 0.5` up, `p < 0.5` down). No hay retornos ni PnL en esta tabla.


## 5. Conclusion

- Rank IC test BTC/ETH: baseline **0.0423** → flow **0.0406** (delta -0.0017).
- Rank IC SOL holdout: **0.0134**.
- Objetivo 0.08+: **no alcanzado**.


La calibracion corrige la escala de probabilidad, no el ranking. Si el IC no sube, el flujo no agrego informacion direccional out-of-sample.
