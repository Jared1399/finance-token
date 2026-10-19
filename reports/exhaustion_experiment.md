# Agotamiento de momentum / reversion a la media (4h)

Generado: `2026-10-19T03:58:44+00:00`  
Train 2020-2023 · val 2024 · test 2025-2026 · SOL holdout.  
**Sin PnL, sin costos y sin Sharpe.**

## 1. Features

Streak length y percentil rolling 2000; z-return 1h/4h/8h; Kaufman efficiency ratio 4h/8h; distancia a SMA100 y VWAP 24h normalizadas por ATR; aceleracion del momentum 4h. Todas causales.

| item | valor |
|---|---|
| train rows | 17,500 |
| val rows | 4,028 |
| test BTC/ETH rows | 7,292 |
| SOL holdout rows | 3,633 |
| exhaustion features | 11 |


## 2. Rank IC (Spearman)

| subset | period | n | baseline_ic | exhaustion_ic | delta | exh_p |
|---|---|---|---|---|---|---|
| test_fit | overall | 7,292 | 0.0423 | 0.0449 | 0.0026 | 1.27e-04 |
| test_fit | 2025 | 4,019 | 0.0521 | 0.0518 | -3.38e-04 | 0.0010 |
| test_fit | 2026 | 3,273 | 0.0288 | 0.0353 | 0.0065 | 0.0434 |
| sol | overall | 3,633 | 0.0137 | 0.0093 | -0.0044 | 0.5745 |
| sol | 2025 | 2,005 | 0.0170 | 0.0140 | -0.0030 | 0.5313 |
| sol | 2026 | 1,628 | 0.0106 | 0.0028 | -0.0078 | 0.9109 |


IC entre `P(subida)` cruda y retorno forward 4h. Platt es monotono y no cambia el ranking.


## 3. ECE (Platt)

| subset | model | raw_ece | platt_ece |
|---|---|---|---|
| test_fit | baseline | 0.0432 | 0.0207 |
| sol | baseline | 0.0450 | 0.0273 |
| test_fit | exhaustion | 0.0370 | 0.0189 |
| sol | exhaustion | 0.0480 | 0.0212 |


ECE en 15 bins sobre test. Platt se ajusta en validacion 2024.


## 4. Test BTC/ETH


### Hit Rate por decil (exhaustion + Platt)

| dec | n | mean_p | up_rate | hit_rate |
|---|---|---|---|---|
| P0 | 730 | 0.4375 | 0.4863 | 0.5137 |
| P1 | 729 | 0.4703 | 0.4829 | 0.5171 |
| P2 | 729 | 0.4857 | 0.4376 | 0.5624 |
| P3 | 729 | 0.4971 | 0.5075 | 0.4856 |
| P4 | 729 | 0.5077 | 0.5226 | 0.5226 |
| P5 | 730 | 0.5182 | 0.4534 | 0.4534 |
| P6 | 729 | 0.5296 | 0.5117 | 0.5117 |
| P7 | 729 | 0.5428 | 0.5267 | 0.5267 |
| P8 | 729 | 0.5590 | 0.5446 | 0.5446 |
| P9 | 729 | 0.5903 | 0.5693 | 0.5693 |


### Colas top/bottom 5% (exhaustion + Platt)

| tail | n | mean_p | up_rate | hit_rate | mean_fwd_bps | t |
|---|---|---|---|---|---|---|
| bottom5 | 364 | 0.4232 | 0.4670 | 0.5330 | -2.3331 | -0.4323 |
| top5 | 364 | 0.6036 | 0.5824 | 0.5824 | 6.7940 | 1.0061 |


### Rachas >95%

| subset | n | streak_mean | mean_fwd_bps | up_rate | reversal_rate | exh_hit | t |
|---|---|---|---|---|---|---|---|
| all | 7,292 | — | 0.1988 | 0.5043 | — | 0.5207 | 0.1394 |
| high_up | 154 | 0.9787 | 2.5384 | 0.4805 | 0.5195 | 0.5974 | 0.2385 |
| high_down | 162 | 0.9786 | 14.5160 | 0.5185 | 0.5185 | 0.4877 | 1.9307 |


## 5. SOL holdout


### Hit Rate por decil (exhaustion + Platt)

| dec | n | mean_p | up_rate | hit_rate |
|---|---|---|---|---|
| P0 | 364 | 0.4441 | 0.4918 | 0.5082 |
| P1 | 363 | 0.4750 | 0.5014 | 0.4986 |
| P2 | 363 | 0.4898 | 0.4793 | 0.5207 |
| P3 | 364 | 0.5004 | 0.5000 | 0.4835 |
| P4 | 363 | 0.5100 | 0.5510 | 0.5510 |
| P5 | 363 | 0.5201 | 0.4656 | 0.4656 |
| P6 | 364 | 0.5307 | 0.5027 | 0.5027 |
| P7 | 363 | 0.5432 | 0.5344 | 0.5344 |
| P8 | 363 | 0.5594 | 0.5041 | 0.5041 |
| P9 | 363 | 0.5897 | 0.5152 | 0.5152 |


### Colas top/bottom 5% (exhaustion + Platt)

| tail | n | mean_p | up_rate | hit_rate | mean_fwd_bps | t |
|---|---|---|---|---|---|---|
| bottom5 | 181 | 0.4304 | 0.4530 | 0.5470 | -5.5159 | -0.5521 |
| top5 | 181 | 0.6017 | 0.5138 | 0.5138 | 2.8952 | 0.2591 |


### Rachas >95%

| subset | n | streak_mean | mean_fwd_bps | up_rate | reversal_rate | exh_hit | t |
|---|---|---|---|---|---|---|---|
| all | 3,633 | — | -0.7084 | 0.5045 | — | 0.5084 | -0.2696 |
| high_up | 88 | 0.9784 | 5.6537 | 0.5114 | 0.4886 | 0.6023 | 0.3388 |
| high_down | 80 | 0.9787 | -7.2344 | 0.5250 | 0.5250 | 0.4375 | -0.3262 |


## 6. Conclusion

- **IC**: test BTC/ETH 0.0423 → 0.0449 (delta +0.0026); SOL 0.0093 (delta -0.0044). La mejora es marginal y no transfiere.

- **ECE**: exhaustion raw 0.0370 → Platt 0.0189 en test; SOL raw 0.0480 → Platt 0.0212. La calibración funciona; el ranking no cambia.

- **Hit rate**: P9 test 56.9%, SOL 51.5%; los deciles no son monotonos (el P2/P8 test tambien aciertan), asi que no hay una escalera de confianza.

- **Colas**: test top5 6.8 bps (t=1.01); bottom5 -2.3 bps. El top5 acierta 58.2%, pero el retorno medio y el t-stat son debiles.

- **Rachas >95%**: test high_up mean 2.5 bps, reversal 51.9%, t=0.24; high_down mean 14.5 bps, reversal 51.9%, t=1.93. En SOL, high_up 5.7 bps y high_down -7.2 bps. No hay reversion efectiva robusta: la tasa de reversion ronda 50% y los t-stats no alcanzan significancia (el high_down de BTC/ETH es sugestivo, t=1.93, pero no concluyente).

- **Veredicto**: las features de agotamiento no mejoran el Rank IC de forma robusta (+0.0026 test, -0.0044 SOL). Las rachas >95% no muestran una reversion explotable. No justifican un modelo de secuencia ni GPU; Platt sí mejora la calibración, pero no crea señal. Sin PnL ni Sharpe.
