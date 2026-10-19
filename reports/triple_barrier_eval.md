# Triples Barreras con Monte Carlo autoregresivo

Generado: `2026-10-19T06:40:22+00:00`  
Checkpoint epoch: 3 · val CE: 8.6152  
M=500 trayectorias/punto · N=48 velas (12h) · stride=16 velas · TP=+1.5 ATR / SL=-1 ATR · costos=10 bps.

**Piloto estratificado**: 167 puntos por símbolo y año (1,002 puntos con Monte Carlo). El grid completo con stride=16 son 10,935 puntos; el run completo no se ejecutó por costo de GPU. La selección es aleatoria dentro de cada año y símbolo; las tasas base del piloto pueden diferir de las completas por muestreo.


La generación es closed-loop: se recompone ATR (Wilder), State (SMA/σ de 100) y Scale (log ATR/mediana 30d) después de cada vela. Las trayectorias se detienen cuando ya se resolvió la primera barrera long y short, para ahorrar cómputo. El target real usa la primera barrera en las 48 velas reales; si una vela toca ambas, se resuelve con la morfología (alcista: low antes que high; bajista: high antes que low; doji: la más cercana al open).


`P_hat(Long)` = fracción de trayectorias que tocan +1.5 ATR antes que -1.0 ATR. `P_hat(Short)` = fracción que toca -1.5 ATR antes que +1.0 ATR (TP short antes de SL short). Las dos probabilidades se calculan por separado; no suman 1.


## 1. Datos y target real

| symbol | year | n | long_tp_rate | short_tp_rate | long_timeout_rate | mean_long_time | atr_bps |
|---|---|---|---|---|---|---|---|
| BTCUSDT | 2,025 | 2,010 | 0.4055 | 0.3930 | 0.0025 | 5.5637 | 29.1734 |
| BTCUSDT | 2,026 | 1,635 | 0.3914 | 0.3994 | 0.0055 | 5.8318 | 29.7059 |
| ETHUSDT | 2,025 | 2,010 | 0.4214 | 0.3990 | 0.0030 | 5.7637 | 51.3534 |
| ETHUSDT | 2,026 | 1,635 | 0.3884 | 0.3963 | 0.0043 | 5.8538 | 39.9773 |
| SOLUSDT | 2,025 | 2,010 | 0.4095 | 0.4159 | 4.98e-04 | 5.3413 | 60.2959 |
| SOLUSDT | 2,026 | 1,635 | 0.3853 | 0.3920 | 0.0049 | 5.8294 | 46.2665 |


Base real calculada sobre **todos** los puntos del grid stride=16 (sin Monte Carlo). El target Long es 1 si el TP Long (+1.5 ATR) se toca primero; 0 si el SL Long (-1.0 ATR) o timeout. El target Short es análogo con -1.5/+1.0 ATR.


## 2. Rank IC OOS (P_hat Long vs target real)

| subset | n | rank_ic | base_rate | mean_p |
|---|---|---|---|---|
| BTC/ETH 2025 | 334 | -0.0498 | 0.4311 | 0.3943 |
| BTC/ETH 2026 | 334 | -0.0115 | 0.4162 | 0.3896 |
| BTC/ETH all | 668 | -0.0301 | 0.4237 | 0.3919 |
| SOL 2025 | 167 | 0.0906 | 0.4491 | 0.4024 |
| SOL 2026 | 167 | 0.0406 | 0.3832 | 0.3987 |
| SOL all | 334 | 0.0707 | 0.4162 | 0.4005 |


Rank IC de Spearman entre `P_hat(Long)` y el target binario de barrera. El IC mide si las trayectorias ordenan los puntos por probabilidad real de TP, no si aciertan el signo de cada vela.


## 3. Calibración y deciles

| subset | n | ece | brier | logloss | base_rate | mean_p |
|---|---|---|---|---|---|---|
| BTC/ETH all | 668 | 0.0399 | 0.2468 | 0.6870 | 0.4237 | 0.3919 |
| SOL all | 334 | 0.0173 | 0.2419 | 0.6768 | 0.4162 | 0.4005 |


### BTC/ETH all

| decile | n | mean_p | hit_rate |
|---|---|---|---|
| P0 | 67 | 0.3446 | 0.4179 |
| P1 | 67 | 0.3673 | 0.5224 |
| P2 | 67 | 0.3771 | 0.3134 |
| P3 | 67 | 0.3843 | 0.3731 |
| P4 | 67 | 0.3898 | 0.4925 |
| P5 | 67 | 0.3953 | 0.4776 |
| P6 | 67 | 0.4019 | 0.5075 |
| P7 | 67 | 0.4089 | 0.3582 |
| P8 | 66 | 0.4171 | 0.3788 |
| P9 | 66 | 0.4342 | 0.3939 |


ECE = suma de |mean_p - hit_rate| ponderada por n. Base rate = 0.4237; P_hat medio = 0.3919.


### SOL all

| decile | n | mean_p | hit_rate |
|---|---|---|---|
| P0 | 34 | 0.3615 | 0.4118 |
| P1 | 34 | 0.3765 | 0.4412 |
| P2 | 34 | 0.3832 | 0.2941 |
| P3 | 34 | 0.3902 | 0.3235 |
| P4 | 33 | 0.3968 | 0.4848 |
| P5 | 33 | 0.4031 | 0.3030 |
| P6 | 33 | 0.4105 | 0.4848 |
| P7 | 33 | 0.4170 | 0.5455 |
| P8 | 33 | 0.4253 | 0.4242 |
| P9 | 33 | 0.4439 | 0.4545 |


ECE = suma de |mean_p - hit_rate| ponderada por n. Base rate = 0.4162; P_hat medio = 0.4005.


## 4. Colas de alta convicción (Top 10%)

| subset | side | n | mean_p | hit_tp | ev_bps | atr_bps |
|---|---|---|---|---|---|---|
| BTC/ETH 2025 | long | 34 | 0.4356 | 0.4412 | -6.0780 | 40.6718 |
| BTC/ETH 2025 | short | 34 | 0.4196 | 0.3529 | -0.9589 | 40.5130 |
| BTC/ETH 2026 | long | 34 | 0.4304 | 0.3529 | -22.9718 | 36.9101 |
| BTC/ETH 2026 | short | 34 | 0.4227 | 0.3824 | -11.0114 | 28.2176 |
| BTC/ETH all | long | 67 | 0.4340 | 0.3881 | -13.2139 | 37.9124 |
| BTC/ETH all | short | 67 | 0.4214 | 0.3731 | -5.8104 | 34.2517 |
| SOL 2025 | long | 17 | 0.4444 | 0.4706 | -12.5455 | 53.3963 |
| SOL 2025 | short | 17 | 0.4155 | 0.4118 | -0.6167 | 56.5112 |
| SOL 2026 | long | 17 | 0.4407 | 0.2941 | -23.5987 | 49.1699 |
| SOL 2026 | short | 17 | 0.4211 | 0.4118 | -1.1899 | 43.2379 |
| SOL all | long | 34 | 0.4435 | 0.5000 | -0.3225 | 52.6006 |
| SOL all | short | 34 | 0.4183 | 0.3824 | -5.2421 | 49.5930 |


`hit_tp` es la tasa real de toque de TP en los puntos seleccionados por mayor P_hat. `ev_bps` usa el resultado real: TP +1.5 ATR, SL -1.0 ATR, timeout al cierre real de la vela 48, menos 10 bps de costos. No es PnL de cartera ni Sharpe; es el valor esperado unitario de la selección.


## 5. Lectura y veredicto

- **No hay consistencia entre fit y holdout.** BTC/ETH all: IC **-0.0301** (2025 -0.0498, t=-0.91; 2026 -0.0115, t=-0.21). SOL all: IC **+0.0707** (2025 +0.0906, t=1.17; 2026 +0.0406, t=0.52). Ningún IC es significativo al 5%; el mejor, SOL 2025, queda lejos de t=2.
- **Calibración global ≠ discriminación.** ECE 0.0399 en BTC/ETH y 0.0173 en SOL, pero `P_hat` medio (0.392/0.400) queda por debajo de la tasa base (0.424/0.416) y los deciles no son monótonos (BTC/ETH: P1=0.522, P2=0.313, P9=0.394). Un ECE bajo solo dice que la probabilidad está centrada; no que ordene los puntos.
- **Colas.** Top 10% long BTC/ETH: hit 0.388 vs base 0.424, EV **-13.21 bps**; short: hit 0.373, EV **-5.81 bps**. SOL long: hit 0.500 vs base 0.416, EV **-0.32 bps** (el bruto positivo queda anulado por los 10 bps); short: hit 0.382, EV **-5.24 bps**. SOL 2026 long: hit 0.294, EV **-23.60 bps**. No hay una cola robusta con EV neto positivo.
- **Path dependency: no.** Las trayectorias reproducen razonablemente la tasa base agregada, pero no ordenan los puntos ni capturan la asimetría de camino necesaria para las barreras. No hay evidencia para pasar a un modelo direccional o gastar GPU en esta línea.
- **Alcance.** Piloto aleatorio de 1,002 puntos (167 por símbolo y año) con M=500; el grid completo costaría ~16 h. Las colas tienen n=17-67, así que el EV tiene ruido muestral. Sin PnL de cartera ni Sharpe: el EV es unitario por selección.

Los deciles se construyen por rango y pueden cortar empates de `P_hat` (M=500 produce valores discretos); por eso la tabla de deciles y el ECE de bins de igual ancho no deben leerse como una curva de calibración fina.
