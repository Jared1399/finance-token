# Barrido del codebook

Generado: `2026-10-19T01:36:24+00:00`  
Fingerprint: `b8be97370673cf71` · Timeframe: `15m`  
Train: 445,299 velas · Validacion: 64,482 velas  
Presupuesto de vocabulario: **10,240** tokens.

## Forma (individual)

Distancia euclidiana de cada vela a su centroide en `(o, c)`.

| k_s | p50 | p95 | p99 | muertos | H/Hmax | seg |
|---|---|---|---|---|---|---|
| 128 | 0.0336 | 0.0564 | 0.0645 | 0 | 0.9949 | 1.4302 |
| 256 | 0.0235 | 0.0392 | 0.0450 | 0 | 0.9939 | 2.8132 |
| 512 | 0.0164 | 0.0275 | 0.0318 | 0 | 0.9933 | 5.7671 |
| 1,024 | 0.0114 | 0.0192 | 0.0221 | 0 | 0.9924 | 11.2797 |


## Magnitud (individual)

Error relativo `|R_hat / R - 1|`.

| k_v | p50 | p95 | p99 | muertos | H/Hmax | seg |
|---|---|---|---|---|---|---|
| 8 | 0.0684 | 0.1563 | 0.2475 | 0 | 0.9005 | 2.5573 |
| 16 | 0.0365 | 0.0816 | 0.1332 | 0 | 0.9053 | 3.8530 |
| 32 | 0.0180 | 0.0414 | 0.0629 | 0 | 0.9274 | 7.4541 |
| 64 | 0.0089 | 0.0192 | 0.0347 | 0 | 0.9440 | 16.0111 |


## Combinaciones (error end-to-end)

`shape_p95` es la distancia media al centroide de forma; `close_p95` es el error del cierre reconstruido, ambos en unidades de R. El objetivo suma los dos: un codebook que solo minimiza el cierre puede destruir la morfologia, que es el proposito del tokenizador.

| k_s | k_v | vocab | shape_p95 | close_p50 | close_p95 | close_p99 | scale_p95 | objetivo |
|---|---|---|---|---|---|---|---|---|
| 128 | 8 | 1,027 | 0.0564 | 0.0409 | 0.1169 | 0.1688 | 0.1537 | 0.1732 |
| 128 | 16 | 2,051 | 0.0564 | 0.0318 | 0.0826 | 0.1071 | 0.0812 | 0.1389 |
| 128 | 32 | 4,099 | 0.0564 | 0.0282 | 0.0713 | 0.0864 | 0.0415 | 0.1277 |
| 128 | 64 | 8,195 | 0.0564 | 0.0272 | 0.0678 | 0.0823 | 0.0191 | 0.1241 |
| 256 | 8 | 2,051 | 0.0392 | 0.0347 | 0.1082 | 0.1617 | 0.1537 | 0.1474 |
| 256 | 16 | 4,099 | 0.0392 | 0.0254 | 0.0682 | 0.0930 | 0.0812 | 0.1073 |
| 256 | 32 | 8,195 | 0.0392 | 0.0210 | 0.0529 | 0.0646 | 0.0415 | 0.0920 |
| 256 | 64 | 16,387 | 0.0392 | 0.0195 | 0.0484 | 0.0579 | 0.0191 | 0.0876 |
| 512 | 8 | 4,099 | 0.0275 | 0.0305 | 0.1036 | 0.1599 | 0.1537 | 0.1312 |
| 512 | 16 | 8,195 | 0.0275 | 0.0207 | 0.0603 | 0.0870 | 0.0812 | 0.0878 |
| 512 | 32 | 16,387 | 0.0275 | 0.0158 | 0.0409 | 0.0528 | 0.0415 | 0.0685 |
| 512 | 64 | 32,771 | 0.0275 | 0.0138 | 0.0350 | 0.0428 | 0.0191 | 0.0626 |
| 1,024 | 8 | 8,195 | 0.0192 | 0.0281 | 0.1018 | 0.1570 | 0.1537 | 0.1210 |
| 1,024 | 16 | 16,387 | 0.0192 | 0.0179 | 0.0561 | 0.0834 | 0.0812 | 0.0753 |
| 1,024 | 32 | 32,771 | 0.0192 | 0.0124 | 0.0337 | 0.0468 | 0.0415 | 0.0529 |
| 1,024 | 64 | 65,539 | 0.0192 | 0.0101 | 0.0256 | 0.0321 | 0.0191 | 0.0448 |


## Recomendacion

Menor `objetivo` con `vocab <= 10,240`: **K_s=512 × K_v=16** → vocabulario **8,195** (forma p95=0.0275, cierre p95=0.0603, objetivo=0.0878 R).


![Barrido](figures/sweep.png)


Leer la tabla de combinaciones, no solo las curvas individuales: repartir el mismo presupuesto entre forma y magnitud no da lo mismo. El error del cierre depende del cuerpo (forma) multiplicado por la escala (magnitud), asi que ambos codebooks compiten por el mismo presupuesto.
