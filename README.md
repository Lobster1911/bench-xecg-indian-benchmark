# bench-xecg-indian-benchmark

Research fork of BenchECG focused on benchmarking and adapting ECG foundation models on Indian ECG datasets.

This repository extends the original BenchECG framework to:
- benchmark existing ECG foundation models on Indian ECG data,
- evaluate population-specific ECG representation learning,
- develop and pretrain ECG foundation models using Indian clinical ECG datasets,
- investigate downstream cardiovascular tasks relevant to Indian populations.

Current models under evaluation include:
- xECG
- ECGFounder
- ECG-JEPA
- ST-MEM (planned)

Downstream tasks include:
- arrhythmia classification,
- myocardial infarction (MI) detection,
- ST-elevation myocardial infarction (STEMI) detection,
- acute coronary syndrome (ACS) analysis,
- ST-segment abnormality detection,
- left ventricular hypertrophy (LVH) classification.

This work builds upon the original BenchECG repository:
https://github.com/dlaskalab/bench-xecg
