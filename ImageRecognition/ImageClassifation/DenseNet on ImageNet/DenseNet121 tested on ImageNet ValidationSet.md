# DenseNet121 trained on ImageNet
![Task](https://img.shields.io/badge/Task-Classifation-Orange.svg)
![Size](https://img.shields.io/badge/Size-32.25%20MB-blue.svg)
![Accuracy](https://img.shields.io/badge/Accuracy-28.743%25-brightgreen.svg)
![Speed](https://img.shields.io/badge/Speed-9.372%20ms-ff69b4.svg)

Automatically generated on Mon 19 Nov 2018 15:24:48

## Network structure:
- Network Size: **32.25 MB**
- Parameters: **8 062 504**
- Nodes Count: **427**
- Speed: **9.372 ms/sample**
- Layers:
  - BatchNormalizationLayer: **121**
  - CatenateLayer: **58**
  - ConvolutionLayer: **120**
  - ElementwiseLayer: **121**
  - LinearLayer: **1**
  - PoolingLayer: **5**
  - SoftmaxLayer: **1**


## Accuracy Curve
![Classification Curve.png](https://i.loli.net/2018/11/19/5bf26bf8f0fa1.png)

![High Precision Classification Curve.png](https://i.loli.net/2018/11/19/5bf26bf8f1979.png)

## Main Indicator
  - Top-1: **28.7439%**
  - Top-2: **38.8480%**
  - Top-5: **52.1380%**
  - Top-25: **72.6640%**
  - LogLikelihood: **-186106.**
  - CrossEntropyLoss: **3.72211**
  - ProbabilityLoss: **0.389028**
  - MeanProbability: **20.4877%**
  - GeometricMeanProbability: **2.41828%**
  - VarianceProbability: **0.0927974**
  - ScottPi: **0.285821**
  - CohenKappa: **0.286727**
  - RejectionRate: **0.00000%**

![Accuracy Rejection Curve.png](https://i.loli.net/2018/11/19/5bf26bff8e3dc.png)

## Class Indicator
| Class | Count | TPRate | TNRate | FPRate | FNRate | F1Score |
|-------|-------|--------|--------|--------|--------|---------|
| Appenzeller | 50 | 0.00000% | 100.000% | 0.00000% | 100.000% | 0.00000 |
| Border terrier | 50 | 0.00000% | 100.000% | 0.00000% | 100.000% | 0.00000 |
| burrito | 50 | 0.00000% | 99.9979% | 0.00200% | 100.000% | 0.00000 |
| butcher shop | 50 | 0.00000% | 100.000% | 0.00000% | 100.000% | 0.00000 |
| chocolate sauce | 50 | 0.00000% | 99.9979% | 0.00200% | 100.000% | 0.00000 |
| cloak | 50 | 0.00000% | 99.9979% | 0.00200% | 100.000% | 0.00000 |
| horned rattlesnake | 50 | 2.00000% | 99.9699% | 0.03003% | 98.0000% | 0.03030 |
| Dutch oven | 50 | 0.00000% | 100.000% | 0.00000% | 100.000% | 0.00000 |
| grocery store | 50 | 0.00000% | 100.000% | 0.00000% | 100.000% | 0.00000 |
| Grifola frondosa | 50 | 0.00000% | 99.9979% | 0.00200% | 100.000% | 0.00000 |
| night snake | 50 | 0.00000% | 99.9979% | 0.00200% | 100.000% | 0.00000 |
| Irish wolfhound | 50 | 0.00000% | 99.9759% | 0.02402% | 100.000% | 0.00000 |
| Loafer | 50 | 0.00000% | 99.9979% | 0.00200% | 100.000% | 0.00000 |
| lumbermill | 50 | 0.00000% | 99.9959% | 0.00400% | 100.000% | 0.00000 |
| malinois | 50 | 0.00000% | 99.9979% | 0.00200% | 100.000% | 0.00000 |
| miniature poodle | 50 | 0.00000% | 99.9979% | 0.00200% | 100.000% | 0.00000 |
| overskirt | 50 | 0.00000% | 99.9939% | 0.00600% | 100.000% | 0.00000 |
| plastic bag | 50 | 32.0000% | 97.8638% | 2.13613% | 68.0000% | 0.02824 |
| power drill | 50 | 0.00000% | 100.000% | 0.00000% | 100.000% | 0.00000 |
| grille | 50 | 0.00000% | 100.000% | 0.00000% | 100.000% | 0.00000 |
| soup bowl | 50 | 0.00000% | 99.9959% | 0.00400% | 100.000% | 0.00000 |
| stone wall | 50 | 0.00000% | 100.000% | 0.00000% | 100.000% | 0.00000 |
| tiger cat | 50 | 2.00000% | 99.9739% | 0.02602% | 98.0000% | 0.03125 |
| velvet | 50 | 2.00000% | 99.9679% | 0.03203% | 98.0000% | 0.02985 |
| wok | 50 | 0.00000% | 99.9779% | 0.02202% | 100.000% | 0.00000 |

## Hard Class
![ConfusionMatrix.png](https://i.loli.net/2018/11/19/5bf26c00cc598.png)

## Evaluation Report
| Index | TestID | Result | Time | MemoryChange |
|-------|--------|--------|------|--------------|
| 1 | Dependency Check | Success | 2.10100 s | +5.09973 MB |
| 2 | Pre-define | Success | 0.00000 s | +0.00164 MB |
| 3 | GPU Warm-Up | Success | 4.54005 s | +53.4947 MB |
| 4 | Loading Model | Success | 0.38696 s | +34.1880 MB |
| 5 | Loading Data | Success | 0.22546 s | +17.8600 MB |
| 6 | Benchmark Test | Success | 767.633 s | +437.230 MB |
| 7 | Result Dump | Success | 3.59747 s | -4.26571 MB |
| 8 | Analyzing | Success | 1645.70 s | +40.6695 MB |
