# LeNet Test on MNIST TestSet
Automatically generated on Sat 17 Nov 2018 19:36:22

## Network structure: 
- Network Size: **1.72432 MB**
- Parameters: **431 080**
- Nodes Count: **11**
- Layer Statistics
  - ConvolutionLayer: **2**
  - ElementwiseLayer: **3**
  - PoolingLayer: **2**
  - FlattenLayer: **1**
  - LinearLayer: **2**
  - SoftmaxLayer: **1**


## Accuracy Curve
![Classification Curve.png](https://i.loli.net/2018/11/17/5bf004151b12a.png)
![High Precision Classification Curve.png](https://i.loli.net/2018/11/17/5bf0041535eab.png)

## Main Indicator
  - Top-1: **0.9848**
  - Top-2: **0.9973**
  - Top-3: **0.9991**
  - Top-5: **0.9999**
  - LogLikelihood: **-659.657**
  - CrossEntropyLoss: **0.0659657**
  - ProbabilityLoss: **0.00522323**
  - MeanProbability: **0.981506**
  - GeometricMeanProbability: **0.936163**
  - VarianceProbability: **0.0130536**
  - ScottPi: **0.983105**
  - CohenKappa: **0.983105**
  - RejectionRate: **0**

![Accuracy Rejection Curve.png](https://i.loli.net/2018/11/17/5bf00415503a1.png)

## Class Indicator
| Class | Count | TPRate | TNRate | FPRate | FNRate | F1Score |
|-------|-------|--------|--------|--------|--------|---------|
| 0 | 980 | 0.993878 | 0.998559 | 0.00144124 | 0.00612245 | 0.990341 |
| 1 | 1135 | 0.994714 | 0.999098 | 0.000902425 | 0.00528634 | 0.993838 |
| 2 | 1032 | 0.974806 | 0.998996 | 0.00100357 | 0.0251938 | 0.982902 |
| 3 | 1010 | 0.989109 | 0.99822 | 0.00177976 | 0.0108911 | 0.986667 |
| 4 | 982 | 0.977597 | 0.999446 | 0.000554447 | 0.0224033 | 0.986133 |
| 5 | 892 | 0.988789 | 0.997145 | 0.00285463 | 0.0112108 | 0.98 |
| 6 | 958 | 0.974948 | 0.999336 | 0.00066357 | 0.0250522 | 0.984194 |
| 7 | 1028 | 0.983463 | 0.996879 | 0.00312082 | 0.016537 | 0.978229 |
| 8 | 974 | 0.9846 | 0.998449 | 0.00155107 | 0.0154004 | 0.985105 |
| 9 | 1009 | 0.985134 | 0.996997 | 0.003003 | 0.0148662 | 0.97931 |

## Hard Class
![ConfusionMatrix.png](https://i.loli.net/2018/11/17/5bf0041545c4e.png)

## Evaluation Report
| Index | TestID | Result | Time | MemoryChange |
|-------|--------|--------|------|--------------|
| 1 | Pre-define | Success | 0 | 0.00204 |
| 2 | GPU Warm-Up | Success | 8.42563 | 53.2446 |
| 3 | Loading Model | Success | 0.0379246 | 1.76696 |
| 4 | Loading Data | Success | 2.02956 | 34.5418 |
| 5 | Benchmark Testing | Success | 18.9374 | 16.5781 |
| 6 | Result Dumping | Success | 0.19564 | -1.12314 |
| 7 | Analyzing | Success | 7.94505 | 13.9795 |

