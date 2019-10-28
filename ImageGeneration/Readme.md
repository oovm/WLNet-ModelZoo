



### Attention



w x + b




### 微调技巧:

#### 消除 Tanh

```mathematica
(Tanh[x] + 1)/2 == LogisticSigmoid[2x]
```

#### 消除系数

这里有个 $2$ 还是很难看, 如果最后一层是卷积的话可以合并掉这个

$k (w x + b) + s = k w + (k b + s)$
