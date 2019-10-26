# NeuralNetworks-Zoo

![NeuralNetworks](https://img.shields.io/badge/NeuralNetworks-11.3.5-orange.svg)
![Release Vision](https://img.shields.io/badge/Release-v0.3.x-ff69b4.svg)
![Models](https://img.shields.io/badge/Models-42-brightgreen.svg)
![Repo Size](https://img.shields.io/github/repo-size/GalAster/WLNet-ModelZoo.svg)



## Formats

You can download these awesome models in https://m.vers.site/NetModel/

There exist the following format:

- `*.WLNet` format

Standard Wolfram Neural Networks model, support for version upgrade sequences

Can be exported directly to ONNX format

- `*.WXF` format

If the model is more complicated, then the official function is a bit stretched.

This is the extended model using [DeepMath](https://github.com/Moe-Net/DeepMathFantasy), you must install `DeepMath` then you can use them normally.

```Mathematica
PacletInstall@"https://github.com/Moe-Net/DeepMathFantasy/releases/download/v0.1.0/DeepMath-0.1.0.paclet";
<< DeepMath`;
DeepMath`Tools`LayersRegister[];
```

This situation can also be exported directly to ONNX format.

But I don't offer a promise of backward compatibility

- `*.APP` format

If the model consists of multiple parts and uses complex operations, then this format is used

Unfortunately, this case cannot be converted to ONNX format.

## Request

If you really like a fantasy model, but this model doesn't have the ONNX format, then you can make a `Request` on the issue page. 

I will try my best to convert that model.


## Contribution

**All Pull Requests are welcome!** 

You can add readme, introduce interesting usage examples, build unit tests, fix wrong references, etc.

**Never upload images!** 

You can use an external link URL like https://sm.ms/ in markdown