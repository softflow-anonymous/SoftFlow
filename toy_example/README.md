# SoftFlow: Probabilistic Framework for Normalizing Flow on Manifolds


## Requirements
- python 3.8.3
- matplotlib
- pytorch 1.5
- sklearn
- torchdiffeq

## Training
```train
python train.py --data 2spirals_1d --dims 64-64-64 --std_min 0.0 --std_max 0.1 --std_weight 2
```
> Datasets: 2spirals_1d, swissroll_1d, circles_1d, 2sines_1d, circles_1d

## Generation

To generate samples from the model, run:

```generate
python generate1.py --data 2spirals_1d --load_path results/2spirals_1d/SoftFlow/checkpt.pth
```

or you can use the pretrained model
```generate
python generate1.py --data 2spirals_1d --load_path pretrained_toy/2spirals_1d/checkpt.pth
```

To generate samples with the different noise distributions, run:

```generate
python generate2.py --data 2spirals_1d --load_path results/2spirals_1d/SoftFlow/checkpt.pth
```

or you can use the pretrained model
```generate
python generate2.py --data 2spirals_1d --load_path pretrained_toy/2spirals_1d/checkpt.pth
```

## Pre-trained models
- Click this [link](https://drive.google.com/open?id=19VzLEOZkr8swP24KUu7EikDpwsy6SX3y) to download the pre-trained models.
```
unzip pretrained_toy.zip
```

## Results
1. Samples from SoftFlow (`generate1.py`)
|               | 2spirals | swissroll | circles | 2sines | target |
|:-------------:|:--------:|:---------:|:-------:|:------:|:------:|
|      Data     |          |           |         |        |        |
| SoftPointFlow |          |           |         |        |        |
<p align="center">
    <img src="generate1/2spirals_1d/sample_softflow.png" height=450/>
</p>

## References
- FFJORD: https://github.com/rtqichen/ffjord
- PointFlow: https://github.com/stevenygd/PointFlow