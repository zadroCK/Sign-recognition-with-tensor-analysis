# Sign-recognition-with-tensor-analysis

This repository was created on the occasion of the development project for [Introduction to complex data search](http://www.pmf.unizg.hr/math/predmet/uuspp_a) at the [Faculty of Science](http://www.pmf.unizg.hr/math) in Zagreb.

## Overview

The goal of this project is to create model that recognize two different signs from sign language.

## Preparing a dataset
Firstly we used [python mediapipe](https://google.github.io/mediapipe/getting_started/python.html) to find area where are the hands.
We get coordinates of hands area and then we append it to file name using simple script.
### Example:
```
05844.mp4 -> 05844_123_50_172_237.mp4
```

## Models
1. In the beginning, convert all the videos into tensors: one sign one tensor.
Shortly, we convert each video into tensor and then concatenate all videos from each sign in one tensor. 
_So, in our example with only two signs we get two tensors._
2. _depend on model_
3. HOSVD of each tensors
4. 

### Model without HOG
2. nothing

### Model with HOG
2. Use [HOG descriptor](https://en.wikipedia.org/wiki/Histogram_of_oriented_gradients) on each frame (of every video) with only one direction (beacuse of dimension problem)




## Results

## Wih

## Repository structure

```
|- code  (final code)
|- videos  (videos used in project)
```