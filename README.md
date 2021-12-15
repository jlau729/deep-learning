# CSE 490L: Deep Learning Project

## Project

### Abstract
This project focuses on Time Series Weather Forecasting based on the Jena climate dataset by the Max Planck Institute for Biogeochemistry. Specifically, in this project, the goal is to predict the a future temperature based on some amount of past data. As there are different types of network architectures that can be used to approach this, a few were selected and evaluated to compare them. 

### Video

## Introduction
The goal of this project is to explore different network architectures that can be used for time series weather forecasting. This problem seemed interesting for a number of reasons:
- It involved tabular data. The datasets used in this course are usually already processed into NumPy arrays or Tensor forms, so interacting with the pandas Dataframe library was an opportunity to explore a new type of dataset.
- It builds off of the previous RNN and NLP assignments in this course. Specifically, I didn't get a chance to try out Transformers for the NLP homework, so I wanted to do a project that allowed me to test that out.
- Weather forecasting is super complicated! I didn't realize this at first, but there are so many aspects of forecasting such as wind speeds and humidity and pressure that go into predicting the weather. And even then, what exactly consistutes weather? If it's sunny or rainy? If it's hot or cold? All of the above or something else? I only focused on temperature in this project, but I think this is a great example of a problem that is difficult to approach due to its complexity and the various ways results can be interpreted.

The models I compared were:
- Baseline
- LSTM RNN
- GRU RNN
- Transformer
- Neural Network with Linear Layers

# Related Work

Here's some tutorials and other resources I used for this. It was kind of rough though because I was using PyTorch and these were in TensorFlow or Keras, so there was a lot of information that I had to fill in for myself.
- https://www.tensorflow.org/tutorials/structured_data/time_series
- https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html
- https://keras.io/examples/timeseries/timeseries_transformer_classification/

In terms of things other people are doing, I saw something about a weather forecasting neural network called MetNet by Google. It seems that it uses self-attention and convolutions because the input to it is a patch containing the weather information, rather than the data points in the Jena climate dataset. So instead of using data points of temperature, humidity, wind speeds, pressure, and more to represent weather, they had the satellite images with the weather events marked on it, and then fed that into a CNN.

## Approach

### Goal
Given the past 12 hours of weather data, predict the temperature 12 hours into the future.

### Dataset

#### Cleaning
The first part of this was cleaning the dataset. The dataset consists of consecutive samples gathered in 10 minute intervals. Each sample had these features:
- Date Time / time the sample was collected
- p (mbar) / pressure
- T (degC) / temperature in Celsius
- Tpot (K) / temperature in Kelvin
- Tdew (degC) / temperature in Celsius relative to humidity
- rh (%) / relative humidity
- VPmax (mbar) / saturation vapor pressure
- VPact (mbar) / vapor pressure
- VPdef (mbar) vapor pressure deficit
- sh (g/kg) / specific humidity
- H2OC (mmol/mol) / water vapor concentration
- rho (g/m^3) / airtight
- wv (m/s) / wind speed
- max. wv (m/s) / max wind speed
- wd (deg) / wind direction in degrees

The first to do was to select a subset of samples. I only wanted hour intervals, so I sliced the dataset and selected one sample from each hour interval.

The next thing was to clean the data and to add more features that would be more interpretable in the model. This involved getting rid of:
- wv (m/s)
- max. wv (m/s)
- wd (deg)
- Date Time

and turning them into vectors. Date Time also had to be converted from a string to a number so that the model could interpret it. The end result was a dataset that had all features except the three listed above, plus these:

- wx (m/s) / wind speed x-component
- wy (m/s) / wind speed y-component
- max. wx (m/s) / max wind speed x-component
- max. wy (m/s) / max wind speed y-component
- Day x / days x-component
- Day y / days y-component
- Year x / years x-component
- Year y / years y-component
This ended up being a dataset where each point had 19 features.
The training data was also normalized, and the evaluation and test sets were normalized according to the training standard deviation and mean.

#### WeatherDataset

The dataset itself consists of inputs and labels. 

Each input is a Tensor of shape (input length, 19), where input length is the number of past points we want to0 use to predict the future points. In this project, the input length for all models was 12 (so we were using the past 12 hours worth of data).

Each target is a Tensor of shape (input_length, 1). It's only 1 because we are predicting only the temperature, but are using all features of the input to inform our prediction. The reason why there input_length number of rows is because the target doesn't contain only the point 12 hours after the last input point, but the future 12 hour value for each of the input points.

Example:
Let's say we're using the past 4 hour's worth of data and want to predict the temperature in 5 hours.
input = [1, 2, 3, 4] (for simplicity's sake, just putting the hour for each sample but remember, each sample consists of the 19 features describing the weather)
target = [5, 7, 8, 9]

It's kind of similar to the datasets we made in the NLP homework.

### Models

I chose 5 types of model to implement.

#### Baseline

The Baseline model literally does nothing. It just returns the input temperatures as its prediction, so it represents a model that has not learned anything and uses the input as its answer. It's mostly here for comparing other models to it.

#### LSTM RNN

The LSTM RNN consists of:

- PyTorch LSTM layer with hidden_size=64, num_layers=2
- Linear layer as a decoder

I tried adding embeddings at first because we did that in the NLP homework and it seemed weird to not use embeddings, but I quickly realized that embeddings didn't make sense in this scenario since we weren't working with categorical data.


#### GRU RNN

The GRU RNN consists of:
- PyTorch GRU layer with feature_size=512 and num_layers=2
- Linear layer as a decoder

#### Transformer

The Transformer consists of:
- PyTorch TransformerEncoder with 4 TransformerEncoderLayers, n_head=1
- Linear layer as a decoder

#### Neural Net

The Neural Net consists of:
- 4 Linear Layers with hidden_size=512 and ReLU layers in between

### Training
The training set was the first ~70% of the data points.
Training was based on previous homeworks. Some specifics I used were:
- criterion = mean average loss. (I accidently used cross entropy at first and panicked when the loss when into -500.)
- number of epochs = 10
- optimizer = Adam
- learning rate = 0.002
- weight decay = 0.0005
- train batch size = 128
- test batch size = 128


### Testing
 
  The test set was the last ~20% of the data points.

## Results

I saved ~10% of the data set for evaluation for plot purposes, but only really ended up using only one point in the evaluation set to graph... Oops.

- Add plots
- Discuss results

## Discussion

Honestly, I was just a little bit disappointed by the results. I probably should have expected that the LSTM RNN and the GRU RNN were similar because they're both RNNs, but the the fact that the Transformer didn't really do better did suck a little. I was like, "Ooh, self-attention!", but the LSTM and GRU basically did marginally better anyways. I'm also concerned that I messed up on the mask for the Transformer, so hopefully that was done okay (if not, I completely blame the PyTorch tutorial).

Anyways, after being disappointed by the RNNs, I threw in a basic neural network with linear layers just to see how it compared. And it did similar to the others, if not better??? Am I messing up the training or something? 

Also: loss is deceptive. At least, average loss. Average loss was ~0.08 which seems good, and then I plotted the predictions vs. the target and it was a lot more off then I thought!
