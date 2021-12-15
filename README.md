# CSE 490L: Deep Learning Project

## Project
 
### Abstract
This project focuses on Time Series Weather Forecasting based on the Jena climate dataset by the Max Planck Institute for Biogeochemistry. Specifically, in this project, the goal is to predict the a future temperature based on some amount of past data. As there are different types of network architectures that can be used to approach this, a few were selected and evaluated based on MSE to compare them. 

### Video

https://user-images.githubusercontent.com/79888079/146142338-5af39ee6-029d-47c8-a450-cd3ce02e3497.mp4

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

The dataset itself consists of inputs and targets. 

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
- criterion = mean squared error
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

The results were evaluated by looking at the mean squared error. I accidently used cross entropy at first and was like, wow, why is the loss -500???

Anyways, I also evaluated it by taking a sample and getting the prediction foreach model. Then, the predictions were plotted alongside the targets. This made me realize that MSE can be deceptive: the loss for each model was fairly low, but each point itself differed more than I expected.

I think overall, the LSTM, GRU, Transformer, and Neural Net all perform similarly in terms of MSE, but the Transformer itself looks like it generally follows the trend.

### Evaluation Plots

#### Baseline
![eval_base](https://user-images.githubusercontent.com/79888079/146132841-96a7e23e-0c45-476c-9422-b670ba719564.png)

#### LSTM
![eval_lstm](https://user-images.githubusercontent.com/79888079/146132920-225b546d-f887-41d6-ac55-ec5b118c268d.png)

#### GRU
![eval_gru](https://user-images.githubusercontent.com/79888079/146132902-38c72741-2f72-4b21-98b7-ba1c1f29270d.png)

#### Transformer
![eval_transformer](https://user-images.githubusercontent.com/79888079/146132964-898727a4-f1dd-400c-b4b7-67343aa8658a.png)

#### Neural Net
![eval_neural](https://user-images.githubusercontent.com/79888079/146132946-428ebfa2-49be-4e8f-891f-54bdbf1240c3.png)

### MSE Plots

#### Baseline
![base_model](https://user-images.githubusercontent.com/79888079/146139817-6005bb8e-0132-473f-a9c8-eb7402108d4b.png)

#### LSTM
![LSTM](https://user-images.githubusercontent.com/79888079/146133615-2fb955e6-2502-4d52-af77-875fb9c70f29.png)

#### GRU
![GRU](https://user-images.githubusercontent.com/79888079/146133660-1d778d18-189b-4967-a376-31959777aad1.png)

#### Transformer
![transformer](https://user-images.githubusercontent.com/79888079/146133755-78c11621-f817-4701-91ff-c2aebac1a080.png)

#### Neural Net
![neural](https://user-images.githubusercontent.com/79888079/146133806-c5f77930-0b05-4287-b995-0d7960c783a9.png)

## Discussion
Honestly, I was just a little bit disappointed by the results. I probably should have expected that the LSTM RNN and the GRU RNN were similar because they're both RNNs, but the the fact that the Transformer didn't really do better did suck a little. I was like, "Ooh, self-attention!", but the LSTM and GRU basically did marginally better anyways. I'm also concerned that I messed up on the mask for the Transformer, so hopefully that was done okay (if not, I completely blame the PyTorch tutorial).

Anyways, after being disappointed by the RNNs, I threw in a basic neural network with linear layers just to see how it compared. And it did similar to the others, if not better??? Am I messing up the training or something? 

The other thing that I wanted to talk about is the weirdness of using weather forecasting deep learning models. When I was researching for this project, I also came across lots of research on the impacts of deep learning models on climate change due to the computational resources it needs. Even though it's a separate arena, weather forecasting
also needs to take into account climate change,  but it also potentially makes things a little worse. Kind of crazy.

And as the last point, here are some learning lessons (or me complaining):
- pandas was more confusing than expected
- Converting TensorFlow and Keras logic to PyTorch is also harder than expected

Overall though, I'm pretty pleased with the project. The results weren't what I expected, but I learned a lot so it was definitely worth it.

Thanks for taking a look at my project!


