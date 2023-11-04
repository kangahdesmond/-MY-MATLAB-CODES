
clc;
clear;
close all;

% Reading the preprocessed data 
trainheightanomaly = readmatrix('PREPROCESSED.xlsx');
testheightanomaly = readmatrix('PREPROCESSED.xlsx',Sheet='Sheet2');
traininputs = readmatrix('PREPROCESSED.xlsx',sheet='sheet3');
testinputs = readmatrix('PREPROCESSED.xlsx',sheet='sheet4');
trainoutput = readmatrix('PREPROCESSED.xlsx',sheet='sheet5');
testoutput = readmatrix('PREPROCESSED.xlsx',sheet='sheet6');

% Defining the inputs 
Traininputs = traininputs';
Testinputs = testinputs';

% Defining the outputs
Trainoutput = trainoutput';
Testouput = testoutput';


% Defining the LSTM architecture 
inputSize = 3;
outputSize = 1;
hiddenLayer1 = 10;
hiddenLayer2 = 15;



layers =[ sequenceInputLayer(inputSize)
lstmLayer(hiddenLayer1)
lstmLayer(hiddenLayer2)
fullyConnectedLayer(outputSize)
regressionLayer ];

options = trainingOptions ("adam",...
    'maxEpochs',1000,...
    'MiniBatchSize', 10, ...
    'GradientThreshold',0.01,...
    'InitialLearnRate',0.001);

% Training the network
net = trainNetwork(Traininputs,Trainoutput,layers,options);

% Prediction 
Prediction_Test = predict(net,Testinputs);

% Denormalizing the predicted Test
denormalizedTestPred = Prediction_Test * (max(testheightanomaly) - min(testheightanomaly)) + min(testheightanomaly);
TestPredicted = denormalizedTestPred';

% Computing Root mean Square Error 
RMSE_Test = sqrt(mean((TestPredicted -testheightanomaly).^2));

% Comparing the predicted test and the desired test 
Test_compare = [testheightanomaly,TestPredicted];