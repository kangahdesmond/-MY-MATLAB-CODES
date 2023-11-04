clc
format long

%%READING THE EXCEL FILE

Training_Data = xlsread('Training_Data');
Testing_Data = xlsread('Testing_Data');

%%Training Data
Xtrain = transpose(Training_Data(:,1:4));%(1:3, input columns) specify training input data
Ytrain = transpose(Training_Data(:,5));% target (Au)

%%Testing data
Xtest = transpose(Testing_Data(:,1:4));%specify testing input data
Ytest = transpose(Testing_Data(:,5));%specify testing target data

%Data normalization of Training, Validation and Testing set into[-1,1]

%%Normalizing Training Set
[train_X1, ps] = mapminmax(Xtrain);% train_X1 contains the normalized values of Training(input);
% ps contains the max and min values of the original training set

[train_X2, pn] = mapminmax(Ytrain);% train_X2 contains the normalized values of Validation set;
% pn contains the max and min values of the original training set

%%Normalizing Testing Input Set
test_X1 = mapminmax('apply',Xtest,ps);% tn contains the normalized values of Target(output);


%% Setting the random seed number to stabilise the BPNN system
setdemorandstream(491218382);

%%Creating a BPANN %%OPTIMAL NEURONS=11
for Nb_Neuron = 1:2
MyNetwork = newff(train_X1,train_X2,[Nb_Neuron],{'tansig' 'purelin'},'trainbr');%%Optimum hidden neuron is 8 after several trials

MyNetwork.trainparam.min_grad = 0.0000001;%%denotes the minimum performance gradient
MyNetwork.trainParam.epochs = 5000;%%denotes the maximum number of epochs to train
MyNetwork.trainParam.goal = 0;
%MyNetwork.trainParam.lr = 0.03;%%denotes the learning rate
%MyNetwork.trainParam.mc = 0.7;%%default momentum value is 0.9
%MyNetwork.trainParam.max_fail = 6;%%denotes the maximum validation failures
MyNetwork.performFcn = 'mse';  % Mean squared error

%%TRAINING THE NETWORK
MyNetwork = train(MyNetwork,train_X1,train_X2);

disp('Nb_Neuron    TRAINING_RMSE    TESTING_RMSE');
%% %% SUMMARY OF NETWORK PERFOMANCE

%%Training data Performance

y = MyNetwork(train_X1);%%New training out values from the trained network

Training_prediction = mapminmax('reverse',y,pn); %denormalizing the BPANN prediction

Training_error = gsubtract(Ytrain,Training_prediction);%%calculates the error between training input and new estimated trained output

trainingPerformance = perform(MyNetwork,Ytrain,Training_prediction);%%MSE training value

TRAINING_RMSE = sqrt(trainingPerformance);
%TRAINING_MAPE = (mean(abs(Ytrain-Training_prediction)./Ytrain))*100;


%% Test data Performance

t = sim(MyNetwork,test_X1);%%Simulating the network with Testing data

Testing_prediction = mapminmax('reverse',t,pn);%denormalizing the BPANN prediction

testing_error = gsubtract(Ytest,Testing_prediction);%calculates the error Ytest and predicted test target (T)

testPerformance = perform(MyNetwork,Ytest,Testing_prediction);%%MSE test value

TESTING_RMSE = sqrt(testPerformance);
%TESTING_MAPE = mean(abs(testing_error)./Ytest)*100;

% View the Network
%view(MyNetwork)

% Plots
% Uncomment these lines to enable various plots.
figure, plotregression(Ytrain,Training_prediction,'Training')
figure, plotregression(Ytest,Testing_prediction,'Test')
fprintf('%d     %f     %f      %f       %f\n', Nb_Neuron, TRAINING_RMSE, TESTING_RMSE);

end