%% reading data set
clear
clc
trainingdata = readmatrix('Testing_Data.txt')';
testingdata = readmatrix ('KamaraTest.txt')';

X_train = trainingdata(2:8,:);
Y_train = trainingdata(1,:);

X_test = testingdata(2:8,:);
Y_test = testingdata(1,:);
rng('default');
%fileID = fopen('result.txt','w');
%% 

spread = 0.64564
MN = 37  % MN is the maximum number of Neorons
%goal = 0;
%DF = 1;  % DF is the number neorons to add to display
net = newgrnn(X_train,Y_train,spread); 

Pred_Y_train = sim(net,X_train);
Pred_Y_test = sim(net,X_test);

error_train = Y_train - Pred_Y_train;
error_train2 = error_train.^2;
n1 = length(Y_train);
train_mse = sum(error_train2)/n1;
train_rmse = sqrt(train_mse);
R_train = corr2(Y_train,Pred_Y_train);



error_test = Y_test-Pred_Y_test;
error_test2 = error_test.^2;
nl2 = length(Y_test);
test_mse = sum(error_test.^2)/nl2;
test_rmse = sqrt(test_mse);
R_test = corr2(Y_test,Pred_Y_test);


%fprintf(fileID,'%.2f %d %.4f %.4f %.4f %.4f\n',spread,MN,R_train,train_mse,...
   % R_test,test_mse);


%fclose(fileID);

%results = readmatrix('result.txt');
%filename = 'result.xlsx';
%writematrix(results, filename)
train_error = Y_train-Pred_Y_train;
VAF_train = (1 -(var(train_error)/var(Y_train)))*100;

test_error = Y_test-Pred_Y_test;
VAF_test = (1-(var(test_error)/var(Y_test)))*100;


