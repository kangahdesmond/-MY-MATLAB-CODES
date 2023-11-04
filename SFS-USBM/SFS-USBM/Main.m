%*************************************************************************%
% Author: Hamid Salimi                                                    %
% Last Edited: 8/16/2014                                                  %
% Email:salimi.hamid86@gmail.com (or) h.salimi@ut.ac.ir                   %
% Reference: Stochastic Fractal Search: A Powerful Metaheuristic Algorithm%
% Published in Knowledge-Based Systems - Journal - Elsevier.              %
% Please refer to mentioned journal in term of citation.                  %
% The first online draft of journal paper can be found at below link:     %
% http://www.sciencedirect.com/science/article/pii/S0950705114002822      %
% You have to refer SFS code provided any use of SFS's part in your codes %
% If you encounter any problems or bugs in the SFS code, you are welcome  % 
% to contact me.                                                          %
%*************************************************************************%

    clear all
    close all
    clc
    
    %% Loading Data
TrainingData = readmatrix('TrainingData.xlsx');
TestingData = readmatrix('TestingData.xlsx');

X_train  = TrainingData(:, 1:2);
y_train = TrainingData(:,3);

X_test = TestingData(:,1:2);
y_test = TestingData(:,3);


%Stochastic Fracral Search-------------------------------------------------
% The structure S contains all the parameters for the SFS algorithm
%--------------------------------------------------------------------------

%Initializing Stochastic Fractal Search Parameters*************************
    % SFS has three main parameters along with an optional parameter:
    % 1- Population size considered as Start_Point
    % 2- Maximum generation considered as Maximum_Generation
    % 3- Maximum Diffusion Number (MDN) considered as Maximum_Diffusion
    % Optional parameter: Choosing diffusion walk considered as Walk
    S.Start_Point = 100;            
    S.Maximum_Generation  = 1000;     
    S.Maximum_Diffusion = 2;
    S.Walk = 0.75; % *Important
%--------------------------------------------------------------------------
    %*Please Note:

    %S.Walk = 1 ----> SFS uses the first Gaussian walk(usually SIMPLE Problems)
    %S.Walk = 0 ----> SFS uses the second Gaussian walk(usually HARD Problems)

    %You can also write:
    %S.Walk = 0.75 ---> SFS uses the first Gaussian walk, with probability 
    %of 75% which comes from uniform, and SFS uses the second Gaussian walk
    %distribution with probability of 25% .
    %Generally, to solve your problem, try to use both walks.
    %If you want to use the second Gaussian walk please increase the maximum
    %generation
%--------------------------------------------------------------------------

%**************************************************************************

%Initializing Problem******************************************************
    % To initialize your problem, you need to set three parameters:
    % 1- Set the name of your function in Function_Name
    % 2- Set the dimension of your problem in Ndim
    % 3- Set the lower and upper vector bounds in Lband and Uband respectively
    % Note: For easy implementation, your functions should take one vector
    % & get back one fitness value.
    S.Function_Name = @(phi)myobj(phi, X_train, y_train);
    S.Ndim = 2;
    S.Lband = ones(1, S.Ndim)*(-1000);
    S.Uband = ones(1, S.Ndim)*(1000);
%**************************************************************************

%Plotting Part*************************************************************
%If you want to plot the problem in 3D, set it to 1. Your dimension problem 
%should be upper than 2, else an error will be occurred.
    S.plot = 0;
%Note: plotting the result consumes the time.
%**************************************************************************

%Printing Result***********************************************************
%if you want to see the best result in each iteration, set it to 1.
    S.ShowResult = 0; 
%Note: printing the result causes consuming time.                                  
%**************************************************************************

%Start Stochastic Fractal Search*******************************************
%compute the time of finding solution
    StartOptimiser = tic;           
    [phi, SSE, F] = Stochastic_Fractal_Search(S);
    EndOptimiser = toc(StartOptimiser);
%**************************************************************************

%Print Final Results*******************************************************
    fprintf('The time of finding solution is: %f\n', EndOptimiser);

    display('The best solution is:');
    phi

    display('The value of the best fitness function is:');
    SSE
%**************************************************************************

k = phi(1);
b = phi(2);

Q_train = X_train(:,1);
D_train = X_train(:,2);

PPV_training = k.*((D_train./sqrt(Q_train)).^(-b));

Q_test = X_test(:,1);
D_test = X_test(:,2);

PPV_testing = k.*((D_test./sqrt(Q_test)).^(-b));
training_MAE = sum(abs(y_train - PPV_training))/size(y_train,1);
trainingRMSE = (sum((y_train - PPV_training).^2)/size(y_train,1))^0.5;
%train_MAPE = ((sum(abs(y_train - PPV_training)./y_train)/size(y_train,1)).*100);
r_train = corr2(y_train,PPV_training);

testing_MAE = sum(abs(y_test - PPV_testing))/size(y_test,1);
testingRMSE = (sum((y_test - PPV_testing).^2)/size(y_test,1))^0.5;
%test_MAPE = ((sum(abs(y_test - PPV_testing)./y_test)/size(y_test,1))*100);
r_test = corr2(y_test,PPV_testing);



