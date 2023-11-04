clc
clear all
close all
format long
%rng default
tic

%Loading the Observational data from excel to matlab
Training_Data = xlsread('Training.xlsx',1); 
Testing_Data = xlsread('Testing.xlsx',1);

%Assigning parts of the loaded observational data matrices
%Training data
Input_Training_Data = (Training_Data(:,1:4))'; %Assign column1 to 3 to input training data
Target_Training_Data = (Training_Data(:,5))'; %Assign column4 to target training data

%Testing data
Input_Testing_Data = (Testing_Data(:,1:4))'; %Assign column1 to 3 to input testing data
Target_Testing_Data = (Testing_Data(:,5))'; %Assign column4 to target testing data

%Normalizing Training data
[Input_Train_Data1, ps] = mapminmax(Input_Training_Data);
[Target_Train_Data1, pn] = mapminmax(Target_Training_Data);

%Normalizing Input Testing data
Input_Test_Data1 = mapminmax('apply',Input_Testing_Data,ps);

inputs = Input_Train_Data1; %normalised training inputs
targets = Target_Train_Data1; %normalised training target

m = length(inputs(:,1)); % Size of Input Decision Variables Matrix
o = length(targets(:,1)); % Size of Target Decision Variables Matrix

neurons = 3;%% number of neurons. This is what needs to be tested till you get the best results.

% create a neural network
net = feedforwardnet(neurons);

% configure the neural network for this dataset
net = configure(net, inputs, targets);


% handle for calling the AOfunc.m file which is the Costfunction
fobj = @(X)  obfunc(X,neurons,m,o,net,inputs,targets);


%% Initialization---------------------------------------------------start
%%[ X ]=initialization(N,dim,ub,lb)

ub=100;  %upper boundary limit

lb=-100; %lower boundary limit

dim = m*neurons+neurons+neurons+o;%number of variables required in the weights and biases column vector

N=20; %number of candidate solutions

pop_size = 20;

max_iter= 100; % maximum number of iterations


%% Initializing population of Gorilla----------------------------start

Silverback=[];
Silverback_Score=inf;

Boundary_no= size(ub,2); % numnber of boundaries

% If the boundaries of all variables are equal and user enter a signle
% number for both ub and lb
if Boundary_no==1
    X=rand(N,dim).*(ub-lb)+lb;
end

% If each variable has a different lb and ub
if Boundary_no>1
    for i=1:dim
        ub_i=ub(i);
        lb_i=lb(i);
        X(:,i)=rand(N,1).*(ub_i-lb_i)+lb_i;
    end
end


convergence_curve=zeros(max_iter,1);

% AO Main loop-----------------------------------------------------end

for i=1:pop_size   
    Pop_Fit(i)=fobj(X(i,:));%#ok
    if Pop_Fit(i)<Silverback_Score 
            Silverback_Score=Pop_Fit(i); 
            Silverback=X(i,:);
    end
end

GX=X(:,:);
lb=ones(1,dim).*lb; 
ub=ones(1,dim).*ub; 

%%  Controlling parameter

p=0.03;
Beta=3;
w=0.8;

%%Main loop
for It=1:max_iter 
    
    a=(cos(2*rand)+1)*(1-It/max_iter);
    C=a*(2*rand-1); 

%% Exploration:

    for i=1:pop_size
        if rand<p    
            GX(i,:) =(ub-lb)*rand+lb;
        else  
            if rand>=0.5
                Z = unifrnd(-a,a,1,dim);
                H=Z.*X(i,:);   
                GX(i,:)=(rand-a)*X(randi([1,pop_size]),:)+C.*H; 
            else   
                GX(i,:)=X(i,:)-C.*(C*(X(i,:)- GX(randi([1,pop_size]),:))+rand*(X(i,:)-GX(randi([1,pop_size]),:))); %ok ok 

            end
        end
    end       
       
    GX = boundaryCheck(GX, lb, ub);
    
    % Group formation operation 
    for i=1:pop_size
         New_Fit= fobj(GX(i,:));          
         if New_Fit<Pop_Fit(i)
            Pop_Fit(i)=New_Fit;
            X(i,:)=GX(i,:);
         end
         if New_Fit<Silverback_Score 
            Silverback_Score=New_Fit; 
            Silverback=GX(i,:);
         end
    end
    
%% Exploitation:  
    for i=1:pop_size
       if a>=w  
            g=2^C;
            delta= (abs(mean(GX)).^g).^(1/g);
            GX(i,:)=C*delta.*(X(i,:)-Silverback)+X(i,:); 
       else
           
           if rand>=0.5
              h=randn(1,dim);
           else
              h=randn(1,1);
           end
           r1=rand; 
           GX(i,:)= Silverback-(Silverback*(2*r1-1)-X(i,:)*(2*r1-1)).*(Beta*h); 
           
       end
    end
   
    GX = boundaryCheck(GX, lb, ub);
    
    % Group formation operation    
    for i=1:pop_size
         New_Fit= fobj(GX(i,:));
         if New_Fit<Pop_Fit(i)
            Pop_Fit(i)=New_Fit;
            X(i,:)=GX(i,:);
         end
         if New_Fit<Silverback_Score 
            Silverback_Score=New_Fit; 
            Silverback=GX(i,:);
         end
    end
end      
convergence_curve(It)=Silverback_Score;
fprintf("In Iteration %d, best estimation of the global optimum is %4.4f \n ", It,Silverback_Score );

%% Final neural network model
disp('Final nn model is final_net')
net_GOA = feedforwardnet(neurons); 
net_GOA = configure(net_GOA,inputs,targets);

disp('neurons  RMSE_training  RMSE_testing')

%updating the weights and biases of the net using the trained results 
%from GOA

t=0; 
for i=1:neurons 
    for j=1:m 
        t=t+1; 
        xi(i,j)= Silverback(t); 
    end
end
for i=1:neurons t=t+1; 
    xl(i)= Silverback(t); 
    xb1(i,1)= Silverback(t+neurons);
end
for i=1:o 
    t=t+1; 
    xb2(i,1)= Silverback(t);
end
net_GOA.iw{1,1}=xi;
net_GOA.lw{2,1}=xl;
net_GOA.b{1,1}=xb1;
net_GOA.b{2,1}=xb2;

%% Test the Network Performance

%Training data Performance
y1 = net_GOA(Input_Train_Data1); %new predicted values

Output_Training_Data = mapminmax('reverse',y1,pn); %denormalizing the predicted train values

y2 = (Output_Training_Data)'; %Transposing the predicted values for analysis in excel

Training_error = gsubtract(Target_Training_Data,Output_Training_Data); %calculating the error between the target data and the newly predicted values

Train_error = (Training_error)'; %Transposing the computed errors for analysis in excel

Training_performance = perform(net_GOA,Target_Training_Data,Output_Training_Data); %computation for MSE

Training_RMSE = sqrt(Training_performance); %Computation for RMSE

%R_training = corrcoef(Target_Training_Data,Output_Training_Data); %%correlation coefficient


%Test data Performance
t1 = sim(net_GOA,Input_Test_Data1); %Simulating the network with testing data

Output_Testing_Data = mapminmax('reverse',t1,pn); %denormalizing the predicted testing values

t2 = (Output_Testing_Data)'; %Tranposing the predicted testing data

Testing_error = gsubtract(Target_Testing_Data,Output_Testing_Data); %calculating the error between the target data and the newly predicted values

Test_error = (Testing_error)'; %Transposing the predicted testing error

Testing_Performance = perform(net_GOA,Target_Testing_Data,Output_Testing_Data); %Computation for MSE

Testing_RMSE = sqrt(Testing_Performance); %Computation for RMSE

%R_testing = corrcoef(Target_Testing_Data,Output_Testing_Data); %correlation coefficient


%view(net_WOA)
 
fprintf('%d %f %f %f %f\n',neurons, Training_RMSE, Testing_RMSE);

