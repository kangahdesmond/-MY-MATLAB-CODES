clc
tic
clear
close all
format long

%Reading Piezometer 1 Trainng Data
P1_TrainingData = xlsread('Training.xlsx',1);

%Reading Piezometer 1 Testing Data 
P1_TestingData = xlsread('Testing.xlsx',1);

%Transposing the Piezometer 1 Data
P1_TrainingInputset = transpose(P1_TrainingData(:,1:4));
P1_TrainingTargetset = transpose(P1_TrainingData(:,5));

P1_TestingInputset = transpose(P1_TestingData(:,1:4));
P1_TestingTargetset = transpose(P1_TestingData(:,5));

%Normalizing the Piezometer 1 Data
[NORM_P1_TrainInputset, ps] = mapminmax(P1_TrainingInputset);
[NORM_P1_TrainTargetset, pn] = mapminmax(P1_TrainingTargetset);
NORM_P1_TestInputset = mapminmax('apply',P1_TestingInputset,ps);

%setting the random seed number to stabilise the BPNN system
setdemorandstream(491218382)

inputs = NORM_P1_TrainInputset; %normalised P1 training inputs
targets = NORM_P1_TrainTargetset; %normalised P1 training target

m = length(inputs(:,1)); 
o = length(targets(:,1));

hidden_neurons = 17;

%% Creating and configuring neural network
net=feedforwardnet(hidden_neurons);
net=configure(net,inputs,targets);

fobj = @(X) obfunc(X,hidden_neurons,m,o,net,inputs,targets);

dim=m*hidden_neurons+hidden_neurons+hidden_neurons+o;

%% Starting initialization
pop_size=30;
% variables_no=10;
max_iter=100;  
lb=-100; % can be a vector too
ub=100; % can be a vector too
N=30;


% initialize Silverback
Silverback=[];
Silverback_Score=inf;

%%Initialization
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

%% Initialization ends


%% Calculating fitness
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


%% GTO Main loop starts

%%Setting GTO parameters    
p=0.03;
Beta=3;
w=0.8;

%%Main loop
for It=1:max_iter 
    
    a=(cos(2*rand)+1)*(1-It/max_iter);
    C=a*(2*rand-1); 

%% Exploration:

    for i=1:N
        if rand<p    
            GX(i,:) =(ub-lb)*rand+lb;
        else  
            if rand>=0.5
                Z = unifrnd(-a,a,1,dim);
                H=Z.*X(i,:);   
                GX(i,:)=(rand-a)*X(randi([1,N]),:)+C.*H; 
            else   
                GX(i,:)=X(i,:)-C.*(C*(X(i,:)- GX(randi([1,N]),:))+rand*(X(i,:)-GX(randi([1,N]),:))); %ok ok 

            end
        end
    end       
       
    GX = boundaryCheck(GX, lb, ub);
    
    % Group formation operation 
    for i=1:N
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
    for i=1:N
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
    for i=1:N
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
             
convergence_curve(It)=Silverback_Score;
fprintf("In Iteration %d, best estimation of the global optimum is %4.4f \n ", It,Silverback_Score );
         
end 
toc

% Final neural network model 
net_GTO = feedforwardnet(hidden_neurons); 
net_GTO = configure(net_GTO,inputs,targets);

disp('hidden_neurons  P1Train_RMSE  P1Test_RMSE')

It=0; 
for i=1:hidden_neurons 
    for j=1:m 
        It=It+1; 
        xi(i,j)=Silverback(It); 
    end
end
for i=1:hidden_neurons It=It+1; 
    xl(i)=Silverback(It); 
    xb1(i,1)=Silverback(It+hidden_neurons);
end
for i=1:o 
    It=It+1; 
    xb2(i,1)=Silverback(It);
end
net_GTO.iw{1,1}=xi;
net_GTO.lw{2,1}=xl;
net_GTO.b{1,1}=xb1;
net_GTO.b{2,1}=xb2;


%% Network performance
%% %% SUMMARY OF NETWORK PERFOMANCE

%%P1 Training Results
P1Train_Results = net_GTO(inputs);
P1Train_Pred = (mapminmax('reverse',P1Train_Results,pn));
P1Train_Error = gsubtract(P1_TrainingTargetset,P1Train_Pred);%calculates the error between Ymeasured and Ypredicted for training
P1Train_Performance = perform(net_GTO,P1_TrainingTargetset,P1Train_Pred);
P1Train_RMSE = sqrt(P1Train_Performance);  % Root Mean Squared Error
Train_R = corr2(P1_TrainingTargetset,P1Train_Results);
Train_VAF = (1-(var(P1_TrainingTargetset-P1Train_Pred)^2/var(P1_TrainingTargetset)))*100;


%%P1 Testing Results
P1Test_Results = sim(net_GTO,NORM_P1_TestInputset);  
P1Test_Pred = mapminmax('reverse',P1Test_Results,pn);
P1Test_Error = gsubtract(P1_TestingTargetset,P1Test_Pred);%calculates the error between Ymeasured and Ypredicted for testing
P1Test_Performance = perform(net_GTO,P1_TestingTargetset,P1Test_Pred);
P1Test_RMSE = sqrt(P1Test_Performance);  % Root Mean Squared Error
Test_VAF = (1-(var(P1_TestingTargetset-P1Test_Pred)^2/var(P1_TestingTargetset)))*100;
Test_R = corr2(P1_TestingTargetset,P1Test_Results);


%%Stabilizing the algorithm
%setdemorandstream(491218382);
 