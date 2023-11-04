clc
tic
clear
close all
format long

%Reading Piezometer 1 Trainng Data
P3_TrainingData = xlsread('hybrid train.xlsx',3);

%Reading Piezometer 1 Testing Data 
P3_TestingData = xlsread('hybrid test.xlsx',3);

%Transposing the Piezometer 1 Data
P3_TrainingInputset = transpose(P3_TrainingData(:,2));
P3_TrainingTargetset = transpose(P3_TrainingData(:,1));

P3_TestingInputset = transpose(P3_TestingData(:,2));
P3_TestingTargetset = transpose(P3_TestingData(:,1));

%Normalizing the Piezometer 1 Data
[NORM_P3_TrainInputset, ps] = mapminmax(P3_TrainingInputset);
[NORM_P3_TrainTargetset, pn] = mapminmax(P3_TrainingTargetset);
NORM_P3_TestInputset = mapminmax('apply',P3_TestingInputset,ps);

inputs = NORM_P3_TrainInputset; %normalised P1 training inputs
targets = NORM_P3_TrainTargetset; %normalised P1 training target

m = length(inputs(:,1)); 
o = length(targets(:,1));

hidden_neurons = 16;

%% Creating and configuring neural network
net=feedforwardnet(hidden_neurons);
net=configure(net,inputs,targets);

fobj = @(X) obfunc(X,hidden_neurons,m,o,net,inputs,targets);

dim=m*hidden_neurons+hidden_neurons+hidden_neurons+o;

%% Starting initialization
pop_size=20;
% variables_no=10;
max_iter=100;  
lower_bound=-100; % can be a vector too
upper_bound=100; % can be a vector too
N=20;


% initialize Silverback
Silverback=[];
Silverback_Score=inf;

%%Initialization
Boundary_no= size(upper_bound,2); % numnber of boundaries

% If the boundaries of all variables are equal and user enter a signle
% number for both ub and lb
if Boundary_no==1
    X=rand(N,dim).*(upper_bound-lower_bound)+lower_bound;
end

% If each variable has a different lb and ub
if Boundary_no>1
    for i=1:dim
        ub_i=upper_bound(i);
        lb_i=lower_bound(i);
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
% lb=ones(1,dim).*lb; 
% ub=ones(1,dim).*ub;


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
            GX(i,:) =(upper_bound-lower_bound)*rand+lower_bound;
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
       
    GX = boundaryCheck(GX, lower_bound, upper_bound);
    
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
   
    GX = boundaryCheck(GX, lower_bound, upper_bound);
    
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

%%P3 Training Results
P3Train_Results = net_GTO(inputs);
P3Train_Pred = (mapminmax('reverse',P3Train_Results,pn));
P3Train_Error = gsubtract(P3_TrainingTargetset,P3Train_Pred);%calculates the error between Ymeasured and Ypredicted for training
P3Train_Performance = perform(net_GTO,P3_TrainingTargetset,P3Train_Pred);
P3Train_RMSE = sqrt(P3Train_Performance);  % Root Mean Squared Error
Train_R = corr2(P3_TrainingTargetset,P3Train_Results);
Train_VAF = (1-(var(P3_TrainingTargetset-P3Train_Pred)^2/var(P3_TrainingTargetset)))*100;


%%P3 Testing Results
P3Test_Results = sim(net_GTO,NORM_P3_TestInputset);  
P3Test_Pred = mapminmax('reverse',P3Test_Results,pn);
P3Test_Error = gsubtract(P3_TestingTargetset,P3Test_Pred);%calculates the error between Ymeasured and Ypredicted for testing
P3Test_Performance = perform(net_GTO,P3_TestingTargetset,P3Test_Pred);
P3Test_RMSE = sqrt(P3Test_Performance);  % Root Mean Squared Error
Test_VAF = (1-(var(P3_TestingTargetset-P3Test_Pred)^2/var(P3_TestingTargetset)))*100;
Test_R = corr2(P3_TestingTargetset,P3Test_Results);
 