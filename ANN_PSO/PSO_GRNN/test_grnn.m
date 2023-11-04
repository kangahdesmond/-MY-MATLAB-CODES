% %%A PSO-GRNN Model for Railway Freight Volume Prediction by Sun Yan
% %%?????
tic
%%READING THE EXCEL FILE
Training_Data = xlsread('TRAIN_WELLA.xlsx');
% Testing_Data = xlsread('BPANN_TEST.xlsx');
Testing_Data = xlsread('TEST_WELLA.xlsx');

%%Training Data
Xtrain = transpose(Training_Data(:,1:3));%1:5 specify training data
Ytrain = transpose(Training_Data(:,4));% specify target data X coordinate

%%Testing data
Xtest = transpose(Testing_Data(:,1:3));%specify testing input data
Ytest = transpose(Testing_Data(:,4));%specify testing target data


% p_train=p(1:3,1:132);
% t_train=t(1:132,:);
% p_test=p(4:6,1:88);
% t_test=t(133:220,:);
p_zz_train=Xtrain;
t_zz_train=Ytrain;
p_zz_test=Xtest;
t_zz_test=Ytest;
% p_gy_train=mapminmax(p_zz_train,0,1);%????? 0,1???
% t_gy_train=mapminmax(t_zz_train,0,1);%????? 0,1???
% p_gy_test=mapminmax(p_zz_test,0,1);%????? 0,1???
% t_gy_test=mapminmax(t_zz_test,0,1);%????? 0,1???

%%Normalizing Training Set
[p_gy_train, ps] = mapminmax(p_zz_train);% train_X1 contains the normalized values of Training(input);

[t_gy_train, pn] = mapminmax(t_zz_train);% train_X2 contains the normalized values of Validation set;

%%Normalizing Testing Input Set
% test_X1 = mapminmax('apply',Xtest,ps);% tn contains the normalized values of Target(output);
[p_gy_test, ps] = mapminmax(p_zz_test);% 

[t_gy_test, pn] = mapminmax(t_zz_test);% 

% %%PSO ??????
% c1initial=0.1;%c1 ???
% c1final=0.005;%c1 ???
% c2initial=0.005;%c2 ???
% c2final=0.1;%c2 ???
% wmin=0.1;%w ???
% wmax=0.3;%w ???
% popmax=1;%pop ???
% popmin=0;%pop ???
% vmax=0.01;%v ???
% vmin=-0.01;%v ???
% maxgen=150;%??????
% popsize=20;%????

%%PSO ??????
c1initial=0.1;%c1 ???
c1final=0.05;%c1 ???
c2initial=0.05;%c2 ???
c2final=0.1;%c2 ???
wmin=0.1;%w ???
wmax=0.5;%w ???
popmax=1;%pop ???
popmin=0;%pop ???
vmax=0.01;%v ???
vmin=-0.01;%v ???
maxgen=200;%??????
popsize=60;%????

% % %%?????????????
test_out_expect=repmat(t_gy_test,popsize,1);%??????????????????,repmat(??????,????,????)
% 
% % %%???:???????????
for i=1:popsize
    pop(i,:)=abs(rands(1,1));%???????????(0,1)??? abs ?????
    v(i,:)=rands(1,1)*0.01;%?????????
    spread(i)=pop(i,:);%??????????????????
    net=newgrnn(p_gy_train,t_gy_train,spread(i));%??????????
    test_out_sim(i,:)=sim(net,p_gy_test);%??????
    error(i,:)=test_out_expect(i,:)-test_out_sim(i,:);%??????
    fitness(i)=mse(error(i,:));%??????????????
end

%%?????????????????
[bestfitness bestindex]=min(fitness);
gbest=pop(bestindex,:);%????????
pbest=repmat(pop,1,1);%????????
fitnesspbest=fitness;%???????
fitnessgbest=bestfitness;%???????

%%PSO ??????
for j=1:maxgen
%??????
%??????
    for i=1:popsize
    w=wmax-(wmax-wmin)*j/maxgen;%??????
    c1=(c1final-c1initial)*j/maxgen+c1initial;%?????? c1
    c2=(c2final-c2initial)*j/maxgen+c2initial;%?????? c2
    v(i,:)= w*v(i,:)+ c1*rand*(pbest(i,:)-pop(i,:))+c2*rand*(gbest-pop(i,:));%??????
    v(i,find(v(i,:)>vmax))=vmax;%??????
    v(i,find(v(i,:)<vmin))=vmin;%??????
    %??????
    pop(i,:)=pop(i,:)+v(i,:);%??????
    pop(i,find(pop(i,:)>popmax))=popmax;%??????
    pop(i,find(pop(i,:)<popmin))=popmin;%??????
    %???????
    spread(i)=pop(i,:);
    net=newgrnn(p_gy_train,t_gy_train,spread(i));
    test_out_sim(i,:)=sim(net,p_gy_test);
    error(i,:)=test_out_expect(i,:)-test_out_sim(i,:);
    fitness(i)=mse(error(i,:));
    end
    for i=1:popsize
    if fitness(i)<fitnesspbest(i)
    pbest(i,:)=pop(i,:);
    fitnesspbest(i)=fitness(i);
    end
    end
    %??????
    for i=1:popsize
    if fitness(i)<fitnessgbest
    gbest=pop(i,:);
    fitnessgbest=fitness(i);
    end
    end
    aa(j)=fitnessgbest;%??????????
    bb(j)=gbest;%???????????
end
        
%%??,?????????????? hold on
figure(1)
plot(aa)
xlabel('Iteration Process', 'fontsize',12);
ylabel('Best Fitness Value of the Swarm', 'fontsize',12);

figure(2)
plot(bb)
xlabel('Iteration Process', 'fontsize',12);
ylabel('Best Position of the Swarm', 'fontsize',12);
%%????????????? GRNN
[globalbestfitness globalbestindex]=min(aa);
globalbestspread=bb(globalbestindex);
net=newgrnn(p_gy_train,t_gy_train,globalbestspread);
test_out_best=sim(net,p_gy_test);
        
%ERROR COMPUTATION
%% %% SUMMARY OF NETWORK PERFOMANCE
%%Training data Performance

y = net(p_gy_train);%%New training out values from the trained network

Y = mapminmax('reverse',y,pn); %denormalizing the BPANN prediction

TRAINING_RESULTS=Y';

e = gsubtract(Ytrain,Y);%%calculates the error between training input and new estimated trained output

MSE_train = mean(e.^2);

%Root mean squared error
RMSE_train=sqrt( mean(e.^2));  % Root Mean Squared Error-lowest value
% 
% %Mean biased error
% MBE_train=mean(sum(e));
% 
% %Average absolute relative deviation (AARD)
% AARD_train=(mean(sum((abs(e/Ytrain))*100)));%the lower the value the better the results

%Coefficent of determination(R2)
R2_train=COD(Ytrain,Y);

figure(3)
plotregression(Ytrain',TRAINING_RESULTS)
legend('Ideal line','Fit line','Production rate (GRNN Training data)')
xlabel('Actual production rate (10^4m^3/day)')
ylabel('Predicted production rate (10^4m^3/day)')
axis on;

%% Test data Performance

t = sim(net,p_gy_test);%%Simulating the network with Testing data

T = mapminmax('reverse',t,pn);%denormalizing the BPANN prediction

TESTING_RESULTS=T';
%Error estimation

etest = gsubtract(Ytest,T);%calculates the error Ytest and predicted test target (T)

%Model performance evaluation
%Mean squared error


MSE_testing = mean(etest.^2);
%Root mean squared error
RMSE_testing=sqrt( mean(etest.^2));  % Root Mean Squared Error-lowest value

% %Man biased error
% MBE_testing=mean(sum(etest));
% 
% %Average absolute relative deviation (AARD)
% AARD_testing=(mean(sum((abs(etest/Ytest))*100)));%the lower the value the better the results

%Coefficent of determination(R2)
R2_testing=COD(Ytest,T);

figure(4)
plotregression(Ytest',TESTING_RESULTS)
legend('Ideal line','Fit line','Production rate (GRNN Testing data)')
xlabel('Actual production rate (10^4m^3/day)')
ylabel('Predicted production rate (10^4m^3/day)')
axis on;

 toc       
        
        
