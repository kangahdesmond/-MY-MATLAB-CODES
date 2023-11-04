clear all 
close all
clc

data = xlsread('C:\Users\kanga\Desktop\DATA\Book1.xlsx');

X_WGS = data(:,1);
Y_WGS = data(:,2);
Z_WGS = data(:,3);

X_War = data(:,4);
Y_War = data(:,5);
Z_War = data(:,6);
% Population size and stoppoing condition 
pop_size = 34;  
max_iter = 100;  

% Define your objective function's details here
fobj = @ObjectiveFunction;
variables_no = 7;
lower_bound = -100; % can be a vector too
upper_bound = 100; % can be a vector too
      
[SSE_X, SSE_Y, SSE_Z] = AVOA(pop_size, max_iter,lower_bound,upper_bound,variables_no,fobj);


figure 

% Best optimal values for the decision variables 
subplot(1,2,1)
parallelcoords(Best_vulture1_X)
xlabel('Decision variables')
ylabel('Best estimated values ')
box on

% Best convergence curve
subplot(1,2,2)
plot(convergence_curve);
title('Convergence curve of AVOA')
xlabel('Current_iteration');
ylabel('Objective value');
box on

