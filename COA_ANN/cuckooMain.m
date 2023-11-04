% -----------------------------------------------------------------
% Cuckoo Optimization Algorithm (COA) by Ramin Rajabioun          %
% Programmed by Ramin Rajabioun                                   %
% -----------------------------------------------------------------
% Paper: R. Rajabioun. Cuckoo Optimization Algorithm, Applied Soft
% Computing 11 (2011) 5508–5518
% ----------------------------------------------------------------%
% This program implements a standard version of Cuckoo            %
% Optimization Algorithm (COA) which minimizes any Cost Function  %
% --------------------------------------------------------------- %
% Email: r.rajabioun@ece.ut.ac.ir                                 %
% Website: www.coasite.info                                       %
% --------------------------------------------------------------- %
%
% To use the code easily prepare your cost function and type its  %
% name in: costFunction = 'YourCostFunctionName', then set number %
% of optimization parameters in "npar" and set the upper and lower%
% bands of the problem                                            %
% --------------------------------------------------------------- %


clc, clear, close all

global input output numOfNeurons NumofInputs

%% Set problem parameters

% select a cost function:

costFunction = 'ANN_Cost';  

input  = (-6:0.5:6)';
output = sin(input+eps)./(input+eps);

%% ANN Structure
numOfNeurons = 3;
NumofInputs = 1;
npar = (NumofInputs+1) * numOfNeurons + (numOfNeurons+1); % Number of Optimization Parameters

varLo = -20;         % Lower  band of parameter
varHi =  20;         % Higher band of parameter


%% Set COA parameters

numCuckooS = 5;            % number of initial population
minNumberOfEggs = 2;        % minimum number of eggs for each cuckoo
maxNumberOfEggs = 4;        % maximum number of eggs for each cuckoo
maxIter = 200;               % maximum iterations of the Cuckoo Algorithm
knnClusterNum = 1;          % number of clusters that we want to make
motionCoeff = 20;            % Lambda variable in COA paper, default=2
accuracy = 0e-10;           % How much accuracy in answer is needed
maxNumOfCuckoos = 20;      % maximum number of cuckoos that can live at the same time
radiusCoeff = 0.05;           % Control parameter of egg laying
cuckooPopVariance = 1e-10;   % population variance that cuts the optimization


%% initialize population:

cuckooPop = cell(numCuckooS,1);
% initialize egg laying center for each cuckoo
for cuckooNumber = 1:numCuckooS    
    cuckooPop{cuckooNumber}.center = ( varHi-varLo )*rand(1,npar) + varLo;
end

%% Start Cuckoo Optimization Algorithm

iteration = 0;
maxProfit = -1e20;        % Let initial value be negative number
goalPoint = (varHi - varLo)*rand(1,npar) + varLo; % a random goalpoint to start COA
globalBestCuckoo = goalPoint;
globalMaxProfit = maxProfit;
profitVector = [];
while ( (iteration <= maxIter) && (-maxProfit > accuracy) )
    
    iteration = iteration + 1
    
    % initialize number of eggs for each cuckoo
    for cuckooNumber = 1:numCuckooS        
        cuckooPop{cuckooNumber}.numberOfEggs = floor( (maxNumberOfEggs - minNumberOfEggs) * rand + minNumberOfEggs );
    end

    % get total number of available eggs between all cuckooS
    summ = 0;
    for cuckooNumber = 1:numCuckooS
        summ = summ + cuckooPop{cuckooNumber}.numberOfEggs;
    end

    % calculate egg laying radius for each Cuckoo, considering problem
    % limitations and ratio of each cuckoo's eggs
    for cuckooNumber = 1:numCuckooS
        cuckooPop{cuckooNumber}.eggLayingRadius = cuckooPop{cuckooNumber}.numberOfEggs/summ * ( radiusCoeff * (varHi-varLo) );
    end

    % To lay eggs, we produced some radius values less than egg laying
    % radius of each cuckoo
    for cuckooNumber = 1:numCuckooS
        cuckooPop{cuckooNumber}.eggLayingRadiuses = cuckooPop{cuckooNumber}.eggLayingRadius * rand(cuckooPop{cuckooNumber}.numberOfEggs,1);
    end
    
    for cuckooNumber = 1:numCuckooS
        params = cuckooPop{cuckooNumber}.center;        % get center values
        tmpRadiuses = cuckooPop{cuckooNumber}.eggLayingRadiuses;
        numRadiuses = numel(tmpRadiuses);
        % divide a (hyper)circle to 'numRadiuses' segments
        % This is to search all over the current habitat
        angles = linspace(0,2*pi,numRadiuses);    % in Radians
        newParams = [];
        for cnt = 1:numRadiuses
            addingValue = zeros(1,npar);
            for iii = 1:npar
                randNum = floor(2*rand)+1;
                addingValue(iii) = (-1)^randNum * tmpRadiuses(cnt)*cos(angles(cnt)) + tmpRadiuses(cnt)*sin(angles(cnt));
            end
            newParams = [newParams; params +  addingValue ];
        end
        
        
        % check for variable limits
        newParams(find(newParams>varHi)) = varHi;
        newParams(find(newParams<varLo)) = varLo;

        cuckooPop{cuckooNumber}.newPosition4Egg = newParams;
    end
    
    % Now egg laying is done

    
    % Now that egg positions are found, they are laid, and so its time to
    % remove the eggs on the same positions (because each egg only can go to one nest)
    for cuckooNumber = 1:numCuckooS
        tmpPopulation = cuckooPop{cuckooNumber}.newPosition4Egg;
        tmpPopulation = floor(tmpPopulation * 1e4)/1e4;
        ii = 2;
        cntt = 1;
        while ii <= size(tmpPopulation,1) || cntt <= size(tmpPopulation,1)
            if all((tmpPopulation(cntt,:) == tmpPopulation(ii,:)))
                tmpPopulation(ii,:) = [];
            end
            ii = ii + 1;
            if ii > size(tmpPopulation,1) && cntt <= size(tmpPopulation,1)
                cntt = cntt + 1;
                ii = cntt + 1;
                if ii > size(tmpPopulation,1)
                    break
                end
            end
        end
        cuckooPop{cuckooNumber}.newPosition4Egg = tmpPopulation;
    end    
    
     
    % Now we evalute egg positions
    for cuckooNumber = 1:numCuckooS
        cuckooPop{cuckooNumber}.profitValues = -feval(costFunction,[cuckooPop{cuckooNumber}.center ; cuckooPop{cuckooNumber}.newPosition4Egg]);        
    end
    
    % Now we check to see if cuckoo population is more than maxNumOfCuckoos
    % this case we should keep 1st maxNumOfCuckoos cuckoos and kill the others
    allPositions = [];
    whichCuckooPopTheEggBelongs = [];
    tmpProfits = [];
    if numCuckooS > maxNumOfCuckoos
        for cuckooNumber = 1:numCuckooS
            tmpProfits = [tmpProfits; cuckooPop{cuckooNumber}.profitValues];
            allPositions = [allPositions; [cuckooPop{cuckooNumber}.center; cuckooPop{cuckooNumber}.newPosition4Egg(:,1:npar)]];
            whichCuckooPopTheEggBelongs = [whichCuckooPopTheEggBelongs; cuckooNumber*ones(size(cuckooPop{cuckooNumber}.newPosition4Egg(:,1:npar),1),1) ];
        end
        % now we sort cuckoo profits
        [sortedProfits, sortingIndex] = sort(tmpProfits,'descend');
        % Get best cuckoo to be copied to next generation
        bestCuckooCenter = allPositions(sortingIndex(1),1:npar);
        
        sortedProfits = sortedProfits(1:maxNumOfCuckoos);
        allPositions = allPositions(sortingIndex(1:maxNumOfCuckoos),:);
        clear cuckooPop
        for ii = 1:maxNumOfCuckoos
            cuckooPop{ii}.newPosition4Egg = allPositions(ii,:);
            cuckooPop{ii}.center = allPositions(ii,:);
            cuckooPop{ii}.profitValues = sortedProfits(ii);
        end
        numCuckooS = maxNumOfCuckoos;
    else
        for cuckooNumber = 1:numCuckooS
            tmpProfits = [tmpProfits; cuckooPop{cuckooNumber}.profitValues];
            allPositions = [allPositions; [cuckooPop{cuckooNumber}.center; cuckooPop{cuckooNumber}.newPosition4Egg(:,1:npar)] ];
            whichCuckooPopTheEggBelongs = [whichCuckooPopTheEggBelongs; cuckooNumber*ones(size(cuckooPop{cuckooNumber}.newPosition4Egg(:,1:npar),1),1) ];
        end
        [sortedProfits, sortingIndex] = sort(tmpProfits,'descend');
        % Get best cuckoo to be copied to next generation
        bestCuckooCenter = allPositions(sortingIndex(1),1:npar);
    end
    
    maxProfit  = sortedProfits(1);
    currentBestCuckoo = bestCuckooCenter;
    currentMaxProfit = -feval(costFunction,currentBestCuckoo);
    if currentMaxProfit > globalMaxProfit
        globalBestCuckoo = currentBestCuckoo;
        globalMaxProfit = currentMaxProfit;
    end
    
    % Update cost minimization plot
%     plot(iteration, -globalMaxProfit,'ks','linewidth',2,'MarkerEdgeColor','k','MarkerFaceColor','g','MarkerSize',10)
%     title([ 'Curent Cost = ' num2str(-globalMaxProfit) ' , at Iteration = ' num2str(iteration) ])
%     pause(0.1)
    

    W = globalBestCuckoo(1:(NumofInputs+1)*numOfNeurons);
    U = globalBestCuckoo((NumofInputs+1)*numOfNeurons+1 : end);

    z1 = tansig(input*W(1)+W(4));
    z2 = tansig(input*W(2)+W(5));
    z3 = tansig(input*W(3)+W(6));
    z4 = 1;   % bias

    ANN_output = z1*U(1) + z2*U(2) + z3*U(3) + z4*U(4);

    figure(2), clf
    plot(input, ANN_output,'r*')
    hold on
    plot(input, output,'ks')
    xlim([min(input)-0.5 max(input)+0.5])
    ylim([min(output)-0.5 max(output)+0.5])
    legend('COA ANN data','Real data','location','North')
    title(['Error = ' num2str(-globalMaxProfit) '%' ' ,  Iteration = ' num2str(iteration) ])
    for cnt = 1:numel(input)
        line([input(cnt) input(cnt)],[ANN_output(cnt) output(cnt)])
    end
    
    profitVector = [profitVector globalMaxProfit];
    
    % ======== now we have some eggs that are safe and will grow up ==========
    %========= mating: =============

    % first we put all egg positions under each other
    allPositions = [];
    whichCuckooPopTheEggBelongs = [];
    for cuckooNumber = 1:numCuckooS
        allPositions = [allPositions; cuckooPop{cuckooNumber}.newPosition4Egg(:,1:npar)];
        whichCuckooPopTheEggBelongs = [whichCuckooPopTheEggBelongs; cuckooNumber*ones(size(cuckooPop{cuckooNumber}.newPosition4Egg(:,1:npar),1),1) ];
    end

    if sum(var(allPositions)) < cuckooPopVariance
        break
    else
        [clusterNumbers, clusterCenters] = kmeans(allPositions,knnClusterNum);
    end
    % make newly made clusters
    cluster = cell(knnClusterNum,1);
    for ii = 1:knnClusterNum
        cluster{ii}.positions = [];
        cluster{ii}.profits = [];
    end
    pointer = 1;
    for cnt = 1:length(clusterNumbers)
        if cnt < length(clusterNumbers)
            if clusterNumbers(cnt) == clusterNumbers(cnt+1)  && ...
                    whichCuckooPopTheEggBelongs(cnt) == whichCuckooPopTheEggBelongs(cnt+1)
                pointer = pointer + 1;
            else
                pointer = 1;
            end
            cluster{clusterNumbers(cnt)}.positions = [cluster{clusterNumbers(cnt)}.positions; cuckooPop{whichCuckooPopTheEggBelongs(cnt)}.newPosition4Egg(pointer,1:npar)];
            cluster{clusterNumbers(cnt)}.profits   = [cluster{clusterNumbers(cnt)}.profits; cuckooPop{whichCuckooPopTheEggBelongs(cnt)}.profitValues(pointer)];
        else
            cluster{clusterNumbers(cnt)}.positions = [cluster{clusterNumbers(cnt)}.positions; cuckooPop{whichCuckooPopTheEggBelongs(cnt)}.newPosition4Egg(pointer,1:npar)];
            cluster{clusterNumbers(cnt)}.profits   = [cluster{clusterNumbers(cnt)}.profits; cuckooPop{whichCuckooPopTheEggBelongs(cnt)}.profitValues(pointer)];
        end
    end

    % Determine the best cluster
    f_mean = zeros(knnClusterNum,1);
    for cnt = 1:knnClusterNum
        f_mean(cnt) = mean(cluster{cnt}.profits);
    end

    [sorted_f_mean, sortingIndex_f_mean] = sort(f_mean,'descend');
    maxFmean = sorted_f_mean(1);   indexOfBestCluster = sortingIndex_f_mean(1);
    
    % now that we know group with number 'indexOfBestCluster' is the best we 
    % should select their best point az Goal Point of other groups
    [maxProfitInBestCluster, indexOfBestEggPosition] = max(cluster{indexOfBestCluster}.profits);
    goalPoint  = cluster{indexOfBestCluster}.positions(indexOfBestEggPosition,1:npar);
    
    % now all other mature Cuckoos must go toward this goal point for laying
    % their eggs
    numNewCuckooS = 0;
    for cntClstr = 1:size(cluster,1)
        tmpCluster = cluster{cntClstr};
        tmpPositions = tmpCluster.positions;
        for cntPosition = 1:size(tmpPositions,1)
            tmpPositions(cntPosition,:) = tmpPositions(cntPosition,:) + ...
                                          motionCoeff * rand(1,npar) .*  (goalPoint  - tmpPositions(cntPosition,:));
        end
        % check if variables are in range
        tmpPositions(find( tmpPositions>varHi )) = varHi;
        tmpPositions(find( tmpPositions<varLo )) = varLo;

        % update cluster positions
        cluster{cntClstr}.positions = tmpPositions;
        cluster{cntClstr}.center = mean(tmpPositions);
        % update number of cuckoos: numCuckooS
        numNewCuckooS = numNewCuckooS + size(cluster{cntClstr}.positions,1);
    end

    numCuckooS = numNewCuckooS;
    % update cuckooPop
    clear cuckooPop
    cuckooPop = cell(numCuckooS,1);
    cntNumCuckooS = 1;
    for cnt = 1:size(cluster,1)
        tmpCluster = cluster{cnt};
        tmpPositions = tmpCluster.positions;
        for cntPosition = 1:size(tmpPositions,1)
            cuckooPop{cntNumCuckooS}.center = cluster{cnt}.positions(cntPosition,1:npar);
            cntNumCuckooS = cntNumCuckooS + 1;
        end
    end
    % Copy the Best cuckoo and its randomized form of this population to go 
    % to the next generation
    currentBestCuckoo = bestCuckooCenter;
    currentMaxProfit = -feval(costFunction,currentBestCuckoo);
    if currentMaxProfit > globalMaxProfit
        globalBestCuckoo = currentBestCuckoo;
        globalMaxProfit = currentMaxProfit;
    end
    cuckooPop{end}.center = globalBestCuckoo; % This is because the best cuckoo will live longer and won't die right after egg laying
    cuckooPop{end}.profitValues = -feval(costFunction,cuckooPop{end}.center);

    tmp = rand(1,npar).*globalBestCuckoo;
    tmp(find( tmp>varHi )) = varHi;
    tmp(find( tmp<varLo )) = varLo;
    cuckooPop{end-1}.center = tmp;
    cuckooPop{end-1}.profitValues = -feval(costFunction,cuckooPop{end-1}.center);
    
end     % end of while

%% Now Algorithm has stopped

disp('Best Params = ')
disp(globalBestCuckoo')

fprintf('\n')

disp('Cost = ')
disp(-globalMaxProfit)

% profit history is in:  profitVector
costVector = - profitVector;


figure
plot(costVector,'-ks','linewidth',3,'MarkerEdgeColor','k','MarkerFaceColor','g','MarkerSize',15)
xlabel 'Cuckoo Iteration'
ylabel 'Cost Value'
title(['Current Cost = ' num2str(costVector(end)) ', at iteration = ' num2str(iteration) ])




