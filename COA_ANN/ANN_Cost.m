function cost = ANN_Cost(POP)

global input output numOfNeurons NumofInputs

cost = zeros(size(POP,1),1);

for ipop = 1:size(POP,1)
    pop = POP(ipop,:);

    W = pop(1:(NumofInputs+1)*numOfNeurons);
    U = pop((NumofInputs+1)*numOfNeurons+1 : end);

    z1 = tansig(input*W(1)+W(4));
    z2 = tansig(input*W(2)+W(5));
    z3 = tansig(input*W(3)+W(6));
    z4 = 1;   % bias
    
    ANN_output = z1*U(1) + z2*U(2) + z3*U(3) + z4*U(4);
    
    cost(ipop) = sum( ( (output - ANN_output).^2 ) )./ sum(output.^2) * 100;
    
end




