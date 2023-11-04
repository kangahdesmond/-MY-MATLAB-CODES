function [f] = obfunc(X,hidden_neurons,m,o,net,inputs,targets)

It=0; 
for i=1:hidden_neurons 
    for j=1:m It=It+1; 
        Xi(i,j)=X(It); 
    end
end
for i=1:hidden_neurons It=It+1; Xl(i)=X(It); 
    Xb1(i,1)=X(It+hidden_neurons);
end
for i=1:o 
    It=It+1; Xb2(i,1)=X(It);
end
net.iw{1,1}=Xi;
net.lw{2,1}=Xl;
net.b{1,1}=Xb1;
net.b{2,1}=Xb2; 
f=sum((net(inputs)-targets).^2)/length(inputs);