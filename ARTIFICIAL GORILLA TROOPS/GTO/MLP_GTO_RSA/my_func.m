function [f] = my_func(X,hidden_neurons,m,o,net,inputs,targets)

t=0; 
for i=1:hidden_neurons 
    for j=1:m t=t+1; 
        Xi(i,j)=X(t); 
    end
end
for i=1:hidden_neurons t=t+1; Xl(i)=X(t); 
    Xb1(i,1)=X(t+hidden_neurons);
end
for i=1:o 
    t=t+1; Xb2(i,1)=X(t);
end
net.iw{1,1}=Xi;
net.lw{2,1}=Xl;
net.b{1,1}=Xb1;
net.b{2,1}=Xb2; 
f=sum((net(inputs)-targets).^2)/length(inputs);