function R2=COD(y,f)
%  y: Actual data
%  f: Model fit
%     y=reshape(y,1,prod(size(y)));
%     f=reshape(f,1,prod(size(f)));
    %This calculates the coefficient of determination
    ymean=mean(y); %
    SStot=0;
    SSerr=0;
    for i=1:length(y); %loop through each point
        SStot=SStot+(y(i)-ymean)^2;
        SSerr=SSerr+(y(i)-f(i))^2;
    end
    R2=abs(1-SSerr/SStot);
end %End function..