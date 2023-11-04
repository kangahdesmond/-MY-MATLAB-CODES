clc;
close all;
clear all;
format long;
Data = xlsread('Data_Abridged');
WAR_LONG = Data(:,1);
WAR_LAT = Data(:,2);
WAR_HEIGHT = Data(:,3);
WGS_LONG = Data(:,4);
WGS_LAT = Data(:,5);
WGS_HEIGHT = Data(:,6);

%Defining the war office 1926 parameters

awar = 6378249.0;
%bwar = 6356751.68824;
fwar = 1/293.465;
bwar = awar*(1-fwar);
ewar = sqrt((2*fwar)-(fwar)^2);
awgs = 6378137;
bwgs = 6356752.314245;
fwgs = 1/298.257223563;
ewgs = sqrt((2*fwgs)-(fwgs)^2);
for i = 1:15;
    %converting geographic coordinates of war office into war office cartesian
    %coordinates
    nw(i,1) = (ewar*sind(WAR_LAT(i)))^2;
    N_war(i,1) = awar/sqrt(1-(nw(i)));
    X_war(i,1) = (N_war(i)+WAR_HEIGHT(i))*cosd(WAR_LAT(i))*cosd(WAR_LONG(i));
    Y_war(i,1) = (N_war(i)+WAR_HEIGHT(i))*cosd(WAR_LAT(i))*sind(WAR_LONG(i));
    Z_war(i,1) = (N_war(i)*(1-(ewar)^2)+WAR_HEIGHT(i))*sind(WAR_LAT(i));
    %converting geographic coordinates of WGS84 into WGS84 cartesian
    %coordinates
    nq(i,1) = (ewgs*sind(WGS_LAT(i)))^2;
    N_wgs(i,1) = awgs/sqrt(1-(nq(i)));
    X_wgs(i,1) = (N_wgs(i)+WGS_HEIGHT(i))*cosd(WGS_LAT(i))*cosd(WGS_LONG(i));
    Y_wgs(i,1) = (N_wgs(i)+WGS_HEIGHT(i))*cosd(WGS_LAT(i))*sind(WGS_LONG(i));
    Z_wgs(i,1) = (N_wgs(i)*(1-(ewgs)^2)+WGS_HEIGHT(i))*sind(WGS_LAT(i));
   
end
A = zeros(15*3,7);%Defining the design matrix
for i = 1:15;
    A(i*3-2,1) = 1;%First row;
    A(i*3-2,5) = -Z_war(i);
    A(i*3-2,6) = Y_war(i);
    A(i*3-2,7) = X_war(i);
    A(i*3-1,2) = 1;%Second row;
    A(i*3-1,4) = Z_war(i);
    A(i*3-1,6) = -X_war(i);
    A(i*3-1,7) = Y_war(i);
    A(i*3,3) = 1;%Third row;
    A(i*3,4) = -Y_war(i);
    A(i*3,5) = X_war(i);
    A(i*3,7) = Z_war(i)
       
  
    
end
%Defining observed values;
L = zeros(15*3,1);
for i =1:15;
    L(i*3-2,1) = X_wgs(i)-X_war(i);
    L(i*3-1,1) = Y_wgs(i)-Y_war(i);
    L(i*3,1) = Z_wgs(i)-Z_war(i);
end
%Computing for the seven unknown parameters USING OLS;
    Q_xx = inv(A'*A);
    X_parameters = Q_xx*A'*L;
    %Computing for residuals using ordinary least squares(OLS);
    V = (A* X_parameters)-L;
   %transforming the cartesian coordinates of war office with the determined
    %parameters
     M = A*X_parameters;
for i = 1:15;
    X_war_trmd(i,1)= X_wgs(i)-M(i*3-2,1);%transformed war office x coordinate
    Y_war_trmd(i,1) = Y_wgs(i)-M(i*3-1,1);;%transformed war office y coordinate
    Z_war_trmd(i,1) = Z_wgs(i)-M(i*3,1);;%transformed war office z coordinate
      %Defining the geographic coordinates of transformed war office cartesian coordinate using Bowering's inverse equation;
    nq(i,1) =(Y_war_trmd(i)/ X_war_trmd(i));
    long_war_trmd(i,1) = atand(nq(i));%Transformed geographic longitude for war
    np = sqrt((ewar)^2)/(1-(ewar)^2);%is the second eccentricity;
    p(i,1) = sqrt((X_war_trmd(i))^2+(Y_war_trmd(i))^2);%perpendicular distance from the rotational axis;
    pl(i,1) = atand((awar*Z_war_trmd(i))/(bwar*p(i)));%parametric latitude;
    ns(i,1) = (bwar*((np)^2)*(sind(pl(i)))^3);
    nr(i,1) = (awar*((ewar)^2)*(cosd(pl(i)))^3);
    lat_war_trmd(i,1) = atand((Z_war_trmd(i)+ns(i))/(p(i)-nr(i)));% transformed geographic latitude for war
    h_war_trmd(i,1) = ((p(i)/cosd(lat_war_trmd(i)))-N_war(i))%transformed geographic height for war;
    
end
    %Performing statistical analysis
    SD = sqrt(V'*V)/(3*15)-7;%standard deviation
    %Calculating for the standard deviation of each of the parameters
    %determined;
    S = SD*sqrt(diag(Q_xx))
    
