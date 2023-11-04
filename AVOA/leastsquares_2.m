clc
clear
close all
format long %It will give me results in 15 decimal places

Data = xlsread('Data_Abridged');    %Reading the excel data 


          %%Reading the War Office 1926 Ellipsoid Data and defining its parameters

WAR_LONG = Data(:,1);      %Longitude of the War Office
WAR_LAT = Data(:,2);       %Latitude of the War Office
h_WAR = Data(:,3)          %Ellipsoidal height for the War Office
 
awar = 6378249.0;                                         %semi-major axis 
 
fwar = 1/293.465;                                              %flatenning 
bwar = awar*(1-fwar);                                   %semi-minor axis
ewar = sqrt((2*fwar)-fwar^2);                              % first eccentricity 
    %Nwar is the radius of curvature in the prime vertical

          %%Reading the WGS84 Ellipsoidal data and defining its parameters
     
WGS_LONG = Data(:,4);    %Longitude of the WGS84
WGS_LAT = Data(:,5);     %Latitude of the WGS84
h_WGS = Data(:,6);       %Ellipsoidal height for the WGS84

     
aWGS = 6378137.000;                                        %semi-major axis 
bWGS = 6356752.314245;                                     %semi-minor axis 
fWGS = 1/298.257223563;                                    %flatenning 
eWGS = sqrt((2*fWGS)-fWGS^2);                              %first eccentricity 

              
            %%Forward Conversion of War Office and WGS Geodetic coordinates to Cartesian X, Y, Z coordinates

for i = 1:15
     Nwar(i,1) = awar/(sqrt(1 - ((ewar*ewar)*(sind(WAR_LAT(i))^2))));
     X_War(i,1) = (Nwar(i)+ h_WAR(i))*cosd(WAR_LAT(i))*cosd(-WAR_LONG(i));
     Y_War(i,1) = (Nwar(i)+ h_WAR(i))*cosd(WAR_LAT(i))*sind(-WAR_LONG(i));
     Z_War(i,1) = (Nwar(i)*(1-(ewar^2))+h_WAR(i))*sind(WAR_LAT(i));
       
     NWGS(i,1) = aWGS/(sqrt(1 - (eWGS*sind(WGS_LAT(i)))^2));    %NWGS is the radius of curvature in the prime vertical
     X_WGS(i,1) = (NWGS(i)+ h_WGS(i))*cosd(WGS_LAT(i))*cosd(-WGS_LONG(i)); 
     Y_WGS(i,1) = (NWGS(i)+ h_WGS(i))*cosd(WGS_LAT(i))*sind(-WGS_LONG(i));
     Z_WGS(i,1) = (NWGS(i)*(1-(eWGS^2))+h_WGS(i))*sind(WGS_LAT(i));   

end

                      %%From L=Ax: where X are the 7 parameters (unkowns)
           %%Formulating the design matrix(A) or the coefficient of the unknowns

A = zeros(15*3,7);

for i = 1:15
    A(i*3-2,1) = 1;             %first row
    A(i*3-2,5) = -Z_War(i);
    A(i*3-2,6) = Y_War(i);
    A(i*3-2,7) = X_War(i);

    A(i*3-1,2) = 1;             %second row
    A(i*3-1,4) = Z_War(i);
    A(i*3-1,6) = -X_War(i);
    A(i*3-1,7) = Y_War(i);

    A(i*3,3) = 1;               %third row
    A(i*3,4) = -Y_War(i);
    A(i*3,5) = X_War(i);
    A(i*3,7) = Z_War(i);
end


             %%Observational vector (L)

L = zeros(15*3,1);

for i =1:15
    L(i*3-2,1) = X_WGS(i) - X_War(i);
    L(i*3-1,1) = Y_WGS(i) - Y_War(i);
    L(i*3,1) = Z_WGS(i) - Z_War(i);
end

                  %%Computing Bursa-Wolf 7 parameters using parametric least square solution

X = inv(A'*A)*A'*L;
          

    %%Defining the translation parameters. (Shifts in the X,Y,Z coordinates between WAR Office and WGS84 measured in meters)
Dx = X(1);
Dy = X(2);
Dz = X(3);

    %%Defining the rotational parameters and converting into seconds 
Rx = ((180/pi)*3600)*X(4);
Ry = ((180/pi)*3600)*X(5);
Rz = ((180/pi)*3600)*X(6);

     %%Definig the scale factor
sf = 1000000*X(7);

      %%computing the residuals by applying least squares solution
V = L-(A*X);

      %%Calculating the standard deviation and reference variance for the transformation parameters

n = 34;                        %number of redundant observations
u = 7;                         %number of unknown parameters
SD = sqrt((V'*V)/((3*n)-u));   %Standard Deviation
variance = SD^2;               %Adjustment Reference Variance

      %%Calculating the Standard deviation for each transformation parameter
S = SD*sqrt(diag(inv(A'*A)));

parameters = [Dx,Dy,Dz,X(4),X(5),X(6),X(7)]';

yI = A * parameters;

        %%performing the transformation

for i = 1:15
    Xw(i,1) = X_WGS(i) - yI(i*3-2,1);
    Yw(i,1) = Y_WGS(i) - yI(i*3-1,1);
    Zw(i,1) = Z_WGS(i) - yI(i*3,1);
end
    
         %%Performing Reverse Conversion of cartesian X, Y, Z coordinates to geodetic coordinates

for i = 1:15

    P(i,1) = sqrt((Xw(i)^2)+(Yw(i)^2));              %perpendicular distance to the rotational axis
    P_lat(i,1) = atand((awar*Zw(i))/(bwar*P(i)));    %parametric latitude
    ewar_2 = (ewar^2)/(1-(ewar^2));                  %second eccentricity
    
    y(i,1) = Zw(i)+(bwar*ewar_2*(sind(P_lat(i)))^3);   %numerator of geodetic latitude 
    x(i,1) = P(i)-(awar*(ewar^2)*(cosd(P_lat(i)))^3);  %denominator of geodetic latitude
    
    Long_T(i,1) = atand(Yw(i)/Xw(i));            %Transformed Longitude for the War Office
    Lat_T(i,1) = atand(y(i)/x(i));               %Transformed Latitude for the War Office
    h_T(i,1) = (P(i)/cosd(Lat_T(i)))- Nwar(i);   %Tranformed ellipsoidal height for the War Office
    
    
end

       %%conversion of the latitude and longitude from degrees to radians
lat_rad = deg2rad(Lat_T);
long_rad = deg2rad(Long_T);
