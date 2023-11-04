function FERR = myobj(phi, X_train, y_train)

k = phi(1);
b = phi(2);

Q = X_train(:,1);
D = X_train(:,2);

PPV = k.*((D./(sqrt(Q))).^(-b));

err = y_train - PPV;

FERR = sum(err.^2);
