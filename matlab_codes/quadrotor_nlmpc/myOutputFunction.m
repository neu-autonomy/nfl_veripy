function y = myOutputFunction(x,u)


y = zeros(6,1);
y(1) = x(1);
y(2) = x(2)+0.2*x(3);
y(3) = x(3)*x(4);