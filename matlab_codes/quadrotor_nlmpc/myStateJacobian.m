function [A,B] = myStateJacobian(x,u)
g=9.8
A = zeros(6,6);
A(1,4) = 1;
A(2,5) = 1;
A(3,6) = 1;
B = zeros(6,4);
B(4,1) = g*tan(u(1))  %% theta
B(5,2) = -g*tan(u(2))  %%phi
B(6,3) = u(3)    %%% tau

