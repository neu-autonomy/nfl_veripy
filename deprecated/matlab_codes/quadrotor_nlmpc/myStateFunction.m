

function x= myStateJacobian(x,u, Ts)
g=9.8;
xtmp =x;
x(1) = xtmp(1) + (x(4)*Ts);
x(2) = xtmp(2) +( x(5)*Ts);
x(3) = xtmp(3) + (x(6)*Ts);

x(4) = xtmp(4)+ (g*tan(u(1))) *Ts; %% theta
x(5) =xtmp(5) -g*tan(u(2))*Ts;  %%phi
x(6) = xtmp(6)+u(3) ;  %%% tau

