clear
close all
clc
nu=3
nx=6
ny =6
nlobj = nlmpc(nx,ny,nu);
Ts = 0.1;
nlobj.Ts = Ts;
nlobj.ControlHorizon = 5;
nlobj.Model.StateFcn = @myStateFunction;
nlobj.Model.IsContinuousTime = false;
nlobj.Model.NumberOfParameters = 1;
Ts = 0.1;
nlob.Ts = Ts;

c = 4;
p=30

nlobj.PredictionHorizon = p;
nlobj.ControlHorizon = c;
nlobj.States(1).Min = -5
nlobj.States(1).Max = 5
nlobj.States(2).Min = -5
nlobj.States(2).Max = 5
nlobj.States(3).Min = -5
nlobj.States(3).Max = 5
nlobj.States(4).Min = -1
nlobj.States(4).Max = 1
nlobj.States(5).Min = -1
nlobj.States(5).Max = 1
nlobj.States(6).Min = -1
nlobj.States(6).Max = 1
nlobj.ManipulatedVariables(1).Min =  -pi/9
nlobj.ManipulatedVariables(2).Min =  -pi/9
nlobj.ManipulatedVariables(3).Min =  0
nlobj.ManipulatedVariables(1).Max =  pi/9
nlobj.ManipulatedVariables(2).Max =  pi/9
nlobj.ManipulatedVariables(3).Max =  2*9.8
nlobj.Weights.OutputVariables = [1 1 1 1 1 1];
nlobj.Weights.ManipulatedVariablesRate =[1 1 1];
xHistory =[]
uHistory=[]
i=0
while i<451
    i=i+1
 %   x0 = [4.7; 4.7; 3 ;0.95; 0; 0];
    
  %  x0 = [0;0;0;0;0;0];
    x0 = [4.5,4.5,3.2,0.86,0,0]';

    u0 = zeros(nu,1);
    nloptions = nlmpcmoveopt;
    nloptions.Parameters = {Ts};
    
    validateFcns(nlobj, x0, u0, [], {Ts});
    yref = [0 0 0 0 0 0];
    
    x =x0;
    Duration =2;
    u =u0;
    x=x0; 

    u1 = tan(u(1));
    u2 =tan(u(2));
   u3 =u(3);
     u_mpc = [u1;u2;u3]
  
    x_ellipse = zeros(6,1);
    x_ellipse(1:3,:) =0.4;
    x_ellipse(4,:) =0.1;
    x_ellipse(5,:) =0.2;
    x_ellipse(6,:) =0.2;

    for j =1:6
        x(j)=x0(j)+(2*rand(1)-1)*x_ellipse(j);
    end
    xHistory = [xHistory x];
    uHistory = [uHistory u_mpc];
    
    for ct = 1:(Duration/Ts)
        % Correct previous prediction
        % Compute optimal control moves
        [u,nloptions] = nlmpcmove(nlobj,x,u,yref,[],nloptions);
        % Predict prediction model states for the next iteration
        %    predict(EKF,[mv; Ts]);
        % Implement first optimal control move
        c= zeros(6,1);
        c(6) =-9.8*Ts;
        x = myStateFunction(x,u,Ts) +c ;
        u1 = tan(u(1));
        u2 =tan(u(2));
        u3 =u(3);
        u_mpc = [u1;u2;u3];
        if u_mpc(3)==0  &ct>2
            xHistory(:,(i-1)*11+1:(i-1)*11+ct)=[];
            uHistory(:,(i-1)*11+1:(i-1)*11+ct)=[];
            i=i-1;
            not_break=0;
            break
        end
        not_break=1;
        % Generate sensor data
        y = x;
        % Save plant states
        xHistory = [xHistory x];
        uHistory = [uHistory u_mpc];
    end
  
end
figure
for i=1:451
     plot(xHistory(1,(i-1)*21+1:(i*21)),xHistory(2,(i-1)*21+1:(i*21)))
     hold on
    
end
csvwrite('quadrotor_nlmpc_x.csv',xHistory)
csvwrite('quadrotor_nlmpc_u.csv',uHistory)

