clc, clear all, close all

% Simulate Lorenz system
dt=0.01; T=8; t=0:dt:T;
b=8/3; sig=10; r=28;

Lorenz = @(t,x)([ sig * (x(2) - x(1))       ; ...
                  r * x(1)-x(1) * x(3) - x(2) ; ...
                  x(1) * x(2) - b*x(3)         ]);              
ode_options = odeset('RelTol',1e-10, 'AbsTol',1e-11);

input=[]; output=[];
figure;
for j=1:100  % training trajectories 100
    x0=30*(rand(3,1)-0.5);
    [t,y] = ode45(Lorenz,t,x0);
    input=[input; y(1:end-1,:)];
    output=[output; y(2:end,:)];
    plot3(y(:,1),y(:,2),y(:,3)), hold on
    plot3(x0(1),x0(2),x0(3),'ro')
end

grid on, view(-23,18)

% Simulate Lorenz system
dt=0.01; T=8; t=0:dt:T;
b=8/3; sig=10; r=10;


% input=[]; output=[];
figure;
for j=1:100  % training trajectories 100
    x0=30*(rand(3,1)-0.5);
    [t,y] = ode45(Lorenz,t,x0);
    input=[input; y(1:end-1,:)];
    output=[output; y(2:end,:)];
    plot3(y(:,1),y(:,2),y(:,3)), hold on
    plot3(x0(1),x0(2),x0(3),'ro')
end


grid on, view(-23,18)

% Simulate Lorenz system
dt=0.01; T=8; t=0:dt:T;
b=8/3; sig=10; r=40;

% input=[]; output=[];
figure;
for j=1:100  % training trajectories 100
    x0=30*(rand(3,1)-0.5);
    [t,y] = ode45(Lorenz,t,x0);
    input=[input; y(1:end-1,:)];
    output=[output; y(2:end,:)];
    plot3(y(:,1),y(:,2),y(:,3)), hold on
    plot3(x0(1),x0(2),x0(3),'ro')
end


grid on, view(-23,18)


%%
net = feedforwardnet([10 10 10]);
net.layers{1}.transferFcn = 'logsig';
net.layers{2}.transferFcn = 'radbas';
net.layers{3}.transferFcn = 'purelin';
net = train(net,input.',output.');

save workspace_lor

%%
figure(2)

dt=0.01; T=8; t=0:dt:T;
b=8/3; sig=10; r=17;

Lorenz = @(t,x)([ sig * (x(2) - x(1))       ; ...
                  r * x(1)-x(1) * x(3) - x(2) ; ...
                  x(1) * x(2) - b*x(3)         ]);              
ode_options = odeset('RelTol',1e-10, 'AbsTol',1e-11);

x0=30*(rand(3,1)-0.5);
[t,y] = ode45(Lorenz,t,x0);
plot3(y(:,1),y(:,2),y(:,3)), hold on
plot3(x0(1),x0(2),x0(3),'ro','Linewidth',[2])
grid on

ynn(1,:)=x0;
for jj=2:length(t)
    y0=net(x0);
    ynn(jj,:)=y0.'; x0=y0;
end
plot3(ynn(:,1),ynn(:,2),ynn(:,3),':','Linewidth',[2])

figure(3)
subplot(3,2,1), plot(t,y(:,1),t,ynn(:,1),'Linewidth',[2])
xlabel('time')
ylabel('x')
subplot(3,2,3), plot(t,y(:,2),t,ynn(:,2),'Linewidth',[2])
xlabel('time')
ylabel('y')
subplot(3,2,5), plot(t,y(:,3),t,ynn(:,3),'Linewidth',[2])
xlabel('time')
ylabel('z')


figure(2)

dt=0.01; T=8; t=0:dt:T;
b=8/3; sig=10; r=35;

Lorenz = @(t,x)([ sig * (x(2) - x(1))       ; ...
                  r * x(1)-x(1) * x(3) - x(2) ; ...
                  x(1) * x(2) - b*x(3)         ]);              
ode_options = odeset('RelTol',1e-10, 'AbsTol',1e-11);

x0=30*(rand(3,1)-0.5);
[t,y] = ode45(Lorenz,t,x0);
plot3(y(:,1),y(:,2),y(:,3)), hold on
plot3(x0(1),x0(2),x0(3),'ro','Linewidth',[2])
grid on

ynn(1,:)=x0;
for jj=2:length(t)
    y0=net(x0);
    ynn(jj,:)=y0.'; x0=y0;
end
plot3(ynn(:,1),ynn(:,2),ynn(:,3),':','Linewidth',[2])

figure(3)
subplot(3,2,2), plot(t,y(:,1),t,ynn(:,1),'Linewidth',[2])
xlabel('time')
ylabel('x')
subplot(3,2,4), plot(t,y(:,2),t,ynn(:,2),'Linewidth',[2])
xlabel('time')
ylabel('y')
subplot(3,2,6), plot(t,y(:,3),t,ynn(:,3),'Linewidth',[2])
xlabel('time')
ylabel('z')


figure(2), view(-75,15)
figure(3)
subplot(3,2,1), set(gca,'Fontsize',[12],'Xlim',[0 8])
subplot(3,2,2), set(gca,'Fontsize',[12],'Xlim',[0 8])
subplot(3,2,3), set(gca,'Fontsize',[12],'Xlim',[0 8])
subplot(3,2,4), set(gca,'Fontsize',[12],'Xlim',[0 8])
subplot(3,2,5), set(gca,'Fontsize',[12],'Xlim',[0 8])
subplot(3,2,6), set(gca,'Fontsize',[12],'Xlim',[0 8])
legend('Lorenz','NN')

clc, clear all, close all

% Simulate Lorenz system
dt=0.01; T=8; t=0:dt:T;
b=8/3; sig=10; r=28;

Lorenz = @(t,x)([ sig * (x(2) - x(1))       ; ...
                  r * x(1)-x(1) * x(3) - x(2) ; ...
                  x(1) * x(2) - b*x(3)         ]);              
ode_options = odeset('RelTol',1e-10, 'AbsTol',1e-11);

input=[]; output=[];
figure;

n_step_prec=5;
for j=1:100  % training trajectories 100
    x0=30*(rand(3,1)-0.5);
    [t,y] = ode45(Lorenz,t,x0);
    input=[input; y(1:end,:)];
    
    for i=1:length(y)-n_step_prec-5
        for k=1:5
    if sign(y(i,1))~=sign(y(i+n_step_prec+k,1))
        outputs(i,1)=1;
        break;
    else
        outputs(i,1)=-1;
    end
        end
    end
    
    for i=length(y)-n_step_prec-4:length(y)-n_step_prec
        k=k-1;
          if sign(y(i,1))~=sign(y(i+n_step_prec+k,1))
        outputs(i,1)=1;
        break;
    else
        outputs(i,1)=-1;
          end
    end
    
    for i=length(y)-n_step_prec+1:length(y)
    outputs(i,1)=outputs(i-1,1);
    end
    
    output=[output;outputs];
    

%     output=[output; y(2:end,:)];
    plot3(y(:,1),y(:,2),y(:,3)), hold on
    plot3(x0(1),x0(2),x0(3),'ro')
end


grid on, view(-23,18)





%%
net = feedforwardnet([10 10 10]);
net.layers{1}.transferFcn = 'logsig';
net.layers{2}.transferFcn = 'radbas';
net.layers{3}.transferFcn = 'purelin';
net = train(net,input.',output.');

save workspace2

%%
figure(2)

dt=0.01; T=8; t=0:dt:T;
b=8/3; sig=10; r=28;

Lorenz = @(t,x)([ sig * (x(2) - x(1))       ; ...
                  r * x(1)-x(1) * x(3) - x(2) ; ...
                  x(1) * x(2) - b*x(3)         ]);              
ode_options = odeset('RelTol',1e-10, 'AbsTol',1e-11);

x0=30*(rand(3,1)-0.5);
[t,y] = ode45(Lorenz,t,x0);
plot3(y(:,1),y(:,2),y(:,3)), hold on
plot3(x0(1),x0(2),x0(3),'ro','Linewidth',[2])
grid on
view(-75,15)

% ynn(1,:)=x0;
for jj=1:length(t)
    y0=net(y(jj,:).');
    ynn(jj,1)=sign(y0); 
%     x0=y0;
end
% plot3(ynn(:,1),ynn(:,2),ynn(:,3),':','Linewidth',[2])

figure(3)
subplot(2,1,1)
plot(t,y(:,1),'Linewidth',[2])
hold on
plot(t(ynn==1),y(ynn==1),'r.','markersize',20)
xlabel('time')
ylabel('x')

figure(2)
hold on

dt=0.01; T=8; t=0:dt:T;
b=8/3; sig=10; r=28;

Lorenz = @(t,x)([ sig * (x(2) - x(1))       ; ...
                  r * x(1)-x(1) * x(3) - x(2) ; ...
                  x(1) * x(2) - b*x(3)         ]);              
ode_options = odeset('RelTol',1e-10, 'AbsTol',1e-11);

x0=30*(rand(3,1)-0.5);
[t,y] = ode45(Lorenz,t,x0);
plot3(y(:,1),y(:,2),y(:,3)), hold on
plot3(x0(1),x0(2),x0(3),'ro','Linewidth',[2])
grid on
view(-75,15)

% ynn(1,:)=x0;
for jj=1:length(t)
    y0=net(y(jj,:).');
    ynn(jj,1)=sign(y0); 
%     x0=y0;
end
% plot3(ynn(:,1),ynn(:,2),ynn(:,3),':','Linewidth',[2])

figure(3)
subplot(2,1,2)
plot(t,y(:,1),'Linewidth',[2])
hold on
plot(t(ynn==1),y(ynn==1),'r.','markersize',20)
xlabel('time')
ylabel('x')
legend('Lorenz','NN')

clc, clear all, close all

% Simulate Lorenz system
dt=0.01; T=8; t=0:dt:T;
b=8/3; sig=10; r=28;

Lorenz = @(t,x)([ sig * (x(2) - x(1))       ; ...
                  r * x(1)-x(1) * x(3) - x(2) ; ...
                  x(1) * x(2) - b*x(3)         ]);              
ode_options = odeset('RelTol',1e-10, 'AbsTol',1e-11);

input=[]; output=[];
figure;

n_step_prec=15;
for j=1:100  % training trajectories 100
    x0=30*(rand(3,1)-0.5);
    [t,y] = ode45(Lorenz,t,x0);
    input=[input; y(1:end,:)];
    
    for i=1:length(y)-n_step_prec-5
        for k=1:5
    if sign(y(i,1))~=sign(y(i+n_step_prec+k,1))
        outputs(i,1)=1;
        break;
    else
        outputs(i,1)=-1;
    end
        end
    end
    
    for i=length(y)-n_step_prec-4:length(y)-n_step_prec
        k=k-1;
          if sign(y(i,1))~=sign(y(i+n_step_prec+k,1))
        outputs(i,1)=1;
        break;
    else
        outputs(i,1)=-1;
          end
    end
    
    for i=length(y)-n_step_prec+1:length(y)
    outputs(i,1)=outputs(i-1,1);
    end
    
    output=[output;outputs];
    

%     output=[output; y(2:end,:)];
    plot3(y(:,1),y(:,2),y(:,3)), hold on
    plot3(x0(1),x0(2),x0(3),'ro')
end


grid on, view(-23,18)





%%
net = feedforwardnet([10 10 10]);
net.layers{1}.transferFcn = 'logsig';
net.layers{2}.transferFcn = 'radbas';
net.layers{3}.transferFcn = 'purelin';
net = train(net,input.',output.');

save workspace2_new_timewindow

%%
figure(2)

dt=0.01; T=8; t=0:dt:T;
b=8/3; sig=10; r=28;

Lorenz = @(t,x)([ sig * (x(2) - x(1))       ; ...
                  r * x(1)-x(1) * x(3) - x(2) ; ...
                  x(1) * x(2) - b*x(3)         ]);              
ode_options = odeset('RelTol',1e-10, 'AbsTol',1e-11);

x0=30*(rand(3,1)-0.5);
[t,y] = ode45(Lorenz,t,x0);
plot3(y(:,1),y(:,2),y(:,3)), hold on
plot3(x0(1),x0(2),x0(3),'ro','Linewidth',[2])
grid on
view(-75,15)

% ynn(1,:)=x0;
for jj=1:length(t)
    y0=net(y(jj,:).');
    ynn(jj,1)=sign(y0); 
%     x0=y0;
end
% plot3(ynn(:,1),ynn(:,2),ynn(:,3),':','Linewidth',[2])

figure(3)
subplot(2,1,1)
plot(t,y(:,1),'Linewidth',[2])
hold on
plot(t(ynn==1),y(ynn==1),'r.','markersize',20)
xlabel('time')
ylabel('x')

figure(2)
hold on

dt=0.01; T=8; t=0:dt:T;
b=8/3; sig=10; r=28;

Lorenz = @(t,x)([ sig * (x(2) - x(1))       ; ...
                  r * x(1)-x(1) * x(3) - x(2) ; ...
                  x(1) * x(2) - b*x(3)         ]);              
ode_options = odeset('RelTol',1e-10, 'AbsTol',1e-11);

x0=30*(rand(3,1)-0.5);
[t,y] = ode45(Lorenz,t,x0);
plot3(y(:,1),y(:,2),y(:,3)), hold on
plot3(x0(1),x0(2),x0(3),'ro','Linewidth',[2])
grid on
view(-75,15)

% ynn(1,:)=x0;
for jj=1:length(t)
    y0=net(y(jj,:).');
    ynn(jj,1)=sign(y0); 
%     x0=y0;
end
% plot3(ynn(:,1),ynn(:,2),ynn(:,3),':','Linewidth',[2])

figure(3)
subplot(2,1,2)
plot(t,y(:,1),'Linewidth',[2])
hold on
plot(t(ynn==1),y(ynn==1),'r.','markersize',20)
xlabel('time')
ylabel('x')
legend('Lorenz','NN')



