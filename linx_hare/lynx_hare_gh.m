%% data

clear all; close all; clc;

t=[0:2:58];

dt=t(2)-t(1); 

lynx=[32 50 12 10 13 36 15 12 6 6 65 70 40 9 20 34 45 40 15 15 60 80 26 18 37 50 35 12 12 25]*1000;

hare=[20 20 52 83 64 68 83 12 36 150 110 60 7 10 70 100 92 70 10 11 137 137 18 22 52 83 18 10 9 65]*1000;

data=[hare;lynx;];

ini_cond=data(:,1);

%%%%%% body of DMD %%%%%%%%%%
v1 = data(:,1:end-1); v2 = data(:,2:end);

[U2,Sigma2,V2] = svd(v1, 'econ');
r=2; U=U2(:,1:r); Sigma=Sigma2(1:r,1:r); V=V2(:,1:r);
Atilde = U'*v2*V/Sigma;
[W,D] = eig(Atilde);
Phi=v2*V/Sigma*W;

lambda=diag(D);
omega=log(lambda)/dt;

y0 = Phi\ini_cond;  % pseudo-inverse initial conditions

u_modes = zeros(r,length(t));  % DMD reconstruction for every time point
for iter = 1:length(t)
    u_modes(:,iter) =(y0.*exp(omega*(t(iter))));
end
u_dmd = Phi*u_modes;   % DMD resconstruction with all modes

figure;
plot(1845:2:1903,data','-o');
legend('hare','lynx')
xlabel('t [years]')
ylabel('population')
figure;
plot(1845:2:1903,real(u_dmd)','-o');
legend('hare DMD model','lynx DMD model')
xlabel('t [years]')
ylabel('population')
figure, plot(omega,[0 0],'ko','Linewidth',[2])
axis([-1 1 -1 1])
xlabel('Re (\omega_k)')
ylabel('Im (\omega_k)')
%% time delay hankel matrix

H3=[];
for j=1:10
  H3=[H3; data(:,j:20+j)]; 
end 
ini_cond=H3(:,1);
v1 = H3(:,1:end-1); v2 = H3(:,2:end);

[U2,Sigma2,V2] = svd(v1, 'econ');
r=15; U=U2(:,1:r); Sigma=Sigma2(1:r,1:r); V=V2(:,1:r);
Atilde = U'*v2*V/Sigma;
[W,D] = eig(Atilde);
Phi=v2*V/Sigma*W;

lambda=diag(D);
omega=log(lambda)/dt;

y0 = Phi\ini_cond;  % pseudo-inverse initial conditions

u_modes = zeros(r,length(H3));  % DMD reconstruction for every time point
for iter = 1:length(t)
    u_modes(:,iter) =(y0.*exp(omega*(t(iter))));
end
u_dmd = Phi*u_modes;   % DMD resconstruction with all modes

figure;

plot(1845:2:1903,data(1,:),'-o');
hold on
plot(1845:2:1903,real(u_dmd(1,:))');

legend('hare data','time-delay DMD')
xlabel('t [years]')
ylabel('population')
figure;
plot(1845:2:1903,data(2,:),'-o');
hold on
plot(1845:2:1903,real(u_dmd(2,:))');
legend('lynx data','time-delay DMD')
xlabel('t [years]')
ylabel('population')

figure, plot(diag(Sigma2)/sum(diag(Sigma2)),'ko','Linewidth',[2])

figure, plot(omega,'ko','Linewidth',[2])
%% time delay 2

H3=[];
for j=1:10
  H3=[H3; data(:,j:20+j)]; 
end 
ini_cond=H3(:,1);
v1 = H3(:,1:end-1); v2 = H3(:,2:end);

[U2,Sigma2,V2] = svd(v1, 'econ');
r=10; U=U2(:,1:r); Sigma=Sigma2(1:r,1:r); V=V2(:,1:r);
Atilde = U'*v2*V/Sigma;
[W,D] = eig(Atilde);
Phi=v2*V/Sigma*W;

lambda=diag(D);
omega=log(lambda)/dt;

y0 = Phi\ini_cond;  % pseudo-inverse initial conditions

u_modes = zeros(r,length(H3));  % DMD reconstruction for every time point
for iter = 1:length(t)
    u_modes(:,iter) =(y0.*exp(omega*(t(iter))));
end
u_dmd = Phi*u_modes;   % DMD resconstruction with all modes

figure(5);

plot(1845:2:1903,real(u_dmd(1,:))');
hold on
plot(1845:2:1903,real(u_dmd(2,:))');

figure(6), plot(diag(Sigma2)/sum(diag(Sigma2)),'bo','Linewidth',[2])
xlabel('rank')
ylabel('singular value')

figure(7), plot(lambda,'ko','Linewidth',[2])

n=length(data);
noise=0.0;
x1=(data(1,:)+noise*randn(n,1)')';
x2=(data(2,:)+noise*randn(n,1)')';

%%derivatives
dt=2;
% center difference scheme
for j=2:n-1
  x1dot(j-1)=(x1(j+1)-x1(j-1))/(2*dt);
  x2dot(j-1)=(x2(j+1)-x2(j-1))/(2*dt);
end

x1s=x1(2:n-1);
x2s=x2(2:n-1);

A1=[x1s -x1s.*x2s];
% 
xi1=A1\x1dot.';
% 
A2=[x1s.*x2s -x2s];
% 
xi2=A2\x2dot.';

mu=[xi1' xi2'];

[t,y]=ode45('rhs_vdp',t,x0,[],mu);

%% plot fit

figure;

plot(1845:2:1903,data(1,:),'-o');
hold on
plot(1845:2:1903,y(:,1))
legend('hare data','L2 fit')
xlabel('t [years]')
ylabel('population')

figure;
plot(1845:2:1903,data(2,:),'-o');
hold on
plot(1845:2:1903,y(:,2))
legend('lynx data','L2 fit')
xlabel('t [years]')
ylabel('population')

%% SYNDY model

A=[x1s x2s x1s.^2 x1s.*x2s x2s.^2 x1s.^3 (x1s.^2).*x2s x1s.*sin(x1s) x2s.*sin(x2s) sin(x1s) sin(x2s) sin(x1s).*sin(x2s) sin(x1s.*x2s)];

xi1=lasso(A,x1dot.','Lambda',0.1);
xi2=lasso(A,x2dot.','Lambda',0.1);
figure
subplot(2,1,1), bar(xi1)
subplot(2,1,2), bar(xi2)

%% run sindy
mu=[xi1(10:13) xi2(10:13)];



sindy = @(t,x)([mu(1) * sin(x(1)) + mu(2) * sin(x(2)) + mu(3) * sin(x(1)).*sin(x(2)) + mu(4) * sin(x(1).*x(2));
                mu(5) * sin(x(1)) + mu(6) * sin(x(2)) + mu(7) * sin(x(1)).*sin(x(2)) + mu(8) * sin(x(1).*x(2))]);              


            
[t,y]=ode45(sindy,t,x0);

figure;

plot(1845:2:1903,data(1,:),'-o');
hold on
plot(1845:2:1903,y(:,1))
legend('hare data','SINDY')
xlabel('t [years]')
ylabel('population')

figure;
plot(1845:2:1903,data(2,:),'-o');
hold on
plot(1845:2:1903,y(:,2))
legend('lynx data','SINDY')
xlabel('t [years]')
ylabel('population')
