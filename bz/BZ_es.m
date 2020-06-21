clear all
close all
clc

load('BZ.mat')

[m,n,k]=size(BZ_tensor); % x vs y vs time data
% for j=1:k
% A=BZ_tensor(:,:,j);
% pcolor(A), shading interp, pause(0.2)
% end

%% SVD
a=reshape(BZ_tensor,351*451,1200);
v1 = a(:,1:end-1); v2 = a(:,2:end);
[U2,Sigma2,V2] = svd(v1,'econ');
figure, plot(diag(Sigma2)/sum(diag(Sigma2)),'ko','Linewidth',[2])
xlabel('mode')
ylabel('singular value')
% v_proj=U'*v1;
save svdspace
%% feature space
figure;
for i=1:4
    subplot(2,2,i)
    pcolor(1:m,1:n,reshape(U2(:,i),351,451)'); shading interp;
    xlabel('x')
    ylabel('y')
    colorbar
    title(strcat('feature ',num2str(i)))
    
end

dt=1;


%1 modes
r=1; U=U2(:,1:r); Sigma=Sigma2(1:r,1:r); V=V2(:,1:r);
Atilde = U'*v2*V/Sigma;
[W,D] = eig(Atilde);
Phi=v2*V/Sigma*W;

lambda=diag(D);
omega=log(lambda)/dt;
ini_cond=a(:,1);

t=1:1200;
y0 = Phi\ini_cond;  % pseudo-inverse initial conditions

u_modes = zeros(r,length(t));  % DMD reconstruction for every time point
for iter = 1:length(t)
    u_modes(:,iter) =(y0.*exp(omega*(t(iter))));
end
u_dmd = Phi*u_modes;   % DMD resconstruction with all modes

figure;
j=randi(1200,1,1);
subplot(2,2,1)
A=BZ_tensor(:,:,j);
pcolor(A), shading interp
xlabel('x')
ylabel('y')
title('snapshot BZ')
subplot(2,2,2)
B=reshape(u_dmd,351,451,1200);
pcolor(real(B(:,:,j))), shading interp
xlabel('x')
ylabel('y')
title('DMD 1 mode')
colorbar


%%30 modes
r=30; U=U2(:,1:r); Sigma=Sigma2(1:r,1:r); V=V2(:,1:r);

Atilde = U'*v2*V/Sigma;
[W,D] = eig(Atilde);
Phi=v2*V/Sigma*W;

lambda=diag(D);
omega=log(lambda)/dt;
ini_cond=a(:,1);

t=1:1200;
y0 = Phi\ini_cond;  % pseudo-inverse initial conditions

u_modes = zeros(r,length(t));  % DMD reconstruction for every time point
for iter = 1:length(t)
    u_modes(:,iter) =(y0.*exp(omega*(t(iter))));
end
u_dmd = Phi*u_modes;   % DMD resconstruction with all modes

subplot(2,2,3)
B=reshape(u_dmd,351,451,1200);
pcolor(real(B(:,:,j))), shading interp
xlabel('x')
ylabel('y')
colorbar
title('DMD 30 modes')

%%100 modes
r=100; U=U2(:,1:r); Sigma=Sigma2(1:r,1:r); V=V2(:,1:r);

Atilde = U'*v2*V/Sigma;
[W,D] = eig(Atilde);
Phi=v2*V/Sigma*W;

lambda=diag(D);
omega=log(lambda)/dt;
ini_cond=a(:,1);

t=1:1200;
y0 = Phi\ini_cond;  % pseudo-inverse initial conditions

u_modes = zeros(r,length(t));  % DMD reconstruction for every time point
for iter = 1:length(t)
    u_modes(:,iter) =(y0.*exp(omega*(t(iter))));
end
u_dmd = Phi*u_modes;   % DMD resconstruction with all modes

subplot(2,2,4)
B=reshape(u_dmd,351,451,1200);
pcolor(real(B(:,:,j))), shading interp
xlabel('x')
ylabel('y')
colorbar
title('DMD 100 modes')