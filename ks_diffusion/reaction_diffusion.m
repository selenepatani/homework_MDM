clear all; close all; clc

t=0:0.05:10;
d1=0.1; d2=0.1; beta=1.0;
L=20; n=512; N=n*n;
x2=linspace(-L/2,L/2,n+1); x=x2(1:n); y=x;
kx=(2*pi/L)*[0:(n/2-1) -n/2:-1]; ky=kx;

% INITIAL CONDITIONS

[X,Y]=meshgrid(x,y);
[KX,KY]=meshgrid(kx,ky);
K2=KX.^2+KY.^2; K22=reshape(K2,N,1);

m=1; % number of spirals

u = zeros(length(x),length(y),length(t));
v = zeros(length(x),length(y),length(t));

u(:,:,1)=tanh(sqrt(X.^2+Y.^2)).*cos(m*angle(X+i*Y)-(sqrt(X.^2+Y.^2)));
v(:,:,1)=tanh(sqrt(X.^2+Y.^2)).*sin(m*angle(X+i*Y)-(sqrt(X.^2+Y.^2)));

% REACTION-DIFFUSION
uvt=[reshape(fft2(u(:,:,1)),1,N) reshape(fft2(v(:,:,1)),1,N)].';
[t,uvsol]=ode45('reaction_diffusion_rhs',t,uvt,[],K22,d1,d2,beta,n,N);


for j=1:length(t)-1
ut=reshape((uvsol(j,1:N).'),n,n);
vt=reshape((uvsol(j,(N+1):(2*N)).'),n,n);
u(:,:,j+1)=real(ifft2(ut));
v(:,:,j+1)=real(ifft2(vt));

figure(1)
pcolor(x,y,v(:,:,j+1)); shading interp; colormap(hot); colorbar; drawnow; 
end

save('reaction_diffusion_big.mat','t','x','y','u','v')

%%
load reaction_diffusion_big
pcolor(x,y,u(:,:,end)); shading interp; colormap(hot)

%% SVD
a=reshape(u,512*512,201);
b=reshape(v,512*512,201);
v1=[a;b];
[U2,Sigma2,V2] = svd(v1,'econ');
figure, plot(diag(Sigma2)/sum(diag(Sigma2)),'ko','Linewidth',[2])
xlabel('mode')
ylabel('singular value')
r=4; U=U2(:,1:r); Sigma=Sigma2(1:r,1:r); V=V2(:,1:r);
v_proj=U'*v1;

%% train NN

input=v_proj(:,1:end-1)';

output=v_proj(:,2:end)';

%%
net = feedforwardnet([20 20 20]);
net.layers{1}.transferFcn = 'logsig';
net.layers{2}.transferFcn = 'radbas';
net.layers{3}.transferFcn = 'purelin';
net = train(net,input.',output.');

save workspace_rdiff

%% prediction with noise
% load workspace_rdiff

t=0:0.05:10;
d1=0.1; d2=0.1; beta=1.0;
L=20; n=512; N=n*n;
x2=linspace(-L/2,L/2,n+1); x=x2(1:n); y=x;
kx=(2*pi/L)*[0:(n/2-1) -n/2:-1]; ky=kx;

% INITIAL CONDITIONS

[X,Y]=meshgrid(x,y);
[KX,KY]=meshgrid(kx,ky);
K2=KX.^2+KY.^2; K22=reshape(K2,N,1);

m=1; % number of spirals

u = zeros(length(x),length(y),length(t));
v = zeros(length(x),length(y),length(t));


noise=0.1*randn(512,512,1);
u(:,:,1)=tanh(sqrt(X.^2+Y.^2)).*cos(m*angle(X+i*Y)-(sqrt(X.^2+Y.^2)))+noise;
v(:,:,1)=tanh(sqrt(X.^2+Y.^2)).*sin(m*angle(X+i*Y)-(sqrt(X.^2+Y.^2)))+noise;


% REACTION-DIFFUSION
uvt=[reshape(fft2(u(:,:,1)),1,N) reshape(fft2(v(:,:,1)),1,N)].';
[t,uvsol]=ode45('reaction_diffusion_rhs',t,uvt,[],K22,d1,d2,beta,n,N);

for j=1:length(t)-1
ut=reshape((uvsol(j,1:N).'),n,n);
vt=reshape((uvsol(j,(N+1):(2*N)).'),n,n);
u(:,:,j+1)=real(ifft2(ut));
v(:,:,j+1)=real(ifft2(vt));

figure(1)
pcolor(x,y,v(:,:,j+1)); shading interp; colormap(hot); colorbar; drawnow; 
end
save workspace_icchanged

%% projection onto sv

a=reshape(u,512*512,201);
b=reshape(v,512*512,201);
v1=[a;b];

x0=U'*v1;

x0=x0(:,1);

ynn(1,:)=x0;
for jj=2:length(t)
    y0=net(x0);
    ynn(jj,:)=y0.'; x0=y0;
end

v1_new=v1;
y_fin_nn=U*ynn';

figure;
pcolor(x,y,u(:,:,end)); shading interp; colormap(hot)
xlabel('x')
ylabel('y')
colorbar
figure;
pcolor(x,y,reshape(y_fin_nn(1:524288/2,201),512,512)); shading interp; colormap(hot)
xlabel('x')
ylabel('y')
colorbar

figure;
for i=1:4
    subplot(2,2,i)
    pcolor(x,y,reshape(U(1:524288/2,i),512,512)); shading interp; colormap(hot)
    xlabel('x')
    ylabel('y')
    title(strcat('feature ',num2str(i)))
    
end

figure;
pcolor(x,y,v(:,:,end)); shading interp; colormap(hot)
xlabel('x')
ylabel('y')
colorbar
figure;
pcolor(x,y,reshape(y_fin_nn(524288/2+1:end,201),512,512)); shading interp; colormap(hot)
xlabel('x')
ylabel('y')
colorbar

figure;
for i=1:4
    subplot(2,2,i)
    pcolor(x,y,reshape(U(524288/2+1:end,i),512,512)); shading interp; colormap(hot)
     xlabel('x')
    ylabel('y')
    title(strcat('feature ',num2str(i)))
   
end