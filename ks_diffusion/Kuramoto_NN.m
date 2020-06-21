clear all; close all; clc

N = 2048;
x = 2*pi*(1:N)'/N;
u=randn(N,1);
v = fft(u);

nu=0.005;

% % % % % %
%Spatial grid and initial condition:
h = 0.01; 
k = [0:N/2-1 0 -N/2+1:-1]';
L = k.^2 - nu*k.^4; %% added the 4* for my specif
E = exp(h*L); E2 = exp(h*L/2);
M = 16;
r = exp(1i*pi*((1:M)-.5)/M);
LR = h*L(:,ones(M,1)) + r(ones(N,1),:);
Q = h*real(mean( (exp(LR/2)-1)./LR ,2));
f1 = h*real(mean( (-4-LR+exp(LR).*(4-3*LR+LR.^2))./LR.^3 , 2) );
f2 = h*real(mean( (2+LR+exp(LR).*(-2+LR))./LR.^3 ,2));
f3 = h*real(mean( (-4-3*LR-LR.^2+exp(LR).*(4-LR))./LR.^3 ,2));

% Main time-stepping loop:
uu = u; tt = 0;
tmax = 10; nmax = round(tmax/h); nplt = floor((tmax/1000)/h); g = -0.5i*k;
tt = zeros(1,nmax);


for n = 1:nmax
    t = n*h;
    Nv = g.*fft(real(ifft(v)).^2);
    a = E2.*v + Q.*Nv;
    Na = g.*fft(real(ifft(a)).^2); 
    b = E2.*v + Q.*Na;
    Nb = g.*fft(real(ifft(b)).^2);
    c = E2.*a + Q.*(2*Nb-Nv);
    Nc = g.*fft(real(ifft(c)).^2);
    v = E.*v + Nv.*f1 + 2*(Na+Nb).*f2 + Nc.*f3;     
    if mod(n,nplt)==0
        n;
        u = real(ifft(v));
        uu(:,n) = u; 
        tt(n) = t;
    end
end

cutoff = tt > 0;
cutoff = cutoff & tt<0.6;

pcolor(x,tt(cutoff),uu(:,cutoff).'),shading interp, colormap(hot), %view(15,50)
tsave = tt(cutoff);
xsave = x/(2*pi);
dt = h;
dx = 1/N;
usave = uu(:,cutoff).';

%%
uu2=uu(:,cutoff); [mm,nn]=size(uu2);
for j=1:nn
    uut(:,j)=abs(fftshift(fft(uu2(:,j))));
end
k=[0:N/2-1 -N/2:-1].';
ks=fftshift(k);
figure(4)
surfl(ks,tt(cutoff),(uut(:,cutoff).')),shading interp, colormap(hot), %view(15,50)
figure(5)
waterfall(ks,tt(cutoff),(uut(:,cutoff).')), colormap([0 0 0])
set(gca,'Xlim',[-50 50])

%% train NN

input=uu2(:,1:end-1)';

output=uu2(:,2:end)';

net = feedforwardnet([10 10 10 10]);
net1.layers{1}.transferFcn='logsig';
 net1.layers{2}.transferFcn='logsig';
 net1.layers{3}.transferFcn='logsig';
net.layers{4}.transferFcn = 'purelin';
net = train(net,input.',output.');

save workspace

%% noisy data and net NN
uu2_old=uu2;

N = 2048;
x = 2*pi*(1:N)'/N;
u=randn(N,1);
v = fft(u);

nu=0.005;

% % % % % %
%Spatial grid and initial condition:
h = 0.01; 
k = [0:N/2-1 0 -N/2+1:-1]';
L = k.^2 - nu*k.^4; %% added the 4* for my specif
E = exp(h*L); E2 = exp(h*L/2);
M = 16;
r = exp(1i*pi*((1:M)-.5)/M);
LR = h*L(:,ones(M,1)) + r(ones(N,1),:);
Q = h*real(mean( (exp(LR/2)-1)./LR ,2));
f1 = h*real(mean( (-4-LR+exp(LR).*(4-3*LR+LR.^2))./LR.^3 , 2) );
f2 = h*real(mean( (2+LR+exp(LR).*(-2+LR))./LR.^3 ,2));
f3 = h*real(mean( (-4-3*LR-LR.^2+exp(LR).*(4-LR))./LR.^3 ,2));

% Main time-stepping loop:
uu = u; tt = 0;
tmax = 10; nmax = round(tmax/h); nplt = floor((tmax/1000)/h); g = -0.5i*k;
tt = zeros(1,nmax);

for n = 1:nmax
    t = n*h;
    Nv = g.*fft(real(ifft(v)).^2);
    a = E2.*v + Q.*Nv;
    Na = g.*fft(real(ifft(a)).^2); 
    b = E2.*v + Q.*Na;
    Nb = g.*fft(real(ifft(b)).^2);
    c = E2.*a + Q.*(2*Nb-Nv);
    Nc = g.*fft(real(ifft(c)).^2);
    v = E.*v + Nv.*f1 + 2*(Na+Nb).*f2 + Nc.*f3;     
    if mod(n,nplt)==0
        n;
        u = real(ifft(v));
        uu(:,n) = u; 
        tt(n) = t;
    end
end



cutoff = tt > 0;
cutoff = cutoff & tt<0.6;

tsave = tt(cutoff);
xsave = x/(2*pi);
dt = h;
dx = 1/N;
usave = uu(:,cutoff).';


uu2=uu(:,cutoff); [mm,nn]=size(uu2);
for j=1:nn
    uut(:,j)=abs(fftshift(fft(uu2(:,j))));
end
k=[0:N/2-1 -N/2:-1].';
ks=fftshift(k);
figure(4)
surfl(ks,tt(cutoff),(uut(:,cutoff).')),shading interp, colormap(hot), %view(15,50)
figure(5)
waterfall(ks,tt(cutoff),(uut(:,cutoff).')), colormap([0 0 0])
set(gca,'Xlim',[-50 50])

x0=uu2(:,1);
ynn(:,1)=x0;
for jj=2:size(uu2,2)
    y0=net(x0);
    ynn(:,jj)=y0; x0=y0;
end

uu2_new=uu2;

figure;
pcolor(x,tt(cutoff),uu2_old(:,cutoff).'),shading interp, colormap(hot), %view(15,50)
xlabel('space')
ylabel('time')
colorbar
figure;
pcolor(x,tt(cutoff),uu2_new(:,cutoff).'),shading interp, colormap(hot), %view(15,50)
xlabel('space')
ylabel('time')
colorbar
figure;
pcolor(x,tt(cutoff),ynn(:,cutoff).'),shading interp, colormap(hot), %view(15,50)
xlabel('space')
ylabel('time')
colorbar
x0=uu2_old(:,1);
ynn(:,1)=x0;
for jj=2:size(uu2,2)
    y0=net(x0);
    ynn(:,jj)=y0; x0=y0;
end
figure;
pcolor(x,tt(cutoff),ynn(:,cutoff).'),shading interp, colormap(hot), %view(15,50)
xlabel('space')
ylabel('time')
colorbar
