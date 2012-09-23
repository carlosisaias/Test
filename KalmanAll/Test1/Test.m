clc;
clear all;
close all;

addpath(genpath('C:\Users\Carlos\Dropbox\RA\KalmanAll'))
clc;
%% Parameters
n = 500:100:800;      %size of full system
Ti=5000;                 %Total Time periods simulated
ti=200;                   %Time periods to be kept
theta = -0.9;  %simple parameter determining model.  should be in (-1,1).
s=1;                   %Number of Simulations

modelReductionMethod = 'balred';  % which model reduction technique to use?
                                  % 'balred' -> requires Matalb control
                                  % system toolbox and does balanced
                                  % reduction with MatchDC.  This is more accurate but
                                  % slower.
                                  % 'Lanczos' -> faster, but somewhat less
                                  % accurate (default).
                                  % 'Truncred' -> requires Matlab control
                                  % system toolbox and does balanced
                                  % reduction with Truncation.
                                  
                                  


h=size(theta,2);                                  
for e=1:1:h                                  

%% Size of Reduced Model
if theta(e)==-0.9 || theta(e)==0.9;
    u=39;
elseif theta(e)==-0.8|| theta(e)==0.8;
    u=30;
elseif theta(e)==-0.7|| theta(e)==0.7;
    u=24;
elseif theta(e)==-0.6|| theta(e)==0.6;
    u=19;
elseif theta(e)==-0.5|| theta(e)==05;
    u=16;
elseif theta(e)==-0.4|| theta(e)==0.4;
    u=14;
elseif theta(e)==-0.3|| theta(e)==0.3;
    u=12;
else u=10;
end
    
    
k = 1:1:u;            %size of reduced system (for the systems I have studied here, k must be small i.e. <= 12)
                                  
%% Initialization
l=size(k,2);
o=size(n,2);
t=zeros(l,2,o,s);   
predata=zeros(o,Ti);
data=zeros(o,ti);
loglike=zeros(l,o,s);

%% Generate Model
for p=1:1:o
[A, B, C, D] = theta2model(theta(e), n(p));

for i=1:s
%% Model data generation
initx=ones(size(A,1),1);
initV=eye(size(C,1));
[~,predata(p,:)]=sample_lds(A,C,B*B',D*D',initx,Ti);
data(p,:)=predata(p,Ti-ti+1:Ti);


%% Model Kalman Filter
%[~,~,~,loglike(p,1)]=kalman_filter(data(p,:),A,C,B*B',D*D',initx,initV);


%% Reduced Model
for j=1:l
clc
display(sprintf('Theta=%g',theta(e)))
display(sprintf('k=%g',k(j))) 
display(sprintf('n=%g',n(p)))
tic

if strcmpi(modelReductionMethod, 'balred')
    disp('Reducing model with balanced reduction with MatchDC.')
    [Ahat, Bhat, Chat] = balred_wrapped(A,B,C,D, k(j));

elseif strcmpi(modelReductionMethod, 'Truncred')
    disp('Reducing model with balanced reduction with Truncation.')
    [Ahat, Bhat, Chat] = balred_wrapped2(A,B,C,D, k(j));   
    
else    
    disp('Reducing model with Lanczos.')
    [V, W, T, f, g] = NSLanczos(A,k(j),B);

    Ahat = W'*A*V;
    Bhat = W' * B;
    Chat = C*V;
end
t(j,1,p,i)=toc;
display(sprintf('Time taken by model reduction step:%g',t(j,1,p)))


assert(all(abs(eig(Ahat)) < 0.99999999), 'reduced system is not stable')


%% Kalman Filter for reduced model
tic;
initx=ones(size(Ahat,1),1);
initV=ones(size(Chat,1));
[~,~,~,loglike(j,p,i)]=kalman_filter(data(p,:),Ahat,Chat,Bhat*Bhat',D*D',initx,initV);
t(j,2,p,i)=toc;
display(sprintf('Time taken to estimate loglikelihood with Kalman Filter:%g', t(j,2,p)))

end
end
%% Estimating Means
meant=mean(t,4);
meanloglike=mean(loglike,3);

%% Plotting Results

%Kalman Time results
plot(meant(:,2,p))
title(sprintf('Time to estimate loglikelihood with Kalman Filter, Theta=%g, n=%g,',theta(e),n(p)))
xlabel('K')
ylabel('Seconds')
saveas(gcf,['Time to estimate loglikelihood with Kalman Filter,',num2str(theta(e)),', ' ,num2str(n(p)),'.fig'])

plot(meant(:,1,p))
title(sprintf('Time to reduce model, Theta=%g, n=%g', theta(e),n(p)))
xlabel('K')
ylabel('Seconds')
saveas(gcf,['Time to reduce model,',num2str(theta(e)),', ' ,num2str(n(p)),'.fig'])

plot(meanloglike(:,p))
title(sprintf('Loglikelihood of reduced models, Theta=%g, n=%g', theta(e), n(p)))
xlabel('K')
saveas(gcf,['Loglikelihood of reduced models,',num2str(theta(e)),', ' ,num2str(n(p)),'.fig'])

close all


end
end
%% 

