% Deep Matrix Factorization Clustering performance on jaffe dataset
% 
% four methods:
%             DGsnMF, Haonan Huang, 2020 
%             Deep nsNMF, Jinshi Yu, 2018 
%             Deep SemiNMF, George Trigeorgis, 2017
%             Deep SemiNMF nonlinear, George Trigeorgis, 2017 
%
% 
% jaffe dataset Reference:
%       M. Lyons, S. Akamatsu, M. Kamachi, and J. Gyoba,
%       "Coding facial expressions with gabor wavelets,"
%       In Proc. Third IEEE ICAFGR, Nara, Japan, Apr. 1998, pp. 200-205  
% 
% Created by Haonan Huang, 2020 
clear;
addpath(genpath('../DeepMF'));
load('jaffe.mat');

nClass = length(unique(gnd));

f_layers_r = 200; % the r of the first layers
s_layers_r = 20;  % the r of the second layers

maxtime = 1000;
G_lambda=1e-3; 

%Normalize each data vector to have L2-norm equal to 1  
fea = NormalizeFea(fea);
 %% Clustering in the DGsnMF subspace
 rand('twister',5489);
 tic;
 [Z, H] = DGsnMF(fea', [f_layers_r  s_layers_r],...
     'maxiter', maxtime,'lambdas',G_lambda);
 time = toc;

[ AC, MIhat ] = evalResults(H, gnd );
disp(['Clustering in the DGsnMF .AC/MI/time(', num2str(nClass), '): ' num2str(AC), '/', num2str(MIhat),'/',num2str(time)]);

%% Clustering in the Deep nsNMF subspace
 rand('twister',5489);
 tic;
 [Z, H] = deep_nsnmf(fea', [f_layers_r  s_layers_r]);
 time = toc;

[ AC, MIhat ] = evalResults(H, gnd );
disp(['Clustering in the Deep nsNMF .AC/MI/time(', num2str(nClass), '): ' num2str(AC), '/', num2str(MIhat),'/',num2str(time)]);

%% Clustering in the Deep Semi-NMF subspace
 rand('twister',5489);
 tic;
 [Z, H] = deep_seminmf(fea', [f_layers_r  s_layers_r],...
     'maxiter', maxtime);
 time = toc;

[ AC, MIhat ] = evalResults(H, gnd );
disp(['Clustering in the Deep Semi-NMF .AC/MI/time(', num2str(nClass), '): ' num2str(AC), '/', num2str(MIhat),'/',num2str(time)]);

%% Clustering in the Deep Semi-NMF nonlinear subspace
 rand('twister',5489);
 tic;
 [Z, H] = deep_seminmf_nonlinear(fea', [f_layers_r  s_layers_r], 'gnd',gnd);
 time = toc;

[ AC, MIhat ] = evalResults(H, gnd );
disp(['Clustering in the Deep Semi-NMF nonlinear .AC/MI/time(', num2str(nClass), '): ' num2str(AC), '/', num2str(MIhat),'/',num2str(time)]);



