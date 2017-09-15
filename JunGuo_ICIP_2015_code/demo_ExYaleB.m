close all;
clear all;
clc;


%% settings
addpath(genpath('.\files'));
load('ExYaleB.mat');
% we recommend to pre-process data via 'normcols.m' for other datasets.
training_feats = normcols(training_feats);	
testing_feats = normcols(testing_feats);
dictsize = 570;   % = 38(classes)*15(samples per class)
kNumNN = 7;       % number of NN-graph
lamda = 2e-1; 
alpha = 5e-4;
tau = 1e-6;


%% initialization
fprintf('\nLSDDL initialization... ');
tic
LapMat = getL(training_feats,H_train,kNumNN,lamda); % Laplacian matrix
[Dinit,Xinit] = INITdic(training_feats,H_train,dictsize);% initialization
TimeForInit = toc;
fprintf('Done!\n');
fprintf('Running time for initialization: %.03f seconds.\n', TimeForInit);


%% LSDDL training
fprintf('\nLSDDL training...');
tic
D = LSDDL(training_feats,Dinit,Xinit,LapMat,alpha,tau);
TimeForLS = toc;
fprintf('Done!\n');
fprintf('Training time: %.03f seconds.\n', TimeForLS);


%% classification
fprintf('\nLSDDL testing...');
tic % 1-NN classification
accuracy = NN_classify(D,training_feats,testing_feats,H_train,H_test,tau);
TtTimeForLS = toc;
fprintf('Done!\n');
fprintf('Testing time: %.06f seconds per sample.\n', TtTimeForLS/size(testing_feats,2));
fprintf('\nAccuracy is %.01f%%.\n', accuracy*100);