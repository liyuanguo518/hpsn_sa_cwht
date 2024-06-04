function [Centroids, covariance, avgD] = Do_training( filename, distance, training, FE, ncluster )

% PROGRAM Do_training.
% Does training on dataset to get cluster mean, covariance and average
% distance of each clusters
% 
% Runs after Get_spikes, which prepares '*_spikes.mat' files.
% 
% input must be:
%     A string with a filename of dataset in folder 'datasets'
%     optional argument 'distance': for diffrent distance metrics used for
%     training algorithm.(mahalanobis/euclidean)
%     optional argument 'training': for trainig algorithm kmean or
%     kmedoids
%     optional argument 'FE': for dimentions of feature extraction
%     optional argument 'ncluster': for the number of clusters.(for
%     convenience of this simulation, ncluster is always 3) 

%use relative path to run code easily on every machine
currentFile = mfilename( 'fullpath' );
[pathstr, ~, ~] = fileparts( currentFile );
cd(pathstr);
addpath( fullfile( pathstr ) );

get_spikes(filename);
[~, fnam, ~] = fileparts(filename);
load(['./data_tmp/' fnam '_spikes.mat']);

%train set split
rng(1, 'twister')   %for repeatable result
r = size(spikes,1);
[trainInd,~,~] = dividerand(r, 0.6, 0.2, 0.2);
 
trainSpikes = spikes(trainInd,:);
trueLabels = classInd(trainInd,1);

%feature extraction using PCA
[~, FE_pca] = pca(trainSpikes);
trainFeature = FE_pca(:, 1:FE);

%running clustering algorithm
opts = statset('Display','final');
switch training
    case 'kmedoid'
        [predictedLabels, Centroids, sumD] = kmedoids(trainFeature,ncluster,'Distance',distance, 'Replicates',5,'Options',opts);
    case 'kmean'
        [predictedLabels, Centroids, sumD] = kmeans(trainFeature,ncluster,'Distance','sqeuclidean', 'Replicates',5,'Options',opts);
    otherwise
        [predictedLabels, Centroids, sumD] = kmeans(trainFeature,ncluster,'Distance','sqeuclidean', 'Replicates',5,'Options',opts);
end
% covariance = cov(trainFeature);
clusN = [length(find(predictedLabels == 1));length(find(predictedLabels == 2));length(find(predictedLabels == 3))];
avgD = sumD./clusN;
covariance = {cov(trainFeature(trueLabels==1,:)),...
    cov(trainFeature(trueLabels==2,:)),...
    cov(trainFeature(trueLabels==3,:))};
