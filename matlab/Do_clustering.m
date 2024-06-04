clear all;

% PROGRAM Do_clustering.
% try the difficult datasets with various number of features from 2 to 10 
% and draw the F-score wrt. number of features to study the impact of the 
% number of features over the performance
%
% if you want to observe the overall trend with the change of FE, you
% should set multiFE as TRUE, otherwise you will get the result for
% specified FE.
%
% parameters needed for each simulation:
%     FE: 2-10 (dimensions used in feature extraction)
%     dist: 'euc' / 'mah'
%     make_plots: true / false (for plotting global enable)

filename = 'C_Difficult2_noise02.mat';
dist = 'euc'; %Distance metrics used in each run
make_plots = true;
multiFE = true;
FE = 3;

%use relative path to run code easily on every machine
currentFile = mfilename( 'fullpath' );
[pathstr, name, ~] = fileparts( currentFile );
cd(pathstr);
addpath( fullfile( pathstr ) );

get_spikes(filename);
[~, fnam, ~] = fileparts(filename);
load(['./data_tmp/' fnam '_spikes.mat']);

%train set split
rng(1, 'twister')   %for repeatable result
r = size(spikes,1);

for splt = 0.6:-0.1:0.1
[trainInd,valInd,testInd] = dividerand(r, splt, 0.8-splt, 0.2);
 
trainSpikes = spikes(trainInd,:);
trueLabels = classInd(trainInd,1);

%cross-validation set
valSpikes = spikes(valInd,:);
valtrueLabels = classInd(valInd,1);

if multiFE
    F1 = [];
    for FE = 2:10
        make_plots = false;
        tmp = clustering(trainSpikes, valSpikes, dist, FE, make_plots,...
                trueLabels, valtrueLabels);
        F1 = [F1, tmp(1,3)];
    end
    plot(2:10, F1, '-o', 'LineWidth', 2);
    hold on
    text(5,F1(1,4),['(5,',num2str(F1(1,4)),')'],'color','b');
    xlabel 'numbers of FE'; ylabel 'F1 score';
    legend('val set 20%','val set 30%','val set 40%','val set 50%','val set 60%', 'val set 70%')
end
end

if ~multiFE
    result = clustering(trainSpikes, valSpikes, dist, FE, make_plots,...
                trueLabels, valtrueLabels);
end

function results = clustering(trainSpikes, valSpikes, dist, FE, dispEn,...
                trueLabels, valtrueLabels)

%feature extraction using PCA
[~, FE_pca] = pca(trainSpikes);
trainFeature = FE_pca(:, 1:FE);

%feature extraction using PCA
[~, FE_pca] = pca(valSpikes);
valFeature = FE_pca(:, 1:FE);

if dispEn
    figure(1);
    scatter3(trainFeature(trueLabels==1,1), trainFeature(trueLabels==1,2), trainFeature(trueLabels==1,3), 'r*','Linewidth',1);
    hold on;
    scatter3(trainFeature(trueLabels==2,1), trainFeature(trueLabels==2,2), trainFeature(trueLabels==2,3), 'b*','Linewidth',1);
    scatter3(trainFeature(trueLabels==3,1), trainFeature(trueLabels==3,2), trainFeature(trueLabels==3,3), 'g*','Linewidth',1);
    title 'Extracted Features using PCA';
    xlabel 'feature1'; ylabel 'feature2';
    hold off;
end

%training using Kmedoids
opts = statset('Display','final');
% [predictedLabels,C] = kmedoids(trainFeature,3,'Distance',dist, 'Replicates',5,'Options',opts);
[~,C] = kmeans(trainFeature,3,'Distance','sqeuclidean', 'Replicates',5,'Options',opts);
covariance = {cov(trainFeature(trueLabels==1,:)),...
    cov(trainFeature(trueLabels==2,:)),...
    cov(trainFeature(trueLabels==3,:))};

cov1 = covariance{1,1};
cov2 = covariance{1,2};
cov3 = covariance{1,3};


switch dist
    case 'euc'
        dist_t = 'euclidean';
        [~,valpredictedLabels] = pdist2(C, valFeature, dist_t,'Smallest',1);
    case 'mah'
        dist_t = 'mahalanobis';
        [ny,py] = size(valFeature);
        for i = 1:ny
%     i = 35;
            dsq1 = pdist2(C, valFeature(i,:), dist_t, cov1);
            dsq2 = pdist2(C, valFeature(i,:), dist_t, cov2);
            dsq3 = pdist2(C, valFeature(i,:), dist_t, cov3);
            D = [dsq1, dsq2, dsq3];
            Dmin = min(D, [], 'all');
            [valpredictedLabels(:,i), ~] = find(D == Dmin);
        end    
    otherwise 
        [~,valpredictedLabels] = pdist2(C, valFeature, dist,'Smallest',1);
end

if dispEn
    figure(2)
    scatter3(valFeature(valpredictedLabels==1,1),...
        valFeature(valpredictedLabels == 1,2),valFeature(valpredictedLabels == 1,3),'r','o');
    hold on;
    scatter3(valFeature(valpredictedLabels==2,1),...
        valFeature(valpredictedLabels == 2,2),valFeature(valpredictedLabels == 2,3),'g','o');
    scatter3(valFeature(valpredictedLabels==3,1),...
        valFeature(valpredictedLabels == 3,2),valFeature(valpredictedLabels == 3,3),'b','o');
    plot3(C(:,1), C(:,2), C(:,3), 'kx', 'MarkerSize',15,'LineWidth',3) ;
    legend('Cluster 1','Cluster 2','Cluster 3','Cluster Centroid')
    hold off;
end

valEval = confusionmat(valtrueLabels,valpredictedLabels);
truePositiveP= max(valEval,[],1);
[truePositiveR, Reorder] = max(valEval,[],2);
valPrecision = truePositiveP./sum(valEval,1);
valPrecision = [valPrecision(1,Reorder(1,1)),valPrecision(1,Reorder(2,1)),valPrecision(1,Reorder(3,1))];
valRecall = truePositiveR./sum(valEval,2);
valRecall = valRecall.';

valF1 = 2.*valPrecision.*valRecall./(valPrecision + valRecall);
valPrecision = mean(valPrecision);
valRecall = mean(valRecall);
valF1 = mean(valF1);
results = [valPrecision valRecall valF1];

if dispEn
    figure(3);
    cm = confusionchart(valtrueLabels,valpredictedLabels);
    cm.ColumnSummary = 'column-normalized';
    cm.RowSummary = 'row-normalized';
end
end
