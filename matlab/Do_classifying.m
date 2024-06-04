function [results] = Do_classifying(filename, dist, training, FE, ncluster)

% PROGRAM Do_classifying.
% Does clustering on dataset to get Precision, Recall and F1 score of the
% dataset
% 
% Runs after Do_training, which prepares cluster mean, covariance and
% average distance of each cluters.
% 
% input must be:
%     A string with a filename of dataset in folder 'datasets'
%     optional argument 'dist': for diffrent distance metrics used for
%     training algorithm.(mahalanobis/euclidean)
%     optional argument 'training': for trainig algorithm kmean or
%     kmedoids
%     optional argument 'FE': for dimentions of feature extraction
%     optional argument 'ncluster': for the number of clusters.(for
%     convenience of this simulation, ncluster is always 3) 

%dist = 'mahalanobis'; %Distance metrics used in each run
[C, covariance, ~] = Do_training( filename, dist, training, FE, ncluster);
[~, fnam, ~] = fileparts(filename);
load(['./data_tmp/' fnam '_spikes.mat']);

rng(1, 'twister')   %for repeatable result
r = size(spikes,1);
[~,valInd,~] = dividerand(r, 0.6, 0.2, 0.2);

%cross-validation set
valSpikes = spikes(valInd,:);
valtrueLabels = classInd(valInd,1);

[~, FE_pca] = pca(valSpikes);
valFeature = FE_pca(:, 1:FE);
cov1 = covariance{1,1};
cov2 = covariance{1,2};
cov3 = covariance{1,3};

switch dist
    case 'euclidean'
        [~,valpredictedLabels] = pdist2(C, valFeature, dist,'Smallest',1);
    case 'mahalanobis'
        [ny,py] = size(valFeature);
        for i = 1:ny
%     i = 35;
            dsq1 = pdist2(C, valFeature(i,:), dist, cov1);
            dsq2 = pdist2(C, valFeature(i,:), dist, cov2);
            dsq3 = pdist2(C, valFeature(i,:), dist, cov3);
            D = [dsq1, dsq2, dsq3];
            Dmin = min(D, [], 'all');
            [valpredictedLabels(:,i), ~] = find(D == Dmin);
        end    
    otherwise 
        [~,valpredictedLabels] = pdist2(C, valFeature, dist,'Smallest',1);
end

% figure(4)
% hold on;
% gscatter(valFeature(:,1),valFeature(:,2),valpredictedLabels,'rbg','ooo');
% plot(C(:,1), C(:,2), 'kx', 'MarkerSize',15,'LineWidth',3);
% legend('Data classified to Cluster 1','Data classified to Cluster 2', ...
%     'Data classified to Cluster 3','Cluster Centroid')
% hold off;

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

% figure(5);
% cm = confusionchart(valtrueLabels,valpredictedLabels);
% cm.ColumnSummary = 'column-normalized';
% cm.RowSummary = 'row-normalized';
