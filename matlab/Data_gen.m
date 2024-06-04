clear all;

% PROGRAM Data_gen.
% manually genarate a 3000 samples dataset with specified covariance to
% test the performance of mahalanobis distance metric.

% parameters needed for each simulation:
%     clustering: 'kmedoid' / 'kmean'
%     dist: 'euclidean' / 'mahalanobis'
%     dispEn: true / false (for plotting global enable)
clustering = 'kmedoid';
dist = 'mahalanobis';
dispEn = false;

mu1 = [2 3];
Sigma1 = [1 1.5; 1.5 3];
rng(1,'twister')  % For reproducibility
R1 = mvnrnd(mu1,Sigma1,1000);

mu2 = [0 -1];
Sigma2 = [1 -1.5; -1.5 3];
rng(1,'twister')  % For reproducibility
R2 = mvnrnd(mu2,Sigma2,1000);

mu3 = [0 4];
Sigma3 = [1 0; 0 0.2];
rng(1,'twister')  % For reproducibility
R3 = mvnrnd(mu3,Sigma3,1000);

if dispEn
    figure(1)
    plot(R1(:,1), R1(:,2), 'k*','MarkerSize', 5);
    hold on
    plot(R2(:,1), R2(:,2), 'b*','MarkerSize', 5);
    plot(R3(:,1), R3(:,2), 'r*','MarkerSize', 5);
    hold off
end

spikes = [R1;R2;R3];
rng(1,'twister')  % For reproducibility
RandIndex = randperm(length(spikes)); 

spikes = spikes(RandIndex.',:);
a = ones([1000,1]);
classInd = [a;2.*a;3.*a];
classInd = classInd(RandIndex);

%train set split
rng(1, 'twister')   %for repeatable result
r = size(spikes,1);
[trainInd,~,~] = dividerand(r, 0.6, 0.2, 0.2);
trainFeature = spikes(trainInd,:);
trueLabels = classInd(trainInd,1);

%running clustering algorithm
opts = statset('Display','final');
switch clustering
    case 'kmedoid'
        [predictedLabels,Centroids] = kmedoids(trainFeature,3,'Distance',dist, 'Replicates',5,'Options',opts);
    case 'kmean'
        [predictedLabels,Centroids] = kmeans(trainFeature,3,'Distance','sqeuclidean', 'Replicates',5,'Options',opts);
    otherwise
        [predictedLabels,Centroids] = kmeans(trainFeature,3,'Distance','sqeuclidean', 'Replicates',5,'Options',opts);
end
covariance = {cov(trainFeature(trueLabels==1,:)),...
    cov(trainFeature(trueLabels==2,:)),...
    cov(trainFeature(trueLabels==3,:))};

cov1 = covariance{1,1};
cov2 = covariance{1,2};
cov3 = covariance{1,3};

if dispEn
    figure(2);
    plot(trainFeature(predictedLabels==1,1), trainFeature(predictedLabels==1,2), 'r.', 'MarkerSize', 12);
    hold on;
    plot(trainFeature(predictedLabels==2,1), trainFeature(predictedLabels==2,2), 'b.', 'MarkerSize', 12);
    plot(trainFeature(predictedLabels==3,1), trainFeature(predictedLabels==3,2), 'g.', 'MarkerSize', 12);
    plot(Centroids(:,1), Centroids(:,2), 'kx', 'MarkerSize',15,'LineWidth',3) ;
    legend('Cluster 1','Cluster 2','Cluster 3','Centroids', 'Location','NW');
    title 'Cluster Assignments and Centroids'
    hold off
end

rng(1, 'twister')   %for repeatable result
r = size(spikes,1);
[~,valInd,~] = dividerand(r, 0.6, 0.2, 0.2);

%cross-validation set
valFeature = spikes(valInd,:);
valtrueLabels = classInd(valInd,1);

switch dist
    case 'euclidean'
        [~,valpredictedLabels] = pdist2(Centroids, valFeature, dist,'Smallest',1);
    case 'mahalanobis'
%         [a,valpredictedLabels] = pdist2(Centroids, valFeature, dist, Sigma2,'Smallest',1);
        [ny,py] = size(valFeature);
        for i = 1:ny
%     i = 35;
            dsq1 = pdist2(Centroids, valFeature(i,:), dist, cov1);
            dsq2 = pdist2(Centroids, valFeature(i,:), dist, cov2);
            dsq3 = pdist2(Centroids, valFeature(i,:), dist, cov3);
            D = [dsq1, dsq2, dsq3];
            Dmin = min(D, [], 'all');
            [valpredictedLabels(:,i), ~] = find(D == Dmin);
        end    
    otherwise 
        [~,valpredictedLabels] = pdist2(Centroids, valFeature, dist,'Smallest',1);
end
if dispEn
    figure(4)
    hold on;
    gscatter(valFeature(:,1),valFeature(:,2),valpredictedLabels,'rbg','ooo');
    plot(Centroids(:,1), Centroids(:,2), 'kx', 'MarkerSize',15,'LineWidth',3);
    legend('Data classified to Cluster 1','Data classified to Cluster 2', ...
        'Data classified to Cluster 3','Cluster Centroid')
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
    figure(5);
    cm = confusionchart(valtrueLabels,valpredictedLabels);
    cm.ColumnSummary = 'column-normalized';
    cm.RowSummary = 'row-normalized';
end