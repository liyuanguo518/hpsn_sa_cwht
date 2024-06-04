clear all;

% PROGRAM Do_clustering_N.
% Dose clustering on the dataset with noise

% parameters needed for each simulation:
%     FE: 2-10 (dimensions used in feature extraction)
%     dist: 'euc' / 'mah'
%     make_plots: true / false (for plotting global enable)
%     par_val: standard parameter for the calculation of valadation thr.

filename = 'C_Difficult1_noise02.mat';
dist = 'mah'; %Distance metrics used in each run
FE = 3;
make_plots = true;
par_val = 4;

%use relative path to run code easily on every machine
currentFile = mfilename( 'fullpath' );
[pathstr, name, ~] = fileparts( currentFile );
cd(pathstr);
addpath( fullfile( pathstr ) );

get_spikes(filename);
[~, fnam, ~] = fileparts(filename);
load(['./data_tmp/' fnam '_spikes.mat']);

r = size(spikes,1);
trainInd = 1:(r*0.6);

trainSpikes = spikes(trainInd,:);
trueLabels = classInd(trainInd,1);

%feature extraction using PCA
[~, FE_pca] = pca(trainSpikes);
trainFeature = FE_pca(:, 1:FE);

if make_plots
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
[predictedLabels, C, sumD, allD] = kmeans(trainFeature,3,'Distance','sqeuclidean', 'Replicates',5,'Options',opts);
clusN = [length(find(predictedLabels == 1));length(find(predictedLabels == 2));length(find(predictedLabels == 3))];
avgD = sumD./clusN;
std_par = [var(allD(predictedLabels == 1,1));...
    var(allD(predictedLabels == 2,2));var(allD(predictedLabels == 2,2))];
covariance = {cov(trainFeature(trueLabels==1,:)),...
    cov(trainFeature(trueLabels==2,:)),...
    cov(trainFeature(trueLabels==3,:))};

cov1 = covariance{1,1};
cov2 = covariance{1,2};
cov3 = covariance{1,3};

load (['../datasets/' filename ], 'data');
load (['../datasets/' filename ], 'spike_times');
load (['../datasets/' filename ], 'spike_class');

spike_t = spike_times{1,1};
valSpikes = [];
valtrueLabels = [];
intl = [];
n = fix(r*0.6) + 1;

for i = n : fix(r*0.8)
    p = data(spike_t(i) : spike_t(i) + 63);
    intl = [intl; spike_t(i) : spike_t(i) + 63];
    valSpikes = [valSpikes; p];
    valtrueLabels = [valtrueLabels; spike_class{1,1}(1,i)];
end

spikelen = length(valtrueLabels);
spike_detect = zeros(spikelen, 1);
noise_std_detect = median((abs(data)) ./ 0.6745);
thrN = 4 * noise_std_detect;

for j = spike_t( n ) : spike_t( fix( r*0.8 ) )
    if abs(data(j)) >= thrN
        if ismember(j, intl)
            [q, ~] = find(intl == j);
            spike_detect(q, :) = 1;
        else
            p = data(j - 31 : j + 32);
            valSpikes = [valSpikes ; p];
            valtrueLabels = [valtrueLabels; 4];
            spike_detect = [spike_detect; 1];
        end
    end    
end

% valSpikes = valSpikes(spike_detect == 1, :);
[~, FE_pca] = pca(valSpikes);
valFeature = FE_pca(:, 1:FE);
thrval = par_val .* avgD ./ std_par;

switch dist
    case 'euc'
        dist_t = 'euclidean';
        [ny,py] = size(valFeature);
        for i = 1:ny
%     i = 35;
            dsq1 = pdist2(C, valFeature(i,:), dist_t);
            dsq2 = pdist2(C, valFeature(i,:), dist_t);
            dsq3 = pdist2(C, valFeature(i,:), dist_t);
            D = [dsq1, dsq2, dsq3];
            Dmin = min(D, [], 'all');
            [tmpInd, ~] = find(D == Dmin);
            
            if length(tmpInd) > 1
                tmpInd = tmpInd(1,1);
            end
            
            if Dmin <= thrval(tmpInd,1)
                valpredictedLabels(1, i) = tmpInd;
            else
                valpredictedLabels(1, i) = 4;
            end
        end    
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
            [tmpInd, ~] = find(D == Dmin);
            if Dmin <= thrval(tmpInd,1)
                valpredictedLabels(1, i) = tmpInd;
            else
                valpredictedLabels(1, i) = 4;
            end
        end    
    otherwise 
        [~,valpredictedLabels] = pdist2(C, valFeature, dist,'Smallest',1);
end

if make_plots
    figure(2)
    scatter3(valFeature(valpredictedLabels==1,1),...
        valFeature(valpredictedLabels == 1,2),valFeature(valpredictedLabels == 1,3),'r','o');
    hold on;
    scatter3(valFeature(valpredictedLabels==2,1),...
        valFeature(valpredictedLabels == 2,2),valFeature(valpredictedLabels == 2,3),'g','o');
    scatter3(valFeature(valpredictedLabels==3,1),...
        valFeature(valpredictedLabels == 3,2),valFeature(valpredictedLabels == 3,3),'b','o');
    scatter3(valFeature(valpredictedLabels==4,1),...
        valFeature(valpredictedLabels == 4,2),valFeature(valpredictedLabels == 4,3),'k','o');
    plot3(C(:,1), C(:,2), C(:,3), 'kx', 'MarkerSize',15,'LineWidth',3) ;
    legend('Cluster 1','Cluster 2','Cluster 3','outlier','Cluster Centroid')
    hold off;
end

valEval = confusionmat(valtrueLabels,valpredictedLabels);
truePositiveP= max(valEval,[],1);
[truePositiveR, Reorder] = max(valEval,[],2);
valPrecision = truePositiveP./sum(valEval,1);
valPrecision = [valPrecision(1,Reorder(1,1)),valPrecision(1,Reorder(2,1)),valPrecision(1,Reorder(3,1))];
valRecall = truePositiveR./sum(valEval,2);
valRecall = valRecall.';
valRecall = valRecall(:,1:3);

valF1 = 2.*valPrecision.*valRecall./(valPrecision + valRecall);
valPrecisionm = mean(valPrecision);
valRecallm = mean(valRecall);
valF1m = mean(valF1);
results = [valPrecisionm valRecallm valF1m];

    figure(3);
    cm = confusionchart(valtrueLabels,valpredictedLabels);
    cm.ColumnSummary = 'column-normalized';
    cm.RowSummary = 'row-normalized';