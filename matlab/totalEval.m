clear all;

% PROGRAM totalEval.
% Dose clustering on all files in folder 'datasets' using both euclidean
% and mahalanobis distance metrics and evaluate the results.

% parameters needed for each simulation:
%     clustering: 'kmedoid' / 'kmean'
%     FE: 2-10 (dimensions used in feature extraction)
%     ncluster: number of clusters.(for convenience of this simulation, 
%               ncluster is always 3) 
training = 'kmean';
FE = 3;
ncluster = 3;

%use relative path to run code easily on every machine
currentFile = mfilename( 'fullpath' );
[pathstr, name, ~] = fileparts( currentFile );
cd(pathstr);
addpath( fullfile( pathstr ) );

dirnames = dir('../datasets');
dirnames = {dirnames.name};
filenames = {};

for i = 1:length(dirnames)
    fname = dirnames{i};
    [unused, f, ext] = fileparts(fname);
    if strcmp(ext,'.mat')
        if ismember('spikes',f)
            continue
        elseif ismember('Test',f)
            continue
        elseif ismember('times',f)
            continue
        elseif ismember('short',f)
            continue
        end
        filenames = [filenames {fname}];
    end
end

evalMahal = {};
for j = 1:length(filenames)
%     j = 18;
    dist = 'mahalanobis'; 
    evalTemp = Do_classifying(filenames{j}, dist, training, FE, ncluster);
    evalMahal = [evalMahal; evalTemp];
end
evalMahal = cell2mat(evalMahal);
evalMahalmean = mean(evalMahal);
evalMahalvar = var(evalMahal,0,1);
 
evalEuc = {};
 for j = 1:length(filenames)
    dist = 'euclidean'; 
    evalTemp = Do_classifying(filenames{j}, dist, training, FE, ncluster);
    evalEuc = [evalEuc; evalTemp];
end
evalEuc = cell2mat(evalEuc);
evalEucmean = mean(evalEuc);
evalEucvar = var(evalEuc,0,1);
%disp([filenames ': Using detected spikes'])
disp([evalMahalmean; evalMahalvar; evalEucmean; evalEucvar]);