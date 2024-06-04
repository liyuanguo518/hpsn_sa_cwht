function get_spikes(raw_filename)

% PROGRAM Get_spikes.
% Detect spikes and save them in a file.
% Saves spikes, spike index in filename_spikes.mat.

load (['../datasets/' raw_filename ], 'data');
load (['../datasets/' raw_filename ], 'spike_times');
load (['../datasets/' raw_filename ], 'spike_class');

spike_t = spike_times{1,1};
spikes = [];
classInd = [];

for i = 1:length(spike_t)
    p = data(spike_t(i) : spike_t(i) + 63);
    spikes = [spikes; p];
    classInd = [classInd; spike_class{1,1}(1,i)];
end

[unused, fnam, ext] = fileparts(raw_filename);
save(['./data_tmp/' fnam '_spikes'], 'spikes','classInd')
