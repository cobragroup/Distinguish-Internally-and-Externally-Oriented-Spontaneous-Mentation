clear ; clc ; close all

rng('default')

addpath('classifiers')

% Select modality
modality = 'fmri';

switch modality
    case 'fmri'
        data = load('data/fmri_feat.mat');
    case 'eeg'
        data = load('data/eeg_feat.mat');
end

x = data.x;
y = data.y;
sub_id = data.sub_id;

% Maximum window size and position before the beep
max_wp = size(x, 4);
max_ws = size(x, 3);

%% I/E classification
acc = zeros(max_ws, max_wp);
acc_bal = zeros(max_ws, max_wp);

cv_params = {'Leaveout', 'on'};

for s = 1:max_ws

    for p = 1:max_wp

        fprintf('Size %i Position %i\n', s, p)

        xx = x(:,:,s,p);

        [~,perf_measures] = svm_in_ex([xx,y], cv_params);

        % Extract accuracy and balanced accuracy of I/E classification
        acc(s,p) = perf_measures.acc;
        acc_bal(s,p) = perf_measures.acc_bal;

    end

end

%% I/E classification - permuted labels
n_perms = 5000;
valid_subs_od_smpls = sub_id;
sub_id_unique = unique(valid_subs_od_smpls);
sub_id_unique_n = sum(valid_subs_od_smpls == sub_id_unique',1);
y_cell = mat2cell(y, sub_id_unique_n, 1);

% Adjust parallel pool size
pool_size = 50;
if isempty(gcp('nocreate')) ; parpool(pool_size) ; end

parfor np = 1:n_perms

    filename = ['num_perm', num2str(np), '.txt'];
    perm_res_dir = fullfile('results', filename);
    fileid = fopen(perm_res_dir, 'w');
    fprintf(fileid, 'Perm %i \n', np);

    fprintf('Perm %i \n', np)

    y_cell_perm = cellfun(@(x) x(randperm(length(x))), y_cell, 'UniformOutput', false);
    y_perm{np} = cell2mat(y_cell_perm);
    yy = y_perm{np};

    for s = 1:max_ws

        for p = 1:max_wp

            xx = x(:,:,s,p);

            [~,perf_measures_perm] = svm_in_ex([xx,yy], cv_params);

            % Extract accuracy of I/E classification on the permuted labels
            acc_perm{np}(s,p) = perf_measures_perm.acc;

        end

    end

end

acc_perm = reshape(cell2mat(acc_perm), [max_ws,max_wp,n_perms]);

% Save results
res_matfile = fullfile('results', [modality,'_results.mat']);
save(res_matfile, 'acc', 'acc_bal', 'acc_perm')