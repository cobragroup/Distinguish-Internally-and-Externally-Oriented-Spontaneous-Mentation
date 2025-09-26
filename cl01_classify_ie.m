clear ; clc ; close all

rng('default')

addpath('classifiers')
addpath('functions')

curr_date = char(datetime('now','Format','yyyyMMdd'));
curr_time = char(datetime('now','Format','yyyyMMdd_HHmmss'));

data = load('data/eeg_feat_20240710_151724.mat');
%data = load('data/fmri_feat_20250128_062205.mat');
feat = data.x2;
feat_len = size(feat,4);
max_ws = size(feat,3);

y = data.y;
sub_id = data.sub_id;

res_dir = fullfile('results',curr_date);
mkdir(res_dir)

cv_params = {'Leaveout', 'on'};

pool_size = 3;
parpool(pool_size)

%% I/E classification
acc = zeros(max_ws,feat_len);
cm = zeros(2,2,max_ws,feat_len);

for s = 1:max_ws

    parfor p = 1:feat_len

        fprintf('Size %i Position %i\n', s, p)

        xx = feat(:,:,s,p);

        [~,perf_measures] = svm_in_ex([xx,y],cv_params);

        acc(s,p) = perf_measures.acc;
        cm(:,:,s,p) = perf_measures.cm;

    end

end

%% I/E classification - permuted labels
n_perms = 5000;
valid_subs_od_smpls = sub_id;
sub_id_unique = unique(valid_subs_od_smpls);
sub_id_unique_n = sum(valid_subs_od_smpls == sub_id_unique',1);
y_cell = mat2cell(y,sub_id_unique_n,1);

parfor np = 1:n_perms

    filename = ['num_perm',num2str(np),'.txt'];
    perm_res_dir = fullfile(res_dir,filename);
    fileid = fopen(perm_res_dir,'w');
    fprintf(fileid,'Perm %i \n',np);

    fprintf('Perm %i \n',np)

    y_cell_perm = cellfun(@(x) x(randperm(length(x))),y_cell,'UniformOutput',false);
    y_perm{np} = cell2mat(y_cell_perm);
    yy = y_perm{np};

    for s = 1:max_ws

        for p = 1:feat_len

            xx = feat(:,:,s,p);

            [~,perf_measures_perm] = svm_in_ex([xx,yy],cv_params);

            acc_perm{np}(s,p) = perf_measures_perm.acc;

        end

    end

end

acc_perm = reshape(cell2mat(acc_perm),[max_ws,feat_len,n_perms]);

res_matfile = fullfile('results','classified_ie.mat');
save(res_matfile,'acc','cm','acc_perm')