function Ex_CORAL(src, tgt)
% Source code adapted from GFK and SA.

addpath('../libsvm-3.20/matlab');

%--------------------I. prepare data--------------------------------------
load(['../data/' src '_SURF_L10.mat']);     % source domain
fts = fts ./ repmat(sum(fts,2),1,size(fts,2));
Source = zscore(fts,1);    clear fts
Source_lbl = labels;           clear labels

load(['../data/' tgt '_SURF_L10.mat']);     % target domain
fts = fts ./ repmat(sum(fts,2),1,size(fts,2));
Target = zscore(fts,1);     clear fts
Target_lbl = labels;            clear labels

Source = double(Source);
Target = double(Target);

fprintf('\nsource (%s) --> target (%s):\n', src, tgt);
fprintf('round     accuracy\n');
%--------------------II. run experiments----------------------------------
round = 20; % 

nPerClass = 20;
Xtt = Target;
Ytt = Target_lbl;

for iter = 1 : round
    fprintf('%4d', iter);
    
    inds = split(Source_lbl, nPerClass);
    
    Xr = Source(inds,:);
    Yr = Source_lbl(inds);
    
    %CORAL
    cov_source = cov(Xr) + eye(size(Xr, 2));
    cov_target = cov(Xtt) + eye(size(Xtt, 2));
    A_coral = cov_source^(-1/2)*cov_target^(1/2);
    Sim_coral = Xr * A_coral * Xtt';
    accy_coral(iter) = SVM_Accuracy(Xr, A_coral, Ytt, Sim_coral, Yr);
    
    %NA
    accy_na(iter) = LinAccuracy(Xr,Xtt,Yr,Ytt);
    
end

mean_accuracy_na = mean(accy_na);
mean_accuracy_coral = mean(accy_coral);

std_na = std(accy_na);
std_coral = std(accy_coral);

save(['Src_' src '_Tgt_' tgt '_subsample_20_runs.mat'], 'accy_na', 'accy_coral', ...
    'mean_accuracy_na','mean_accuracy_coral','std_na', 'std_coral','-v7.3');
end

function res = SVM_Accuracy (trainset, M,testlabelsref,Sim,trainlabels)
Sim_Trn = trainset * M *  trainset';
index = [1:1:size(Sim,1)]';
Sim = [[1:1:size(Sim,2)]' Sim'];
Sim_Trn = [index Sim_Trn ];

C = [0.001 0.01 0.1 1.0 10 100 1000 10000];
parfor i = 1 :size(C,2)
    model(i) = svmtrain(trainlabels, Sim_Trn, sprintf('-t 4 -c %d -v 2 -q',C(i)));
end
[val indx]=max(model);
CVal = C(indx);
model = svmtrain(trainlabels, Sim_Trn, sprintf('-t 4 -c %d -q',CVal));
[predicted_label, accuracy, decision_values] = svmpredict(testlabelsref, Sim, model);
res = accuracy(1,1);
end


function acc = LinAccuracy(trainset,testset,trainlbl,testlbl)
model = trainSVM_Model(trainset,trainlbl);
[predicted_label, accuracy, decision_values] = svmpredict(testlbl, testset, model);
acc = accuracy(1,1);
end

function svmmodel = trainSVM_Model(trainset,trainlbl)
C = [0.001 0.01 0.1 1.0 10 100 ];
parfor i = 1 :size(C,2)
    model(i) = svmtrain(double(trainlbl), sparse(double((trainset))),sprintf('-c %d -q -v 2',C(i) ));
end
[val indx]=max(model);
CVal = C(indx);
svmmodel = svmtrain(double(trainlbl), sparse(double((trainset))),sprintf('-c %d -q',CVal));
end

function [idx1 idx2] = split(Y,nPerClass, ratio)
% [idx1 idx2] = split(X,Y,nPerClass)
idx1 = [];  idx2 = [];
for C = 1 : max(Y)
    idx = find(Y == C);
    rn = randperm(length(idx));
    if exist('ratio')
        nPerClass = floor(length(idx)*ratio);
    end
    idx1 = [idx1; idx( rn(1:min(nPerClass,length(idx))) ) ];
    idx2 = [idx2; idx( rn(min(nPerClass,length(idx))+1:end) ) ];
end
end
