clear; close;

load D:\study\dataset\cifar10\cifar-10-batches-mat\cifar_10_gist.mat;

traindata = double(traindata);
testdata = double(testdata);
cateTrainTest = bsxfun(@eq, traingnd, testgnd');

if sum(traingnd == 0)
    traingnd = traingnd + 1;
    testgnd = testgnd + 1;
end


Ntrain = size(traindata,1);
% Use all the training data
X = traindata;
label = double(traingnd);

% get anchors
%n_anchors = 1000;
%rand('seed',1);
%anchor = X(randsample(Ntrain, n_anchors),:);


% % determin rbf width sigma
% Dis = EuDist2(X,anchor,0);
% % sigma = mean(mean(Dis)).^0.5;
% sigma = mean(min(Dis,[],2).^0.5);
% clear Dis

% sigma = 0.4; % for normalized data
% PhiX = exp(-sqdist(X,anchor)/(2*sigma*sigma));
% PhiX = [PhiX, ones(Ntrain,1)];
 
% Phi_testdata = exp(-sqdist(testdata,anchor)/(2*sigma*sigma)); clear testdata
% Phi_testdata = [Phi_testdata, ones(size(Phi_testdata,1),1)];
% Phi_traindata = exp(-sqdist(traindata,anchor)/(2*sigma*sigma)); clear traindata;
 %Phi_traindata = [Phi_traindata, ones(size(Phi_traindata,1),1)];


%zero-centered
%X = normalize(X);
%traindata = normalize(traindata);
%testdata = normalize(testdata);
%X = bsxfun(@minus, X, mean(X));
%traindata = bsxfun(@minus, traindata, mean(traindata));
%testdata = bsxfun(@minus, testdata, mean(testdata));


maxItr = 3;

%% run algo
nbits = 32
sparsity = 0.5

debug = 0;
tic
[R, R2] = SSP(X, label, nbits, sparsity, [], maxItr, debug);
toc
fprintf('non-zero elements %.2f%%\n', 100*nnz(R)/numel(R));

%% evaluation
display('Evaluation...');

R = R';
H = traindata * R > 0;

tic;
tH = testdata * R > 0;
toc

hammRadius = 2;

B = compactbit(H);
tB = compactbit(tH);

hammTrainTest = hammingDist(tB, B)';
% hash lookup: precision and reall
Ret = (hammTrainTest <= hammRadius+0.00001);
% hamming ranking: MAP
[~, HammingRank]=sort(hammTrainTest,1);
MAP = cat_apcal(traingnd,testgnd,HammingRank)

[Pre_K, Rec_K] = cat_ap_topK(cateTrainTest, HammingRank, 500)

