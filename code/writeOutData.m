function writeOutData(outDir)
%function that takes processed aedat with Jt and generates files for training

%where are the .mat files
files =  dir([outDir '*_epm.mat']);

for fLoop = 1:numel(files)

    clear aedat Xexp Xhist Y
    
    [~,fn,~] = fileparts(files(fLoop).name);
    if exist([outDir fn '_labels.mat'], 'file')
        disp('file already processed')
        pause(1)
        continue
    end
    
    load([dataDir files(fLoop).name], 'aedat');
    load([dataDir files(fLoop).name], 'inputVar');

    
    %delete unneeded data from aedat
    aedat2.data.frame.size = aedat.data.frame.size;
    aedat2.data.polarity.numEvents = aedat.data.polarity.numEvents;
    aedat2.data.polarity.x = aedat.data.polarity.x;
    aedat2.data.polarity.y = aedat.data.polarity.y;
    aedat2.data.polarity.timeStamp = aedat.data.polarity.timeStamp;
    aedat2.data.polarity.polarity = aedat.data.polarity.polarity;
%     aedat2.data.polarity.duringAPS =uint8(aedat.data.polarity.duringAPS>0);
%     aedat2.data.polarity.apsIntGood = aedat.data.polarity.apsIntGood;
%     aedat2.data.polarity.Prob = aedat.data.polarity.Prob;
%     aedat2.data.polarity.Jt = aedat.data.polarity.Jt; %added 2/25/20
    aedat2.importParams.filePath = aedat.importParams.filePath;
    aedat2.data.frame.apsIntGood = aedat.data.frame.apsIntGood;
    aedat2.data.frame.diffImTime = aedat.data.frame.diffImTime;
    aedat2.data.frame.Ix = aedat.data.frame.Gx;
    aedat2.data.frame.Iy = aedat.data.frame.Gy;
    aedat2.data.frame.Jx = aedat.data.frame.Jx;
    aedat2.data.frame.Jy = aedat.data.frame.Jy;
    aedat2.data.frame.Vx = aedat.data.frame.Vx;
    aedat2.data.frame.Vy = aedat.data.frame.Vy;
    aedat2.data.frame.Jt = aedat.data.frame.Jt;
    aedat2.data.frame.I = cell2mat(reshape(aedat.data.frame.samples,1,1,[])).*repmat(inputVar.fpn.slope,1,1,aedat.data.frame.numDiffImages);

    aedat = aedat2;
    clear aedat2
    
%     inputVar.maxNumSamples = 20e3; %Only sample up to this many events each for pos and neg polarities combined
%     inputVar.nonCausal = false; %
%     inputVar.neighborhood = 6; %added 2/25/20

    %added 3/12/2020
    inputVar.maxNumSamples = 20; %Only sample up to this many chips per frame
    inputVar.nonCausal = false; %
    inputVar.neighborhood = 12; %
    
    %     tW = 0:100:max(abs(aedat.data.polarity.frameTimeDelta));
    %     for loop2=1:numel(tW)
    %         posAcc(loop2) = mean(aedat.data.polarity.Prob(aedat.data.polarity.polarity==1 & (abs(aedat.data.polarity.frameTimeDelta)<tW(loop2)))>0);
    %         negAcc(loop2) = mean(aedat.data.polarity.Prob(aedat.data.polarity.polarity==0 & (abs(aedat.data.polarity.frameTimeDelta)<tW(loop2)))<0);
    %     end
    %     clf
    %     hold on
    %     plot(tW,posAcc,'b')
    %     plot(tW,negAcc,'r')
    
    % makeAnimatedGif(aedat)
    
    % Process event points into feature vectors
    % [X, Y] = events2Feat(aedat, inputVar);
    %     [X, Y, samples] = events2FeatAll(aedat, inputVar);
%     [X, Y, samples, pol] = events2FeatML(aedat, inputVar); %edited 13FEB2020

%     [X, Y, samples, pol] = events2FeatExp(aedat, inputVar);
    tic;
%     [X, Y, samples] = events2FeatExpSurfChipFast(aedat, inputVar);%added 3/12/2020
    [Xexp, Xhist, Y, samples] = events2Feat2SurfChipFast(aedat, inputVar);%added 4/1/2020
    toc
    
%     save([outDir fn '_labels.mat'],'X','Y','samples','-v7.3')
    save([outDir fn '_labels.mat'],'Xexp','Xhist','Y','samples','-v7.3')
    
end

%Combine all data into one mat file
files = dir([outDir '*_labels.mat'])
fileName = {files(:).name};

%% test data
testIdx = [1:3 13:15 28:30];
numSamples = 0
for loop = testIdx
    loop
    [outDir files(loop).name]
    load([outDir files(loop).name], 'samples')
    numSamples = numSamples +numel(samples.f);
end

%preallocate
% numSamples = 187570;
% XTestExp = nan(25,25,32,numSamples);
% XTestHist = nan(25,25,32,numSamples);
YTest = nan(25,25,8,numSamples);
samples_test.r = nan(numSamples,1);
samples_test.c = nan(numSamples,1);
samples_test.f = nan(numSamples,1);
samples_test.fileID = nan(numSamples,1);
cnt1 = 1;
cnt2 = 1;

% cntr = 0;
for loop = testIdx
    clc, loop
    load([outDir files(loop).name],'samples','Y')

    samples.fileID = zeros(size(samples.r)) + loop;
    
    cnt2 = cnt1 + numel(samples.r) - 1;
    
%     XTestExp(:,:,:,cnt1:cnt2) = Xexp;
%     XTestHist(:,:,:,cnt1:cnt2) = Xhist;
    YTest(:,:,:,cnt1:cnt2) = Y;
    samples_test.r(cnt1:cnt2) = samples.r;
    samples_test.c(cnt1:cnt2) = samples.c;
    samples_test.f(cnt1:cnt2) = samples.f;
    samples_test.fileID(cnt1:cnt2) = samples.fileID;
    
    cnt1 = cnt2 + 1;
end

clear Y samples
save([outDir 'all_labels_test.mat'],'YTest','samples_test','fileName','-v7.3')
clear YTest samples_test

%preallocate
% numSamples = 187570;
XTestExp = nan(25,25,32,numSamples);
% XTestHist = nan(25,25,32,numSamples);
% YTest = nan(25,25,8,numSamples);
% samples_test.r = nan(numSamples,1);
% samples_test.c = nan(numSamples,1);
% samples_test.f = nan(numSamples,1);
% samples_test.fileID = nan(numSamples,1);
cnt1 = 1;
cnt2 = 1;

% cntr = 0;
for loop = testIdx
    clc, loop
    load([outDir files(loop).name],'samples','Xexp')

%     samples.fileID = zeros(size(samples.r)) + loop;
    
    cnt2 = cnt1 + numel(samples.r) - 1;
    
    XTestExp(:,:,:,cnt1:cnt2) = Xexp;
%     XTestHist(:,:,:,cnt1:cnt2) = Xhist;
%     YTest(:,:,:,cnt1:cnt2) = Y;
%     samples_test.r(cnt1:cnt2) = samples.r;
%     samples_test.c(cnt1:cnt2) = samples.c;
%     samples_test.f(cnt1:cnt2) = samples.f;
%     samples_test.fileID(cnt1:cnt2) = samples.fileID;
    
    cnt1 = cnt2 + 1;
end

clear Xexp
save([outDir 'all_labels_test.mat'],'XTestExp','-append')
clear XTestExp

%preallocate
% numSamples = 187570;
% XTestExp = nan(25,25,32,numSamples);
XTestHist = nan(25,25,32,numSamples);
% YTest = nan(25,25,8,numSamples);
% samples_test.r = nan(numSamples,1);
% samples_test.c = nan(numSamples,1);
% samples_test.f = nan(numSamples,1);
% samples_test.fileID = nan(numSamples,1);
cnt1 = 1;
cnt2 = 1;

% cntr = 0;
for loop = testIdx
    clc, loop
    load([outDir files(loop).name],'samples','Xhist')

%     samples.fileID = zeros(size(samples.r)) + loop;
    
    cnt2 = cnt1 + numel(samples.r) - 1;
    
%     XTestExp(:,:,:,cnt1:cnt2) = Xexp;
    XTestHist(:,:,:,cnt1:cnt2) = Xhist;
%     YTest(:,:,:,cnt1:cnt2) = Y;
%     samples_test.r(cnt1:cnt2) = samples.r;
%     samples_test.c(cnt1:cnt2) = samples.c;
%     samples_test.f(cnt1:cnt2) = samples.f;
%     samples_test.fileID(cnt1:cnt2) = samples.fileID;
    
    cnt1 = cnt2 + 1;
end

clear Xhist
save([outDir 'all_labels_test.mat'],'XTestHist','-append')
clear XTestHist


%% train data
trainIdx = [4:12 16:27 31:48];
numSamples = 0
for loop = trainIdx
    loop
    [outDir files(loop).name]
    load([outDir files(loop).name], 'samples')
    numSamples = numSamples +numel(samples.f);
end

%preallocate
% numSamples = 187570;
% XTestExp = nan(25,25,32,numSamples);
% XTestHist = nan(25,25,32,numSamples);
YTrain = nan(25,25,8,numSamples);
samples_train.r = nan(numSamples,1);
samples_train.c = nan(numSamples,1);
samples_train.f = nan(numSamples,1);
samples_train.fileID = nan(numSamples,1);
cnt1 = 1;
cnt2 = 1;

% cntr = 0;
for loop = trainIdx
    clc, loop
    load([outDir files(loop).name],'samples','Y')

    samples.fileID = zeros(size(samples.r)) + loop;
    
    cnt2 = cnt1 + numel(samples.r) - 1;
    
%     XTestExp(:,:,:,cnt1:cnt2) = Xexp;
%     XTestHist(:,:,:,cnt1:cnt2) = Xhist;
    YTrain(:,:,:,cnt1:cnt2) = Y;
    samples_train.r(cnt1:cnt2) = samples.r;
    samples_train.c(cnt1:cnt2) = samples.c;
    samples_train.f(cnt1:cnt2) = samples.f;
    samples_train.fileID(cnt1:cnt2) = samples.fileID;
    
    cnt1 = cnt2 + 1;
end

clear Y samples
save([outDir 'all_labels_train.mat'],'YTrain','samples_train','fileName','-v7.3')
clear YTrain samples_train

%preallocate
% numSamples = 187570;
XTrainExp = nan(25,25,32,numSamples);
% XTestHist = nan(25,25,32,numSamples);
% YTest = nan(25,25,8,numSamples);
% samples_test.r = nan(numSamples,1);
% samples_test.c = nan(numSamples,1);
% samples_test.f = nan(numSamples,1);
% samples_test.fileID = nan(numSamples,1);
cnt1 = 1;
cnt2 = 1;

% cntr = 0;
for loop = trainIdx
    clc, loop
    load([outDir files(loop).name],'samples','Xexp')

%     samples.fileID = zeros(size(samples.r)) + loop;
    
    cnt2 = cnt1 + numel(samples.r) - 1;
    
    XTrainExp(:,:,:,cnt1:cnt2) = Xexp;
%     XTestHist(:,:,:,cnt1:cnt2) = Xhist;
%     YTest(:,:,:,cnt1:cnt2) = Y;
%     samples_test.r(cnt1:cnt2) = samples.r;
%     samples_test.c(cnt1:cnt2) = samples.c;
%     samples_test.f(cnt1:cnt2) = samples.f;
%     samples_test.fileID(cnt1:cnt2) = samples.fileID;
    
    cnt1 = cnt2 + 1;
end

clear Xexp
save([outDir 'all_labels_train.mat'],'XTrainExp','-append')
clear XTrainExp

%preallocate
% numSamples = 187570;
% XTestExp = nan(25,25,32,numSamples);
XTrainHist = nan(25,25,32,numSamples);
% YTest = nan(25,25,8,numSamples);
% samples_test.r = nan(numSamples,1);
% samples_test.c = nan(numSamples,1);
% samples_test.f = nan(numSamples,1);
% samples_test.fileID = nan(numSamples,1);
cnt1 = 1;
cnt2 = 1;

% cntr = 0;
for loop = trainIdx
    clc, loop
    load([outDir files(loop).name],'samples','Xhist')

%     samples.fileID = zeros(size(samples.r)) + loop;
    
    cnt2 = cnt1 + numel(samples.r) - 1;
    
%     XTestExp(:,:,:,cnt1:cnt2) = Xexp;
    XTrainHist(:,:,:,cnt1:cnt2) = Xhist;
%     YTest(:,:,:,cnt1:cnt2) = Y;
%     samples_test.r(cnt1:cnt2) = samples.r;
%     samples_test.c(cnt1:cnt2) = samples.c;
%     samples_test.f(cnt1:cnt2) = samples.f;
%     samples_test.fileID(cnt1:cnt2) = samples.fileID;
    
    cnt1 = cnt2 + 1;
end

clear Xhist
save([outDir 'all_labels_train.mat'],'XTrainHist','-append')
clear XTrainHist

% %% train data
% trainIdx = [4:12 16:27 31:48];
% numSamples = 0
% for loop = trainIdx
%     clc, loop
%     load([outDir files(loop).name], 'samples')
%     numSamples = numSamples +numel(samples.f);
% end
% 
% %preallocate
% % numSamples = 187570;
% XTrainExp = nan(25,25,32,numSamples);
% XTrainHist = nan(25,25,32,numSamples);
% YTrain = nan(25,25,8,numSamples);
% samples_train.r = nan(numSamples,1);
% samples_train.c = nan(numSamples,1);
% samples_train.f = nan(numSamples,1);
% samples_train.fileID = nan(numSamples,1);
% 
% cnt1 = 1;
% cnt2 = 1;
% 
% % cntr = 0;
% for loop = 1:numel(files)
%     
%     clc, loop
%     
%     load([outDir files(loop).name])
%     
% %     cntr = cntr + numel(samples.r);
% 
%     samples.fileID = zeros(size(samples.r)) + loop;
%     
%     cnt2 = cnt1 + numel(samples.r) - 1;
%     
%     XTrainExp(:,:,:,cnt1:cnt2) = Xexp;
%     XTrainHist(:,:,:,cnt1:cnt2) = Xhist;
%     YTrain(:,:,:,cnt1:cnt2) = Y;
%     samples_train.r(cnt1:cnt2) = samples.r;
%     samples_train.c(cnt1:cnt2) = samples.c;
%     samples_train.f(cnt1:cnt2) = samples.f;
%     samples_train.fileID(cnt1:cnt2) = samples.fileID;
%     
%     cnt1 = cnt2 + 1;
% 
% end
% 
% clear Xexp Xhist Y samples
% 
% save([outDir 'all_labels_train.mat'],'XTrainExp','XTrainHist','YTrain','samples_train','fileName','-v7.3')


% % % numSamples = 0
% % % for loop = 1:numel(files)
% % %     
% % %     clc, loop
% % %     
% % %     load([outDir files(loop).name], 'samples')
% % %     
% % %     numSamples = numSamples +numel(samples.f);
% % % 
% % % end
% % % 
% % % %preallocate
% % % % numSamples = 187570;
% % % X_all = nan(25,25,32,numSamples);
% % % Y_all = nan(25,25,8,numSamples);
% % % samples_all.r = nan(numSamples,1);
% % % samples_all.c = nan(numSamples,1);
% % % samples_all.f = nan(numSamples,1);
% % % samples_all.fileID = nan(numSamples,1);
% % % 
% % % cnt1 = 1;
% % % cnt2 = 1;
% % % 
% % % % cntr = 0;
% % % for loop = 1:numel(files)
% % %     
% % %     clc, loop
% % %     
% % %     load([outDir files(loop).name])
% % %     
% % % %     cntr = cntr + numel(samples.r);
% % % 
% % %     samples.fileID = zeros(size(samples.r)) + loop;
% % %     
% % %     cnt2 = cnt1 + numel(samples.r) - 1;
% % %     
% % %     X_all(:,:,:,cnt1:cnt2) = X;
% % %     Y_all(:,:,:,cnt1:cnt2) = Y;
% % %     samples_all.r(cnt1:cnt2) = samples.r;
% % %     samples_all.c(cnt1:cnt2) = samples.c;
% % %     samples_all.f(cnt1:cnt2) = samples.f;
% % %     samples_all.fileID(cnt1:cnt2) = samples.fileID;
% % %     
% % %     cnt1 = cnt2 + 1;
% % % 
% % % end
% % % 
% % % clear X Y samples
% % % 
% % % save([outDir 'all_labels.mat'],'X','Y','samples','fileName','-v7.3')

    % %% Visualize a feature
    % sampleIdx = randperm(size(X,4),1);
    % sList = find(samples);
    % pol = aedat.data.polarity.polarity(sList(sampleIdx));
    % wes = X(:,:,:,sampleIdx);
    % [x,y] = meshgrid(1:size(X,1),1:size(X,2));
    % x = repmat(x,1,1,size(X,3));
    % y = repmat(y,1,1,size(X,3));
    % c=repmat(reshape([1:size(X,3)],1,1,size(X,3)),size(X,1),size(X,2),1);
    % scatter3(x(:),y(:),wes(:),15,c(:),'filled')
    % title([num2str(Y(sampleIdx)) ' - ' num2str(pol)])
    %
    %
    % %% Train CNN to learn probabilities from feature vectors
    % [net, trainMask] = trainDenoiseNetwork(X, Y, inputVar);
    %
    % makeAnimatedGifResults(aedat, samples, trainMask)
    
    %% Write out images for use with pytorch
    
%     [~,fn,~] = fileparts(aedat.importParams.filePath);
%     folderDir = [outDir fn filesep];
%     
%     if ~exist(folderDir,'dir')
%         mkdir(folderDir)
%     end
%     
%     %TIFF tag setup
%     tagstruct.ImageLength = size(X,1);
%     tagstruct.ImageWidth = size(X,2);
%     tagstruct.Photometric = Tiff.Photometric.MinIsBlack;
%     tagstruct.BitsPerSample = 32;
%     tagstruct.SamplesPerPixel = size(X,3);
%     tagstruct.SampleFormat = Tiff.SampleFormat.IEEEFP;
%     tagstruct.PlanarConfiguration = Tiff.PlanarConfiguration.Chunky;
%     tagstruct.Software = 'MATLAB';
%     
%     %Write train data
%     sList = find(samples);
%     dList = 1:numel(sList);
%     file_name = cell(numel(sList),1);
%     est_probability = zeros(numel(sList),1);
%     polarity = zeros(numel(sList),1);
%     for loop = 1:numel(sList)
%         clc, loop/numel(sList)
%         file_name{loop} = [fn filesep fn '_' num2str(loop) '.tif'];
%         est_probability(loop) = Y(dList(loop));
%         polarity(loop) = aedat.data.polarity.polarity(sList(loop));
%         t = Tiff([outDir file_name{loop}],'w');
%         setTag(t,tagstruct)
%         write(t,X(:,:,:,dList(loop)));
%         close(t);
%     end
%     T{fLoop} = table(file_name, est_probability, polarity);
%     writetable(T{fLoop}, [outDir fn '_gt.csv'])
%     
%     save([outDir fn '_labels.mat'],'-v7.3')
% 
% end

% 
% %% Split the data into train/test randomly - put images from the same collect in the same folder train or test
% trainPercent = 90;
% %How many samples to randomly grab from test dataset for validation
% numValidSamples = 2e5;
% 
% %Histogram of all values
% % allTable = T{1};
% % for loop = 1:numel(files)
% %     allTable = cat(1,allTable,T{loop});
% % end
% 
% rng('shuffle')
% rIdx = randperm(numel(files));
% 
% trainIdx = rIdx <= (floor(numel(files)*trainPercent/100))
% 
% mkdir([outDir 'train'])
% mkdir([outDir 'test'])
% 
% %Move data into train/test subfolder
% for fLoop = 1:numel(files)
%     [~,fn,~] = fileparts(files(fLoop).name);
%     folderDir = [outDir fn];
%     folderFile = [outDir fn '_gt.csv'];
%     if trainIdx(fLoop)
%         moveDir = [outDir 'train' filesep fn]
%         moveFile = [outDir 'train' filesep fn '_gt.csv']
%     else
%         moveDir = [outDir 'test' filesep fn]
%         moveFile = [outDir 'test' filesep fn '_gt.csv']
%     end
% %     unix(['mv ' folderDir ' ' moveDir])
%     movefile(folderDir,moveDir)
%     movefile(folderFile,moveFile)
%     
% end
% 
% %Copy some test data into validation
% 
% %Build the labels
% 
% testTruth = dir([outDir filesep 'test' filesep '*_gt.csv']);
% for loop = 1:numel(testTruth)
%     if loop == 1
%         testTable = readtable([outDir filesep 'test' filesep testTruth(loop).name],'Delimiter',',');
%     else
%         testTable = cat(1,testTable,readtable([outDir filesep 'test' filesep testTruth(loop).name],'Delimiter',','));
%     end
% end
% writetable(testTable, [outDir 'gt_test.csv'])
% 
% 
% trainTruth = dir([outDir filesep 'train' filesep '*_gt.csv']);
% for loop = 1:numel(trainTruth)
%     if loop == 1
%         trainTable = readtable([outDir filesep 'train' filesep trainTruth(loop).name],'Delimiter',',');
%     else
%         trainTable = cat(1,trainTable,readtable([outDir filesep 'train' filesep trainTruth(loop).name],'Delimiter',','));
%     end
% end
% writetable(trainTable, [outDir 'gt_train.csv'])
% 
% 
% % idx = find(trainIdx);
% % trainTable = T{idx(1)};
% % for loop = 2:numel(idx)
% %     trainTable = cat(1,trainTable,T{idx(loop)});
% % end
% % writetable(trainTable, [outDir 'gt_train.csv'])
% % 
% % idx = find(~trainIdx);
% % testTable = T{idx(1)};
% % for loop = 2:numel(idx)
% %     testTable = cat(1,testTable,T{idx(loop)});
% % end
% % writetable(testTable, [outDir 'gt_test.csv'])
% 
% %For valid dataset use a max of N samples
% numValidSamples = min(size(testTable,1),numValidSamples)
% validIdx = randperm(size(testTable,1),numValidSamples);
% writetable(testTable(validIdx,:), [outDir 'gt_valid.csv'])
% unix(['ln -s ' outDir 'test ' outDir 'valid']) %no need to reproduce all the files for validation, just make a symlink
% 
% %For training iteration dataset use a max of N samples
% smallTrainIdx = randperm(size(trainTable,1),numValidSamples);
% writetable(trainTable(smallTrainIdx,:), [outDir 'gt_smalTrain.csv'])
% unix(['ln -s ' outDir 'train ' outDir 'smallTrain']) %no need to reproduce all the files for small train, just make a symlink

