function results = trainEDnCNN3D(outDir)

load([outDir 'all_labels.mat'],'X','Y','setLabel','grpLabel')

%Leave one out testing
for grpID = 1:max(grpLabel)
    
    %Split train test
    testSet = (grpID-1).*3 + 2;
    
    %Print out name
%     files(testSet).name
    
    %Put data from the test scene into test
    XTest = X(:,:,:,setLabel==testSet);
    YTest = Y(setLabel==testSet)>0.5; %Assign probability to a binary class
    
    %Put data from the other scenes into train
    XTrain = X(:,:,:,grpLabel~=grpID);
    YTrain = Y(grpLabel~=grpID)>0.5; %Assign probability to a binary class
    
    %Allow x/y reflection for data augmentation
    augmenter = imageDataAugmenter( ...
        'RandXReflection',true, ...
        'RandYReflection',true);
    
    imageSize = [size(XTrain,1) size(XTrain,2) size(XTrain,3)]
    layers = [
        image3dInputLayer(imageSize, 'Normalization', 'zscore', 'NormalizationDimension', 'all') %added normalization on 2/18/2020)
        convolution3dLayer([3 3 7], 8,'Padding','same')
        reluLayer
        convolution3dLayer([3 3 5], 16,'Padding','same')
        reluLayer
        convolution3dLayer([3 3 3], 32,'Padding','same')
        reluLayer
        reshapeLayer('rs',[imageSize(1) imageSize(2) imageSize(3) 32],[imageSize(1) imageSize(2) 1 imageSize(3)*32])
        convolution3dLayer([3 3 1], 64,'Padding','same')
        reluLayer
        fullyConnectedLayer(256)
        dropoutLayer(.4) %was.4
        fullyConnectedLayer(128)
        dropoutLayer(.4) %was.4
        fullyConnectedLayer(2)
        softmaxLayer
        classificationLayer];
    
    %reshape for 3d conv
    XTrain = reshape(XTrain,size(XTrain,1), size(XTrain,2), size(XTrain,3),1,[]);
    XTest = reshape(XTest,size(XTrain,1), size(XTrain,2), size(XTrain,3),1,[]);
    
    miniBatchSize  = 2^8;
    validationFrequency = floor(numel(YTrain)/miniBatchSize);
    options = trainingOptions('adam', ...
        'MiniBatchSize',miniBatchSize, ...
        'MaxEpochs',100, ...
        'InitialLearnRate',2e-4, ...
        'LearnRateSchedule','piecewise', ...
        'LearnRateDropFactor',0.1, ...
        'LearnRateDropPeriod',3, ...
        'ValidationFrequency',validationFrequency, ...
        'ValidationData',{XTest,categorical(YTest')}, ...
        'ValidationPatience',7, ...
        'Shuffle','every-epoch', ...
        'Plots','training-progress', ...
        'CheckpointPath','/media/wescomp/WesDataDrive/savedNetworks',...
        'Verbose',true);

    [net,info] = trainNetwork(XTrain, categorical(YTrain'), layers, options);
    
    [results.bestAccuracy(grpID), results.bestAccuracyIdx(grpID)] = max(info.ValidationAccuracy);
    results.numEpochs(grpID) = sum(~isnan(info.ValidationAccuracy)) - 1;
            
            
    %% Test Network
    %Binary Classification
    YPredicted = classify(net,XTest);
    accuracy = mean(YPredicted == categorical(YTest))
   
    
    %% Save out trained network
%     save([outDir num2str(grpID) '_trained_v1.mat'],'net','YPred')
    
end
