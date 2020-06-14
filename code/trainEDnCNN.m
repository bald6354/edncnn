function results = trainEDnCNN(outDir)

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
    
    %Feature dimensions
    imageSize = [size(XTrain,1) size(XTrain,2) size(XTrain,3)];
    
    %Binary Classification
    auimds = augmentedImageDatastore(imageSize,XTrain,categorical(YTrain),'DataAugmentation',augmenter);
    
    %Define architecture
    layers = [
        imageInputLayer(imageSize, 'Normalization', 'zscore') %added normalization on 2/18/2020)
        
        convolution2dLayer(3,16,'Padding','same')
        batchNormalizationLayer
        reluLayer
        
        maxPooling2dLayer(2,'Stride',2)
        
        convolution2dLayer(3,32,'Padding','same')
        batchNormalizationLayer
        reluLayer
        
        maxPooling2dLayer(2,'Stride',2)
        
        convolution2dLayer(3,64,'Padding','same')
        batchNormalizationLayer
        reluLayer
        
        dropoutLayer(0.5)
        
        fullyConnectedLayer(256)
        fullyConnectedLayer(2)
        softmaxLayer
        classificationLayer];
    
    
    %% Train Network
    miniBatchSize  = 2048;
    validationFrequency = floor(numel(YTrain)/miniBatchSize/2);
    options = trainingOptions('adam', ...
        'MiniBatchSize',miniBatchSize, ...
        'MaxEpochs',5, ...
        'InitialLearnRate',2e-4, ...
        'LearnRateSchedule','piecewise', ...
        'LearnRateDropFactor',0.1, ...
        'LearnRateDropPeriod',3, ...
        'ValidationData',{XTest,categorical(YTest)}, ...
        'ValidationFrequency',validationFrequency, ...
        'Shuffle','every-epoch', ...
        'Verbose',true);
    
    [net,info] = trainNetwork(auimds,layers,options);
    
    [results.bestAccuracy(grpID), results.bestAccuracyIdx(grpID)] = max(info.ValidationAccuracy);
    results.numEpochs(grpID) = sum(~isnan(info.ValidationAccuracy)) - 1;

    %% Test Network
    %Binary Classification
    YPredicted = classify(net,XTest);
    accuracy = mean(YPredicted == categorical(YTest))
   
    
    %% Save out trained network
    save([outDir num2str(grpID) '_trained_v1.mat'],'net','accuracy')
    
end
