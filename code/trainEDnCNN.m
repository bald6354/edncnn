function trainEDnCNN(outDir)

load([outDir 'all_labels.mat'],'X','Y','setLabel','grpLabel')

%Leave one out testing
for grpID = 1:max(grpLabel)
    
    %Split train test
    testSet = (grpID-1).*3 + 2;
    
    %Print out name
    files(testSet).name
    
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
    validationFrequency = floor(numel(YTrain)/miniBatchSize);
    options = trainingOptions('adam', ...
        'MiniBatchSize',miniBatchSize, ...
        'MaxEpochs',80, ...
        'InitialLearnRate',1e-4, ...
        'LearnRateSchedule','piecewise', ...
        'LearnRateDropFactor',0.1, ...
        'LearnRateDropPeriod',70, ...
        'ValidationData',{XTest,categorical(YTest)}, ...
        'ValidationFrequency',validationFrequency, ...
        'Shuffle','every-epoch', ...
        'Plots','training-progress', ...
        'Verbose',true);
    
    net = trainNetwork(auimds,net.Layers,options);
    
    
    %% Test Network
    %Binary Classification
    YPredicted = classify(net,XTest);
    %         accuracy = sum(YPredicted == categorical(YTest>0.5))/numel(YTest)
    accuracy = sum(YPredicted == categorical(YTest))/numel(YTest)
   
    
    %% Save out trained network
    save([num2str(grpID) '_exp_v1.mat'],'net','YPred')
    
end
