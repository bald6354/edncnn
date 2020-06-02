function [X, Y, samples] = events2FeatML_Fast(aedat, inputVar)

%calculate pdf for Jt - used to ensure a uniform sample from Jt later
% pd = fitdist(aedat.data.frame.Jt(:), 'normal');

aedat.data.polarity.x = double(aedat.data.polarity.x);
aedat.data.polarity.y = double(aedat.data.polarity.y);
aedat.data.polarity.timeStamp = double(aedat.data.polarity.timeStamp);

%Adjust times to minimum timestamp == 0 (all times in microseconds)
frameTimes = cellfun(@(x) mode(x(:)), aedat.data.frame.diffImTime);
firstTime = min([min(aedat.data.polarity.timeStamp) min(frameTimes)]);
frameTimes = frameTimes - firstTime;
aedat.data.polarity.timeStamp = aedat.data.polarity.timeStamp - firstTime;

decayWeights = repmat([1 2 4 8]',1,4).*[1e3 1e4 1e5 1e6]
decayWeights = reshape(decayWeights,1,1,[]);
inputVar.depth = numel(decayWeights);

%generate a weighted random sample the data to generate an even sample of probabilties
samples = false(aedat.data.polarity.numEvents,1);

numRows = double(aedat.data.frame.size(1));
numCols = double(aedat.data.frame.size(2));

% Filter to middle of recorded data
timeQuantiles = quantile(aedat.data.polarity.timeStamp,[0.4 0.6]);
qFilter = aedat.data.polarity.timeStamp >= timeQuantiles(1) & ...
    aedat.data.polarity.timeStamp <= timeQuantiles(2);

%Filter to DVS data during APS
duringAPS = (aedat.data.polarity.duringAPS > 0);

%do not to sample near an edge
nearEdgeIdx = ((aedat.data.polarity.y-inputVar.neighborhood) < 1) | ...
    ((aedat.data.polarity.x-inputVar.neighborhood) < 1) | ...
    ((aedat.data.polarity.y+inputVar.neighborhood) > numRows) | ...
    ((aedat.data.polarity.x+inputVar.neighborhood) > numCols);

%Do not sample events calculated from extreme APS intensities
goodAPSIntensity = aedat.data.polarity.apsIntGood;

%Ensure even number of pos/neg samples up to maxNumSamples
idxP = find(aedat.data.polarity.polarity>0 & ~nearEdgeIdx & qFilter & duringAPS>0 & goodAPSIntensity);
idxN = find(aedat.data.polarity.polarity==0 & ~nearEdgeIdx & qFilter & duringAPS>0 & goodAPSIntensity);
numSamples = min([sum(idxP) sum(idxN) round(inputVar.maxNumSamples./2)]);

%For positive events
ptsA = linspace(0,inputVar.maxProb,50);
% binWidth = maxProb/(numBins);
[pdf,~] = ksdensity(aedat.data.polarity.Prob(idxP),ptsA,'BoundaryCorrection','reflection');
pdf(pdf<eps) = eps;
%interpolate prob on ditribution
weight = log(1+1./interp1(ptsA,pdf,aedat.data.polarity.Prob(idxP),'nearest','extrap')); %use log of weight to avoid huge differences
randomSelection = datasample(idxP,numSamples,'Replace',false,'Weights',weight);
samples(randomSelection) = true;

%For negative events
ptsA = linspace(0,inputVar.maxProb,50);
% binWidth = maxProb/(numBins);
[pdf,~] = ksdensity(aedat.data.polarity.Prob(idxN),ptsA,'BoundaryCorrection','reflection');
pdf(pdf<eps) = eps;
%interpolate prob on ditribution
weight = log(1+1./interp1(ptsA,pdf,aedat.data.polarity.Prob(idxN),'nearest','extrap')); %use log of weight to avoid huge differences
randomSelection = datasample(idxN,numSamples,'Replace',false,'Weights',weight);
samples(randomSelection) = true;

totalSamples = sum(samples==true)
sampleList = find(samples);

chipSize = 2*inputVar.neighborhood + 1;

%initialize variables
[Xexp, Xhist] = deal(zeros(chipSize,chipSize,2*inputVar.depth,2,totalSamples)); %4th dim is causal/non-causal
Y = zeros(3,totalSamples);
% [r,c,f] = deal(nan(totalSamples,1));
priorSampleTime = 0;

%exp
[oldPos, oldNeg] = deal(zeros(aedat.data.frame.size(1), aedat.data.frame.size(2), numel(decayWeights)));
decayScalarBase = repmat(decayWeights,aedat.data.frame.size(1), aedat.data.frame.size(2),1);

%hist
[oldPosHist, oldNegHist] = deal(inf(aedat.data.frame.size(1), aedat.data.frame.size(2), numel(decayWeights)));

for sampleLoop = 1:numel(sampleList)
    
    clc, sampleLoop/numel(sampleList)
    
    currentSample = sampleList(sampleLoop);
    currentSampleTime = aedat.data.polarity.timeStamp(currentSample);
    
    addEventIdx = aedat.data.polarity.timeStamp >= priorSampleTime & ...
        aedat.data.polarity.timeStamp < currentSampleTime;
    
    %Build new-data surfaces
    p = addEventIdx & aedat.data.polarity.polarity>0;
    newPos = cell2mat(accumarray([aedat.data.polarity.y(p) aedat.data.polarity.x(p)],aedat.data.polarity.timeStamp(p)-currentSampleTime,aedat.data.frame.size,@(x) {sum(exp(x./decayWeights),1)},{zeros(1,1,numel(decayWeights))}));
    newPosHist = cell2mat(accumarray([aedat.data.polarity.y(p) aedat.data.polarity.x(p)],currentSampleTime-aedat.data.polarity.timeStamp(p),aedat.data.frame.size,@(x) {mink(cat(3,reshape(x,1,1,[]),inf(1,1,inputVar.depth)),inputVar.depth)},{inf(1,1,numel(decayWeights))}));
    p = addEventIdx & aedat.data.polarity.polarity<=0;
    newNeg = cell2mat(accumarray([aedat.data.polarity.y(p) aedat.data.polarity.x(p)],aedat.data.polarity.timeStamp(p)-currentSampleTime,aedat.data.frame.size,@(x) {sum(exp(x./decayWeights),1)},{zeros(1,1,numel(decayWeights))}));
    newNegHist = cell2mat(accumarray([aedat.data.polarity.y(p) aedat.data.polarity.x(p)],currentSampleTime-aedat.data.polarity.timeStamp(p),aedat.data.frame.size,@(x) {mink(cat(3,reshape(x,1,1,[]),inf(1,1,inputVar.depth)),inputVar.depth)},{inf(1,1,numel(decayWeights))}));
    
    %Decay existing surface(exp)
    decayScalar = exp((priorSampleTime - currentSampleTime)./decayScalarBase);
    oldPos = oldPos.*decayScalar + newPos;
    oldNeg = oldNeg.*decayScalar + newNeg;
    
    %Decay existing surface(hist)
    oldPosHist = oldPosHist + (currentSampleTime - priorSampleTime);
    oldPosHist = mink(cat(3,oldPosHist,newPosHist),inputVar.depth,3);
    oldNegHist = oldNegHist + (currentSampleTime - priorSampleTime);
    oldNegHist = mink(cat(3,oldNegHist,newNegHist),inputVar.depth,3);
    
    priorSampleTime = currentSampleTime;
    
    %         frameOffset = (frameIdx-1)*inputVar.maxNumSamples;
    %         for chipNum = 1:inputVar.maxNumSamples
    
    chipRow = aedat.data.polarity.y(currentSample);
    chipCol = aedat.data.polarity.x(currentSample);
    
    rChip = chipRow-inputVar.neighborhood:chipRow+inputVar.neighborhood;
    cChip = chipCol-inputVar.neighborhood:chipCol+inputVar.neighborhood;
    
    Xexp(:,:,:,1,sampleLoop) = cat(3, ...
        oldPos(rChip, cChip,:), ...
        oldNeg(rChip, cChip,:));
    Xhist(:,:,:,1,sampleLoop) = cat(3, ...
        oldPosHist(rChip, cChip,:), ...
        oldNegHist(rChip, cChip,:));
    Y(1,sampleLoop) =  aedat.data.polarity.Prob(currentSample);
    Y(2,sampleLoop) =  aedat.data.polarity.Jt(currentSample);
    Y(3,sampleLoop) =  aedat.data.polarity.polarity(currentSample);
    
end

if false
    %write out a full image for testing purposes
    X_all = cat(3, ...
        oldPos, ...
        oldNeg);
    Y_all(:,:,1) =  aedat.data.frame.I(:,:, currentFrame);
    Y_all(:,:,2) =  aedat.data.frame.Jt(:,:, currentFrame);
    Y_all(:,:,3) =  aedat.data.frame.Ix(:,:, currentFrame);
    Y_all(:,:,4) =  aedat.data.frame.Iy(:,:, currentFrame);
    Y_all(:,:,5) =  aedat.data.frame.Jx(:,:, currentFrame);
    Y_all(:,:,6) =  aedat.data.frame.Jy(:,:, currentFrame);
    Y_all(:,:,7) =  aedat.data.frame.Vx(:,:, currentFrame);
    Y_all(:,:,8) =  aedat.data.frame.Vy(:,:, currentFrame);
end


%Non-causal
priorSampleTime = 0;

%exp
[oldPos, oldNeg] = deal(zeros(aedat.data.frame.size(1), aedat.data.frame.size(2), numel(decayWeights)));
decayScalarBase = repmat(decayWeights,aedat.data.frame.size(1), aedat.data.frame.size(2),1);

%hist
[oldPosHist, oldNegHist] = deal(inf(aedat.data.frame.size(1), aedat.data.frame.size(2), numel(decayWeights)));

%reverse time
aedat.data.polarity.timeStamp = max(aedat.data.polarity.timeStamp) - aedat.data.polarity.timeStamp;

for sampleLoop = numel(sampleList):-1:1
    
    clc, sampleLoop/numel(sampleList)
    
    currentSample = sampleList(sampleLoop);
    currentSampleTime = aedat.data.polarity.timeStamp(currentSample);
    
    addEventIdx = aedat.data.polarity.timeStamp >= priorSampleTime & ...
        aedat.data.polarity.timeStamp < currentSampleTime;
    
    %Build new-data surfaces
    p = addEventIdx & aedat.data.polarity.polarity>0;
    newPos = cell2mat(accumarray([aedat.data.polarity.y(p) aedat.data.polarity.x(p)],aedat.data.polarity.timeStamp(p)-currentSampleTime,aedat.data.frame.size,@(x) {sum(exp(x./decayWeights),1)},{zeros(1,1,numel(decayWeights))}));
    newPosHist = cell2mat(accumarray([aedat.data.polarity.y(p) aedat.data.polarity.x(p)],currentSampleTime-aedat.data.polarity.timeStamp(p),aedat.data.frame.size,@(x) {mink(cat(3,reshape(x,1,1,[]),inf(1,1,inputVar.depth)),inputVar.depth)},{inf(1,1,numel(decayWeights))}));
    p = addEventIdx & aedat.data.polarity.polarity<=0;
    newNeg = cell2mat(accumarray([aedat.data.polarity.y(p) aedat.data.polarity.x(p)],aedat.data.polarity.timeStamp(p)-currentSampleTime,aedat.data.frame.size,@(x) {sum(exp(x./decayWeights),1)},{zeros(1,1,numel(decayWeights))}));
    newNegHist = cell2mat(accumarray([aedat.data.polarity.y(p) aedat.data.polarity.x(p)],currentSampleTime-aedat.data.polarity.timeStamp(p),aedat.data.frame.size,@(x) {mink(cat(3,reshape(x,1,1,[]),inf(1,1,inputVar.depth)),inputVar.depth)},{inf(1,1,numel(decayWeights))}));
    
    %Decay existing surface(exp)
    decayScalar = exp((priorSampleTime - currentSampleTime)./decayScalarBase);
    oldPos = oldPos.*decayScalar + newPos;
    oldNeg = oldNeg.*decayScalar + newNeg;
    
    %Decay existing surface(hist)
    oldPosHist = oldPosHist + (currentSampleTime - priorSampleTime);
    oldPosHist = mink(cat(3,oldPosHist,newPosHist),inputVar.depth,3);
    oldNegHist = oldNegHist + (currentSampleTime - priorSampleTime);
    oldNegHist = mink(cat(3,oldNegHist,newNegHist),inputVar.depth,3);
    
    priorSampleTime = currentSampleTime;
    
    %         frameOffset = (frameIdx-1)*inputVar.maxNumSamples;
    %         for chipNum = 1:inputVar.maxNumSamples
    
    chipRow = aedat.data.polarity.y(currentSample);
    chipCol = aedat.data.polarity.x(currentSample);
    
    rChip = chipRow-inputVar.neighborhood:chipRow+inputVar.neighborhood;
    cChip = chipCol-inputVar.neighborhood:chipCol+inputVar.neighborhood;
    
    Xexp(:,:,:,2,sampleLoop) = cat(3, ...
        oldPos(rChip, cChip,:), ...
        oldNeg(rChip, cChip,:));
    Xhist(:,:,:,2,sampleLoop) = cat(3, ...
        oldPosHist(rChip, cChip,:), ...
        oldNegHist(rChip, cChip,:));
   
end

%scale the history surfaces
%Set missing data to max
Xhist(isinf(Xhist)) = inputVar.maxTime;
%Scale values above 5 seconds (or maxTime) down to 5 sec
Xhist(Xhist>inputVar.maxTime) = inputVar.maxTime;
%Log scale the time data
Xhist = log(Xhist+1);
%Remove time information within 150 usec of the event
Xhist = Xhist - log(inputVar.minTime+1);
Xhist(Xhist<0) = 0;
% X = X ./ max(X(:));

% Y(Y>inputVar.maxProb) = inputVar.maxProb;
samples = sampleList;
