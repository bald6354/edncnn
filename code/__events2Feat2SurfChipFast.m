function [Xexp, Xhist, Y, samples] = events2Feat2SurfChipFast(aedat, inputVar)

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

numFrames = size(aedat.data.frame.I,3);
sampleFrames = round(numFrames/2):round(numFrames*3/4-1); %just look at the 3rd quarter frames of each sequence (first ~3 sec is IMU cal)

nearEdgeIdx = ones(aedat.data.frame.size);
nearEdgeIdx(inputVar.neighborhood+1:end-inputVar.neighborhood,inputVar.neighborhood+1:end-inputVar.neighborhood) = 0;

chipSize = 2*inputVar.neighborhood + 1;

[Xexp, Xhist] = deal(zeros(chipSize,chipSize,2*inputVar.depth,inputVar.maxNumSamples*numel(sampleFrames)));
Y = zeros(chipSize,chipSize,8,inputVar.maxNumSamples*numel(sampleFrames));
[r,c,f] = deal(nan(inputVar.maxNumSamples*numel(sampleFrames),1));

badFrameList = []; %bad frames are frames that do not have enough valid pixels
priorFrameTime = 0;

%exp
[oldPos, oldNeg] = deal(zeros(aedat.data.frame.size(1), aedat.data.frame.size(2), numel(decayWeights)));
decayScalarBase = repmat(decayWeights,aedat.data.frame.size(1), aedat.data.frame.size(2),1);

%hist
[oldPosHist, oldNegHist] = deal(inf(aedat.data.frame.size(1), aedat.data.frame.size(2), numel(decayWeights)));

for frameIdx = 1:numel(sampleFrames)
    
    clc, frameIdx/numel(sampleFrames)
    
    try
        
        currentFrame = sampleFrames(frameIdx);
        currentFrameTime = frameTimes(currentFrame);
        %     imagesc(accumarray([aedat.data.polarity.y(yIdx&xIdx&tIdx) aedat.data.polarity.x(yIdx&xIdx&tIdx)],exp(double(aedat.data.polarity.timeStamp(yIdx&xIdx&tIdx)-aedat.data.polarity.timeStamp(eventIdx))./1e5),[],@sum))
        
        % tic
        
        if priorFrameTime == 0
            addEventIdx = aedat.data.polarity.timeStamp <= currentFrameTime;
        else
            addEventIdx = aedat.data.polarity.timeStamp > priorFrameTime & ...
                aedat.data.polarity.timeStamp <= currentFrameTime;
        end
        
        %Build new-data surfaces
        p = addEventIdx & aedat.data.polarity.polarity>0;
        newPos = cell2mat(accumarray([aedat.data.polarity.y(p) aedat.data.polarity.x(p)],aedat.data.polarity.timeStamp(p)-currentFrameTime,aedat.data.frame.size,@(x) {sum(exp(x./decayWeights),1)},{zeros(1,1,numel(decayWeights))}));
        newPosHist = cell2mat(accumarray([aedat.data.polarity.y(p) aedat.data.polarity.x(p)],currentFrameTime-aedat.data.polarity.timeStamp(p),aedat.data.frame.size,@(x) {mink(cat(3,reshape(x,1,1,[]),inf(1,1,inputVar.depth)),inputVar.depth)},{inf(1,1,numel(decayWeights))}));
        p = addEventIdx & aedat.data.polarity.polarity<=0;
        newNeg = cell2mat(accumarray([aedat.data.polarity.y(p) aedat.data.polarity.x(p)],aedat.data.polarity.timeStamp(p)-currentFrameTime,aedat.data.frame.size,@(x) {sum(exp(x./decayWeights),1)},{zeros(1,1,numel(decayWeights))}));
        newNegHist = cell2mat(accumarray([aedat.data.polarity.y(p) aedat.data.polarity.x(p)],currentFrameTime-aedat.data.polarity.timeStamp(p),aedat.data.frame.size,@(x) {mink(cat(3,reshape(x,1,1,[]),inf(1,1,inputVar.depth)),inputVar.depth)},{inf(1,1,numel(decayWeights))}));
        
        %Decay existing surface(exp)
        decayScalar = exp((priorFrameTime - currentFrameTime)./decayScalarBase);
        oldPos = oldPos.*decayScalar + newPos;
        oldNeg = oldNeg.*decayScalar + newNeg;
        
        %Decay existing surface(hist)
        oldPosHist = oldPosHist + (currentFrameTime - priorFrameTime);
        oldPosHist = mink(cat(3,oldPosHist,newPosHist),inputVar.depth,3);
        oldNegHist = oldNegHist + (currentFrameTime - priorFrameTime);
        oldNegHist = mink(cat(3,oldNegHist,newNegHist),inputVar.depth,3);
        
        priorFrameTime = currentFrameTime;
        %
%         priorEventIdx = aedat.data.polarity.timeStamp <= currentFrameTime;
%         % toc
%         
%         p = priorEventIdx & aedat.data.polarity.polarity>0;
%         q1 = cell2mat(accumarray([aedat.data.polarity.y(p) aedat.data.polarity.x(p)],aedat.data.polarity.timeStamp(p)-currentFrameTime,aedat.data.frame.size,@(x) {sum(exp(x./decayWeights),1)},{zeros(1,1,numel(decayWeights))}));
%         p = priorEventIdx & aedat.data.polarity.polarity<=0;
%         q2 = cell2mat(accumarray([aedat.data.polarity.y(p) aedat.data.polarity.x(p)],aedat.data.polarity.timeStamp(p)-currentFrameTime,aedat.data.frame.size,@(x) {sum(exp(x./decayWeights),1)},{zeros(1,1,numel(decayWeights))}));
%         
        %Pick random samples from the good aps area
        goodAps = aedat.data.frame.apsIntGood(:,:,currentFrame);
        %erode the size of the chip to ensure the entire chip is good
        goodAps2 = imerode(goodAps,ones(inputVar.neighborhood));
        potentialChipCenters = find(goodAps2&~nearEdgeIdx);
%         randomSelection = randperm(numel(potentialChipCenters),inputVar.maxNumSamples);
      
        %Attempt to get a uniform distributions for Jt
        currentJt = aedat.data.frame.Jt(:,:,currentFrame);
%         sampleWeights = 1./pdf(pd, currentJt(potentialChipCenters));
        sampleWeights = currentJt(potentialChipCenters).^2; %bias toward sampling further from zero
%         [f,xi] = ksdensity(currentJt(potentialChipCenters),'Support',[-500 500]); 
        randomSelection = datasample(potentialChipCenters,inputVar.maxNumSamples,'Replace',false,'Weights',sampleWeights);
%         histogram(currentJt(randomSelection))
        [chipRow, chipCol] = ind2sub(aedat.data.frame.size, randomSelection);
        
        frameOffset = (frameIdx-1)*inputVar.maxNumSamples;
        for chipNum = 1:inputVar.maxNumSamples
            rChip = chipRow(chipNum)-inputVar.neighborhood:chipRow(chipNum)+inputVar.neighborhood;
            cChip = chipCol(chipNum)-inputVar.neighborhood:chipCol(chipNum)+inputVar.neighborhood;
            
            Xexp(:,:,:,frameOffset+chipNum) = cat(3, ...
                oldPos(rChip, cChip,:), ...
                oldNeg(rChip, cChip,:));
            Xhist(:,:,:,frameOffset+chipNum) = cat(3, ...
                oldPosHist(rChip, cChip,:), ...
                oldNegHist(rChip, cChip,:));
            Y(:,:,1,frameOffset+chipNum) =  aedat.data.frame.I(rChip, cChip, currentFrame);
            Y(:,:,2,frameOffset+chipNum) =  aedat.data.frame.Jt(rChip, cChip, currentFrame);
            Y(:,:,3,frameOffset+chipNum) =  aedat.data.frame.Ix(rChip, cChip, currentFrame);
            Y(:,:,4,frameOffset+chipNum) =  aedat.data.frame.Iy(rChip, cChip, currentFrame);
            Y(:,:,5,frameOffset+chipNum) =  aedat.data.frame.Jx(rChip, cChip, currentFrame);
            Y(:,:,6,frameOffset+chipNum) =  aedat.data.frame.Jy(rChip, cChip, currentFrame);
            Y(:,:,7,frameOffset+chipNum) =  aedat.data.frame.Vx(rChip, cChip, currentFrame);
            Y(:,:,8,frameOffset+chipNum) =  aedat.data.frame.Vy(rChip, cChip, currentFrame);
            r(frameOffset+chipNum) = chipRow(chipNum);
            c(frameOffset+chipNum) = chipCol(chipNum);
            f(frameOffset+chipNum) = currentFrame;
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
            
    catch
        badFrameList(end+1) = frameIdx
        disp('bad frame')
    end
    
end

%if some frames had bad data remove them
rmIdx = isnan(f);
Xexp(:,:,:,rmIdx) = [];
Xhist(:,:,:,rmIdx) = [];
Y(:,:,:,rmIdx) = [];
r(rmIdx) = [];
c(rmIdx) = [];
f(rmIdx) = [];

samples.r = r;
samples.c = c;
samples.f = f;
