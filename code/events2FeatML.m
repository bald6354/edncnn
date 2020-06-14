function [X, Y, samples, pol] = events2FeatML(aedat, inputVar)

%Generate features from DVS events

%Setup sensor size
numRows = double(aedat.data.frame.size(1));
numCols = double(aedat.data.frame.size(2));

%Calculate time differences
ts = cat(1,0,diff(aedat.data.polarity.timeStamp));

%generate a weighted random sample the data to get a uniform sample of
%probabilities
samples = false(aedat.data.polarity.numEvents,1);

% Filter to middle of recorded data (avoid temporal edges)
timeQuantiles = quantile(aedat.data.polarity.timeStamp,[0.2 0.8]);
qFilter = aedat.data.polarity.timeStamp >= timeQuantiles(1) & ...
    aedat.data.polarity.timeStamp <= timeQuantiles(2);

%Filter to DVS data during APS
duringAPS = (aedat.data.polarity.duringAPS > 0);

%do not to sample near a spatial edge
nearEdgeIdx = ((aedat.data.polarity.y-inputVar.neighborhood) < 1) | ...
    ((aedat.data.polarity.x-inputVar.neighborhood) < 1) | ...
    ((aedat.data.polarity.y+inputVar.neighborhood) > numRows) | ...
    ((aedat.data.polarity.x+inputVar.neighborhood) > numCols);

%Do not sample events calculated from extreme APS intensities
goodAPSIntensity = aedat.data.polarity.apsIntGood;

%Ensure even number of pos/neg samples up to maxNumSamples
idxP = aedat.data.polarity.polarity>0 & ~nearEdgeIdx & qFilter & duringAPS>0 & goodAPSIntensity;
idxN = aedat.data.polarity.polarity==0 & ~nearEdgeIdx & qFilter & duringAPS>0 & goodAPSIntensity;
numSamples = min([sum(idxP) sum(idxN) round(inputVar.maxNumSamples./2)]);

%Pos
if numSamples == sum(idxP)
    samples(idxP) = true;
else
    ptsA = linspace(0,inputVar.maxProb,50);
    % binWidth = maxProb/(numBins);
    [pdf,~] = ksdensity(aedat.data.polarity.Prob(idxP),ptsA,'BoundaryCorrection','reflection');
    pdf(pdf<eps) = eps;
    %interpolate prob on ditribution
    weight = log(1+1./interp1(ptsA,pdf,aedat.data.polarity.Prob(idxP),'nearest','extrap')); %use log of weight to avoid huge differences
    frameVals = find(idxP);
    randSampIdx = unique(randsample(numel(frameVals),numSamples,true,weight));
    while numel(randSampIdx) < numSamples
        randSampIdx = unique(cat(1,randSampIdx,randsample(numel(frameVals),numSamples,true,weight)));
    end
    randSampIdx = randSampIdx(1:numSamples);
    samples(frameVals(randSampIdx)) = true;
end

%Neg
if numSamples == sum(idxN)
    samples(idxN) = true;
else
    % ptsA = linspace(0,inputVar.maxProb,100);
    % binWidth = maxProb/(numBins);
    [pdf,~] = ksdensity(aedat.data.polarity.Prob(idxN),ptsA,'BoundaryCorrection','reflection');
    pdf(pdf<eps) = eps;
    %interpolate prob on ditribution
    weight = log(1+1./interp1(ptsA,pdf,aedat.data.polarity.Prob(idxN),'nearest','extrap'));
    frameVals = find(idxN);
    randSampIdx = unique(randsample(numel(frameVals),numSamples,true,weight));
    while numel(randSampIdx) < numSamples
        randSampIdx = unique(cat(1,randSampIdx,randsample(numel(frameVals),numSamples,true,weight)));
    end
    randSampIdx = randSampIdx(1:numSamples);
    samples(frameVals(randSampIdx)) = true;
end

totalSamples = sum(samples==true)

%Create blank surfaces
[sp, sn] = deal(nan(numRows,numCols,inputVar.depth));

X = nan(2.*inputVar.neighborhood+1,2.*inputVar.neighborhood+1,2*inputVar.depth*(inputVar.nonCausal+1),totalSamples,'single');
Y = nan(totalSamples,1);
cntr = 1;
clf
tic
disp('Forward time')
for eventIdx = 1:aedat.data.polarity.numEvents
    
    %update periodically
    if mod(eventIdx,100000)==0
        clc, eventIdx./aedat.data.polarity.numEvents
        imagesc(flipud(-1.*(log(sn(:,:,1)))))
        pause(.0001)
    end
    
    %Shift the surface with the timestep
    if ts(eventIdx)>0
        sp = sp + double(ts(eventIdx));
        sn = sn + double(ts(eventIdx));
    end
    
    if (samples(eventIdx)==true)
        
        %Capture the surface as feature
        rows = aedat.data.polarity.y(eventIdx)-(inputVar.neighborhood):aedat.data.polarity.y(eventIdx)+(inputVar.neighborhood);
        cols = aedat.data.polarity.x(eventIdx)-(inputVar.neighborhood):aedat.data.polarity.x(eventIdx)+(inputVar.neighborhood);
        
        %Top/bottom switch based on polarity of event
        if aedat.data.polarity.polarity(eventIdx) == 1
            X(:,:,1:2*inputVar.depth,cntr) = cat(3,sp(rows,cols,:),sn(rows,cols,:));
        else
            X(:,:,1:2*inputVar.depth,cntr) = cat(3,sn(rows,cols,:),sp(rows,cols,:));
        end
        
        Y(cntr) = aedat.data.polarity.Prob(eventIdx);
        pol(cntr) = aedat.data.polarity.polarity(eventIdx);
        
        cntr = cntr + 1;
    end
    
    %Update the surface
    if aedat.data.polarity.polarity(eventIdx) == 1
        sp(aedat.data.polarity.y(eventIdx),aedat.data.polarity.x(eventIdx),:) = ...
            cat(3, 0, sp(aedat.data.polarity.y(eventIdx),aedat.data.polarity.x(eventIdx),1:end-1));
    else
        sn(aedat.data.polarity.y(eventIdx),aedat.data.polarity.x(eventIdx),:) = ...
            cat(3, 0, sn(aedat.data.polarity.y(eventIdx),aedat.data.polarity.x(eventIdx),1:end-1));
    end
end
toc

if inputVar.nonCausal
    [sp, sn] = deal(nan(numRows,numCols,inputVar.depth));
    ts = cat(1,diff(aedat.data.polarity.timeStamp),0);
    cntr = cntr - 1;
    clf
    disp('Reverse time')
    for eventIdx = aedat.data.polarity.numEvents:-1:1
        
        %update periodically
        if mod(eventIdx,100000)==0
            clc, eventIdx./aedat.data.polarity.numEvents
            imagesc(flipud(-1.*(log(sn(:,:,1)))))
            pause(.0001)
        end
        
        %Shift the surface with the timestep
        if ts(eventIdx)>0
            sp = sp + double(ts(eventIdx));
            sn = sn + double(ts(eventIdx));
        end
        
        if (samples(eventIdx)==true)
            
            %Capture the surface as feature
            rows = aedat.data.polarity.y(eventIdx)-(inputVar.neighborhood):aedat.data.polarity.y(eventIdx)+(inputVar.neighborhood);
            cols = aedat.data.polarity.x(eventIdx)-(inputVar.neighborhood):aedat.data.polarity.x(eventIdx)+(inputVar.neighborhood);
            
            %Top/bottom switch based on polarity of event
            if aedat.data.polarity.polarity(eventIdx) == 1
                X(:,:,2*inputVar.depth+1:end,cntr) = cat(3,sp(rows,cols,:),sn(rows,cols,:));
            else
                X(:,:,2*inputVar.depth+1:end,cntr) = cat(3,sn(rows,cols,:),sp(rows,cols,:));
            end
            
            cntr = cntr - 1;
        end
        
        %Update the surface
        if aedat.data.polarity.polarity(eventIdx) == 1
            sp(aedat.data.polarity.y(eventIdx),aedat.data.polarity.x(eventIdx),:) = ...
                cat(3, 0, sp(aedat.data.polarity.y(eventIdx),aedat.data.polarity.x(eventIdx),1:end-1));
        else
            sn(aedat.data.polarity.y(eventIdx),aedat.data.polarity.x(eventIdx),:) = ...
                cat(3, 0, sn(aedat.data.polarity.y(eventIdx),aedat.data.polarity.x(eventIdx),1:end-1));
        end
    end
    toc
end

%% Scale the data to reasonable ranges

%Set missing data to max
X(isnan(X)) = inputVar.maxTime;
%Scale values above 5 seconds (or maxTime) down to 5 sec
X(X>inputVar.maxTime) = inputVar.maxTime;
%Log scale the time data
X = log(X+1);
%Remove time information within 150 usec of the event (150usec is limited by sensor temporal accuracy)
X = X - log(inputVar.minTime+1);
X(X<0) = 0;

Y(Y>inputVar.maxProb) = inputVar.maxProb;

