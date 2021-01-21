function [noisyScore, denoiseScore] = scoreDenoise(aedat, YPred)

%RPMD from paper

%YPred~=0 means likely noise, YPred~=1 means likely valid event (use nans
%if prediction is not possible

%Only score events where EPM can generate valid data
validEventsWithinFrameIdx = ~isnan(YPred) & (aedat.data.polarity.duringAPS>0) & aedat.data.polarity.apsIntGood;

%Filter out data with multiple events per time window tau
rpsData = [aedat.data.polarity.x(validEventsWithinFrameIdx) aedat.data.polarity.y(validEventsWithinFrameIdx) aedat.data.polarity.polarity(validEventsWithinFrameIdx) aedat.data.polarity.duringAPS(validEventsWithinFrameIdx) YPred(validEventsWithinFrameIdx)];
[rpsData,sIndex] = sortrows(rpsData,5,'descend');
[~,ia,~] = unique(rpsData(:,1:4),'first','rows');
onePerLocation = find(validEventsWithinFrameIdx);
onePerLocation = onePerLocation(sIndex(ia));
validEventsWithinFrameIdxOnlyOne = false(size(validEventsWithinFrameIdx));
validEventsWithinFrameIdxOnlyOne(onePerLocation) = true;

N = numel(onePerLocation);
logOptimalScore = 1/N.*(sum(log(aedat.data.polarity.Prob(validEventsWithinFrameIdxOnlyOne & aedat.data.polarity.Prob>0.5))) + ...
    sum(log(1-aedat.data.polarity.Prob(validEventsWithinFrameIdxOnlyOne & aedat.data.polarity.Prob<=0.5))));

logAllScore = 1/N.*sum(log(max(1-aedat.data.polarity.Prob(validEventsWithinFrameIdxOnlyOne),realmin)));
noisyScore = logOptimalScore - logAllScore;
disp(['Noisy: ' num2str(noisyScore)])

logDenoiseScore = 1/N.*(sum(log(max(aedat.data.polarity.Prob(validEventsWithinFrameIdxOnlyOne & (YPred>0.5)),realmin))) + ...
    sum(log(max(1-aedat.data.polarity.Prob(validEventsWithinFrameIdxOnlyOne & (YPred<=0.5)),realmin))));
denoiseScore = logOptimalScore - logDenoiseScore;
disp(['Denoised: ' num2str(denoiseScore)])
