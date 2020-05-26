function aedat = eventTiming(aedat, inputVar)

% h = figure;
% axis tight manual % this ensures that getframe() returns a consistent size
% filename = 'Jt.gif';

%Match each event to an image
% % magicTime = mean([double(aedat.data.frame.diffImTime) double(aedat.data.frame.diffImEndTime)],2);
% magicTime = double(aedat.data.frame.diffImEndTime);
% magicOffsetMin = mean(double(aedat.data.frame.diffImStartTime) - aedat.data.frame.diffImTime);
% magicOffsetMax = mean(double(aedat.data.frame.diffImEndTime) - aedat.data.frame.diffImTime);
% for magicOffset = linspace(magicOffsetMin,magicOffsetMax,10)

numRows = double(aedat.data.frame.size(1));
numCols = double(aedat.data.frame.size(2));

%for each frame
% for fLoop = 1:aedat.data.frame.numDiffImages
diffImTime = cell2mat(aedat.data.frame.diffImTime);
diffImTime = reshape(diffImTime,aedat.data.frame.size(1),aedat.data.frame.size(2),[]);

isRollingShutter = max(max(max(diffImTime,[],1),[],2)-min(min(diffImTime,[],1),[],2))>0;
aedat.cameraSetup.isRollingShutter = isRollingShutter;

magicOffset = 0;
allMagicTimes = median(diffImTime,1) + magicOffset;
%match each event to a frame by frame/column
if isRollingShutter
    for cLoop = 1:(aedat.data.frame.size(2))
        cIdx = (aedat.data.polarity.x) == cLoop;
        magicTime = squeeze(allMagicTimes(:,cLoop,:));
        aedat.data.polarity.closestFrame(cIdx,1) = interp1(magicTime,double([1:aedat.data.frame.numDiffImages]),double(aedat.data.polarity.timeStamp(cIdx)),'nearest','extrap');
        aedat.data.polarity.frameTimeDelta(cIdx,1) = double(aedat.data.polarity.timeStamp(cIdx)) - double(magicTime(aedat.data.polarity.closestFrame(cIdx)));
    end
else
    %All columns have same exposure window
    magicTime = squeeze(median(allMagicTimes,2));
    aedat.data.polarity.closestFrame(:,1) = interp1(magicTime,double([1:aedat.data.frame.numDiffImages]),double(aedat.data.polarity.timeStamp),'nearest','extrap');
    aedat.data.polarity.frameTimeDelta(:,1) = double(aedat.data.polarity.timeStamp) - double(magicTime(aedat.data.polarity.closestFrame));
end
    
aedat.data.polarity.duringAPS = zeros(size(aedat.data.polarity.x));

st = reshape(cell2mat(aedat.data.frame.diffImStartTime),numRows,numCols,[]);
et = reshape(cell2mat(aedat.data.frame.diffImEndTime),numRows,numCols,[]);

%Find events that occured during frame integration
if isRollingShutter
    st = median(st,1);
    et = median(et,1);
    for cLoop = 1:(aedat.data.frame.size(2))
        cIdx = find((aedat.data.polarity.x) == cLoop);
        stLoop = squeeze(st(:,cLoop,:));
        etLoop = squeeze(et(:,cLoop,:));
        eventTime = aedat.data.polarity.timeStamp(cIdx);
        for fLoop = 1:numel(stLoop)
            idx = (eventTime >= stLoop(fLoop)) & (eventTime <= etLoop(fLoop));
            aedat.data.polarity.duringAPS(cIdx(idx)) = fLoop;
        end
    end
else
    st = squeeze(median(median(st,1),2));
    et = squeeze(median(median(et,1),2));
    for fLoop = 1:numel(st)
        idx = (aedat.data.polarity.timeStamp >= st(fLoop)) & (aedat.data.polarity.timeStamp <= et(fLoop));
        aedat.data.polarity.duringAPS(idx) = fLoop;
    end
end

%convert quiver motion from pix/sec to pix/frame
fDiff = diff(diffImTime,1,3);
estFramesPerSec = 1e6/mode(fDiff(:))
aedat.cameraSetup.estFramesPerSec = estFramesPerSec;

%Estimate integration time
diffImStartTime=cell2mat(aedat.data.frame.diffImStartTime);
diffImStartTime = reshape(diffImStartTime,aedat.data.frame.size(1),aedat.data.frame.size(2),[]);
diffImEndTime=cell2mat(aedat.data.frame.diffImEndTime);
diffImEndTime = reshape(diffImEndTime,aedat.data.frame.size(1),aedat.data.frame.size(2),[]);
estIntegrationTime = mode(diffImEndTime(:) - diffImStartTime(:)) ./ 1e6
aedat.cameraSetup.estIntegrationTime = estIntegrationTime;

%Interpolate IMU data for the frameTime - MAY NEED TO ZERO IMU (WITH NO
%MOTION NOT ALLWAYS CENTERED ON ZERO)
if isRollingShutter
    aedat.data.frame.imu = interpolateIMU(aedat.data.imu6, squeeze(allMagicTimes(:)), true, inputVar.removeGyroBias);
else
    aedat.data.frame.imu = interpolateIMU(aedat.data.imu6, magicTime, true, inputVar.removeGyroBias);
end

% %Find fastest firing time
% idx = aedat.data.polarity.polarity == 1;
% fastestFireP = accumarray([aedat.data.polarity.y(idx)+1 aedat.data.polarity.x(idx)+1],aedat.data.polarity.timeStamp(idx),[180 240],@(x) min(diff(cat(1,sort(double(x)),Inf))),Inf);
% idx = aedat.data.polarity.polarity == 0;
% fastestFireN = accumarray([aedat.data.polarity.y(idx)+1 aedat.data.polarity.x(idx)+1],aedat.data.polarity.timeStamp(idx),[180 240],@(x) min(diff(cat(1,sort(double(x)),Inf))),Inf);
% clf
% histogram(fastestFireN)
% hold on
% histogram(fastestFireP)
% legend({'Neg','Pos'})
% title('Min usec between firing')
% pause(2)

%calculate relative pixel offsets
% aps = reshape(cell2mat(aedat.data.frame.samples),aedat.data.frame.size(1),aedat.data.frame.size(2),[]);
% rowOffset = median(diff(aps,1,1),3);
% colOffset = median(diff(aps,1,2),3);

