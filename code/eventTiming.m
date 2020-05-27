function aedat = eventTiming(aedat, inputVar)

%image size
numRows = double(aedat.data.frame.size(1));
numCols = double(aedat.data.frame.size(2));

%load in the aps image times (this is per pixel in case of rolling shutter)
diffImTime = cell2mat(aedat.data.frame.diffImTime);
diffImTime = reshape(diffImTime,aedat.data.frame.size(1),aedat.data.frame.size(2),[]);

%Was this a rolling shutter or global shutter
isRollingShutter = max(max(max(diffImTime,[],1),[],2)-min(min(diffImTime,[],1),[],2))>0;
aedat.cameraSetup.isRollingShutter = isRollingShutter;

medImageTime = median(diffImTime,1);
%match each event to a frame by frame/column
if isRollingShutter
    for cLoop = 1:(aedat.data.frame.size(2))
        cIdx = (aedat.data.polarity.x) == cLoop;
        apsTime = squeeze(medImageTime(:,cLoop,:));
        aedat.data.polarity.closestFrame(cIdx,1) = interp1(apsTime,double([1:aedat.data.frame.numDiffImages]),double(aedat.data.polarity.timeStamp(cIdx)),'nearest','extrap');
        aedat.data.polarity.frameTimeDelta(cIdx,1) = double(aedat.data.polarity.timeStamp(cIdx)) - double(apsTime(aedat.data.polarity.closestFrame(cIdx)));
    end
else
    %All columns have same exposure window - just match to a frame
    apsTime = squeeze(median(medImageTime,2));
    aedat.data.polarity.closestFrame(:,1) = interp1(apsTime,double([1:aedat.data.frame.numDiffImages]),double(aedat.data.polarity.timeStamp),'nearest','extrap');
    aedat.data.polarity.frameTimeDelta(:,1) = double(aedat.data.polarity.timeStamp) - double(apsTime(aedat.data.polarity.closestFrame));
end

%duringAPS = DVS event occur during the APS integration time
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

%used to convert motion from pix/sec to pix/frame
fDiff = diff(diffImTime,1,3);
estFramesPerSec = 1e6/mode(fDiff(:));
aedat.cameraSetup.estFramesPerSec = estFramesPerSec;

%Estimate integration time
diffImStartTime = cell2mat(aedat.data.frame.diffImStartTime);
diffImStartTime = reshape(diffImStartTime,aedat.data.frame.size(1),aedat.data.frame.size(2),[]);
diffImEndTime = cell2mat(aedat.data.frame.diffImEndTime);
diffImEndTime = reshape(diffImEndTime,aedat.data.frame.size(1),aedat.data.frame.size(2),[]);
estIntegrationTime = mode(diffImEndTime(:) - diffImStartTime(:)) ./ 1e6;
aedat.cameraSetup.estIntegrationTime = estIntegrationTime;

%Interpolate IMU data for the frameTime - MAY NEED TO ZERO IMU (WITH NO
%MOTION NOT ALLWAYS CENTERED ON ZERO)
if isRollingShutter
    aedat.data.frame.imu = interpolateIMU(aedat.data.imu6, squeeze(medImageTime(:)), true, inputVar.removeGyroBias);
else
    aedat.data.frame.imu = interpolateIMU(aedat.data.imu6, apsTime, true, inputVar.removeGyroBias);
end


