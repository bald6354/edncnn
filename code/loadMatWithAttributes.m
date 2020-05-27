function aedat = loadMatWithAttributes(file)

%load the mat file
load(file, 'aedat')

if isfield(aedat, 'importParams')
    disp('data already converted')
    return
end

aedat.importParams.filePath = file;

%add attributes
aedat.data.frame.expTime = aedat.data.frame.expEnd - aedat.data.frame.expStart;
aedat.data.frame.diffImTime = [];

%make this a double to work with previous code
aedat.data.polarity.numEvents = double(aedat.data.polarity.numEvents);

%ensure size is in correct order
aedat.data.frame.size = sort(aedat.data.frame.size);

%0-based to 1-based indexing
aedat.data.polarity.x = aedat.data.polarity.x' + 1;
%flip y values to match imu data
aedat.data.polarity.y = aedat.data.frame.size(1) - aedat.data.polarity.y';

%flip orientation
aedat.data.polarity.timeStamp = aedat.data.polarity.timeStamp';
aedat.data.polarity.polarity = aedat.data.polarity.polarity';

%repack samples
tmp = aedat.data.frame.samples;
aedat.data.frame.samples = cell(1,aedat.data.frame.numDiffImages);
for loop = 1:aedat.data.frame.numDiffImages
    %image should be flipped to match imu and flipped y values
    aedat.data.frame.samples{loop} = double(flipud(tmp(:,:,loop)));
end

%estimate rolling shutter times
for loop = 1:aedat.data.frame.numDiffImages
    aedat.data.frame.diffImStartTime{loop} = repmat(double(aedat.data.frame.expStart(loop)),aedat.data.frame.size(1),aedat.data.frame.size(2));
    aedat.data.frame.diffImEndTime{loop} = repmat(double(aedat.data.frame.expEnd(loop)),aedat.data.frame.size(1),aedat.data.frame.size(2));
    aedat.data.frame.diffImTime{loop} = mean(cat(3,aedat.data.frame.diffImStartTime{loop},aedat.data.frame.diffImEndTime{loop}),3);
end

% %estimate rolling shutter times - not likely accurate
% for loop = 1:aedat.data.frame.numDiffImages
%     rollingTimeStart = linspace(double(aedat.data.frame.expStart(loop)),double(aedat.data.frame.frameEnd(loop))-(double(aedat.data.frame.expTime(loop))),aedat.data.frame.size(2));
%     aedat.data.frame.diffImStartTime{loop} = repmat(rollingTimeStart,aedat.data.frame.size(1),1);
%     rollingTimeEnd = linspace(double(aedat.data.frame.expEnd(loop)),double(aedat.data.frame.frameEnd(loop)),aedat.data.frame.size(2));
%     aedat.data.frame.diffImEndTime{loop} = repmat(rollingTimeEnd,aedat.data.frame.size(1),1);
%     aedat.data.frame.diffImTime{loop} = mean(cat(3,aedat.data.frame.diffImStartTime{loop},aedat.data.frame.diffImEndTime{loop}),3);
% end
