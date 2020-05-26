function aedat = loadMatWithAttributes(file)

%load the mat file
load(file)

[fp,fn,fe] = fileparts(file);

if isfield(aedat, 'importParams')
    disp('data already converted')
    return
end

aedat.importParams.filePath = file;
aedat.importParams.source = 'davis204c';

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

%flip
aedat.data.polarity.timeStamp = aedat.data.polarity.timeStamp';
aedat.data.polarity.polarity = aedat.data.polarity.polarity';

%repack samples
tmp = aedat.data.frame.samples;
aedat.data.frame.samples = cell(1,aedat.data.frame.numDiffImages);
for loop = 1:aedat.data.frame.numDiffImages
    %image should be flipped to match imu and flipped y values
    aedat.data.frame.samples{loop} = double(flipud(tmp(:,:,loop)));
end

%Read from settings file or hard code to default
% try
%    settings = xml2struct([fp filesep 'settings.xml']);
%    %find the index for the capture
%    for loop1 = 1:numel(settings.dv.node.node)
%        if strcmp(settings.dv.node.node{loop1}.Attributes.path,'/mainloop/capture/')
%            for loop2 = 1:numel(settings.dv.node.node{loop1}.node)
%                if strcmp(settings.dv.node.node{loop1}.node{loop2}.Attributes.path,'/mainloop/capture/DAVIS240C/')
%                    for loop3 = 1:numel(settings.dv.node.node{loop1}.node{loop2}.node)
%                         if strcmp(settings.dv.node.node{loop1}.node{loop2}.node{loop3}.Attributes.path,'/mainloop/capture/DAVIS240C/bias/')
%                             for loop4 = 1:numel(settings.dv.node.node{loop1}.node{loop2}.node{loop3}.node)
%                                 if strcmp(settings.dv.node.node{loop1}.node{loop2}.node{loop3}.node{loop4}.Attributes.path,'/mainloop/capture/DAVIS240C/bias/DiffBn/')
%                                     for loop5 = 1:numel(settings.dv.node.node{loop1}.node{loop2}.node{loop3}.node{loop4}.attr)
%                                         if strcmp(settings.dv.node.node{loop1}.node{loop2}.node{loop3}.node{loop4}.attr{loop5}.Attributes.key,'coarseValue')
%                                             diffC = str2num(settings.dv.node.node{loop1}.node{loop2}.node{loop3}.node{loop4}.attr{loop5}.Text);
%                                         elseif strcmp(settings.dv.node.node{loop1}.node{loop2}.node{loop3}.node{loop4}.attr{loop5}.Attributes.key,'fineValue')
%                                             diffF = str2num(settings.dv.node.node{loop1}.node{loop2}.node{loop3}.node{loop4}.attr{loop5}.Text);
%                                         end
%                                     end
%                                 elseif strcmp(settings.dv.node.node{loop1}.node{loop2}.node{loop3}.node{loop4}.Attributes.path,'/mainloop/capture/DAVIS240C/bias/OnBn/')
%                                     for loop5 = 1:numel(settings.dv.node.node{loop1}.node{loop2}.node{loop3}.node{loop4}.attr)
%                                         if strcmp(settings.dv.node.node{loop1}.node{loop2}.node{loop3}.node{loop4}.attr{loop5}.Attributes.key,'coarseValue')
%                                             diffOnC = str2num(settings.dv.node.node{loop1}.node{loop2}.node{loop3}.node{loop4}.attr{loop5}.Text);
%                                         elseif strcmp(settings.dv.node.node{loop1}.node{loop2}.node{loop3}.node{loop4}.attr{loop5}.Attributes.key,'fineValue')
%                                             diffOnF = str2num(settings.dv.node.node{loop1}.node{loop2}.node{loop3}.node{loop4}.attr{loop5}.Text);
%                                         end
%                                     end
%                                 elseif strcmp(settings.dv.node.node{loop1}.node{loop2}.node{loop3}.node{loop4}.Attributes.path,'/mainloop/capture/DAVIS240C/bias/OffBn/')
%                                     for loop5 = 1:numel(settings.dv.node.node{loop1}.node{loop2}.node{loop3}.node{loop4}.attr)
%                                         if strcmp(settings.dv.node.node{loop1}.node{loop2}.node{loop3}.node{loop4}.attr{loop5}.Attributes.key,'coarseValue')
%                                             diffOffC = str2num(settings.dv.node.node{loop1}.node{loop2}.node{loop3}.node{loop4}.attr{loop5}.Text);
%                                         elseif strcmp(settings.dv.node.node{loop1}.node{loop2}.node{loop3}.node{loop4}.attr{loop5}.Attributes.key,'fineValue')
%                                             diffOffF = str2num(settings.dv.node.node{loop1}.node{loop2}.node{loop3}.node{loop4}.attr{loop5}.Text);
%                                         end
%                                     end
%                                 end
%                             end
%                         end
%                    end
%                end
%            end
%        end
%    end
% catch
%     msgbox('cannot find biases - using defaults')
%     diffC = 4;
%     diffF = 39;
%     diffOnC = 5;
%     diffOnF = 255;
%     diffOffC = 4;
%     diffOffF = 0;
    diffC = 3;
    diffF = 39;
    diffOnC = 2;
    diffOnF = 255;
    diffOffC = 3;
    diffOffF = 1;
% end
%    aedat.info.

%Calculate bias diff setting values
courseBiasLookupTable = fliplr(logspace(log10(12e-12),log10(25e-6),8));
diffFineTable = linspace(0,courseBiasLookupTable(diffC+1),257);
aedat.info.diffB = diffFineTable(diffF+2);
diffOnFineTable = linspace(0,courseBiasLookupTable(diffOnC+1),257);
aedat.info.diffOn = diffOnFineTable(diffOnF+2);
diffOffFineTable = linspace(0,courseBiasLookupTable(diffOffC+1),257);
aedat.info.diffOff = diffOffFineTable(diffOffF+2);

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
