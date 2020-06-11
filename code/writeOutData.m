function writeOutData(outDir)
%function that takes processed aedat with Jt and generates files for training

%where are the .mat files
files =  dir([outDir '*_epm.mat']);

for fLoop = 1:numel(files)

    clear aedat inputVar X Y
    
    [~,fn,~] = fileparts(files(fLoop).name);
    if exist([outDir fn '_labels.mat'], 'file')
        disp('file already processed')
        pause(1)
        continue
    end
    
    load([outDir files(fLoop).name], 'aedat');
    load([outDir files(fLoop).name], 'inputVar');

    % Process event points into feature vectors
    [X, Y, samples, ~] = events2FeatML(aedat, inputVar); %edited 13FEB2020
    
    save([outDir fn '_labels.mat'],'X','Y','samples','-v7.3')
    
end

