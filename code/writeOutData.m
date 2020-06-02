function writeOutData(outDir)
%function that takes processed aedat with Jt and generates files for training

%where are the .mat files
files =  dir([outDir '*_epm.mat']);

for fLoop = 1:numel(files)

    clear aedat Xexp Xhist Y
    
    [~,fn,~] = fileparts(files(fLoop).name);
    if exist([outDir fn '_labels.mat'], 'file')
        disp('file already processed')
        pause(1)
        continue
    end
    
    load([outDir files(fLoop).name], 'aedat');
    load([outDir files(fLoop).name], 'inputVar');

    
%     %delete unneeded data from aedat
%     aedat2.data.frame.size = aedat.data.frame.size;
%     aedat2.data.polarity.numEvents = aedat.data.polarity.numEvents;
%     aedat2.data.polarity.x = aedat.data.polarity.x;
%     aedat2.data.polarity.y = aedat.data.polarity.y;
%     aedat2.data.polarity.timeStamp = aedat.data.polarity.timeStamp;
%     aedat2.data.polarity.polarity = aedat.data.polarity.polarity;
% %     aedat2.data.polarity.duringAPS =uint8(aedat.data.polarity.duringAPS>0);
% %     aedat2.data.polarity.apsIntGood = aedat.data.polarity.apsIntGood;
% %     aedat2.data.polarity.Prob = aedat.data.polarity.Prob;
% %     aedat2.data.polarity.Jt = aedat.data.polarity.Jt; %added 2/25/20
%     aedat2.importParams.filePath = aedat.importParams.filePath;
%     aedat2.data.frame.apsIntGood = aedat.data.frame.apsIntGood;
%     aedat2.data.frame.diffImTime = aedat.data.frame.diffImTime;
%     aedat2.data.frame.Ix = aedat.data.frame.Gx;
%     aedat2.data.frame.Iy = aedat.data.frame.Gy;
%     aedat2.data.frame.Jx = aedat.data.frame.Jx;
%     aedat2.data.frame.Jy = aedat.data.frame.Jy;
%     aedat2.data.frame.Vx = aedat.data.frame.Vx;
%     aedat2.data.frame.Vy = aedat.data.frame.Vy;
%     aedat2.data.frame.Jt = aedat.data.frame.Jt;
%     aedat2.data.frame.I = cell2mat(reshape(aedat.data.frame.samples,1,1,[])).*repmat(inputVar.fpn.slope,1,1,aedat.data.frame.numDiffImages);
% 
%     aedat = aedat2;
%     clear aedat2
    
%     inputVar.maxNumSamples = 20e3; %Only sample up to this many events each for pos and neg polarities combined
%     inputVar.nonCausal = false; %
%     inputVar.neighborhood = 6; %added 2/25/20

%     %added 3/12/2020
%     inputVar.maxNumSamples = 20; %Only sample up to this many chips per frame
%     inputVar.nonCausal = false; %
%     inputVar.neighborhood = 12; %
    
    %     tW = 0:100:max(abs(aedat.data.polarity.frameTimeDelta));
    %     for loop2=1:numel(tW)
    %         posAcc(loop2) = mean(aedat.data.polarity.Prob(aedat.data.polarity.polarity==1 & (abs(aedat.data.polarity.frameTimeDelta)<tW(loop2)))>0);
    %         negAcc(loop2) = mean(aedat.data.polarity.Prob(aedat.data.polarity.polarity==0 & (abs(aedat.data.polarity.frameTimeDelta)<tW(loop2)))<0);
    %     end
    %     clf
    %     hold on
    %     plot(tW,posAcc,'b')
    %     plot(tW,negAcc,'r')
    
    % makeAnimatedGif(aedat)
    
    % Process event points into feature vectors
    % [X, Y] = events2Feat(aedat, inputVar);
    %     [X, Y, samples] = events2FeatAll(aedat, inputVar);
    
    inputVar.maxNumSamples = 10000;
    inputVar.neighborhood = 2;
    inputVar.depth = 9;
    
    [X, Y, samples, ~] = events2FeatML(aedat, inputVar); %edited 13FEB2020

%     [X, Y, samples, pol] = events2FeatExp(aedat, inputVar);
%     tic;
% %     [X, Y, samples] = events2FeatExpSurfChipFast(aedat, inputVar);%added 3/12/2020
%     profile on;
%     [X, Y, samples] = events2FeatML_Fastv2(aedat, inputVar);%added 4/1/2020
%     profile viewer
%     toc
    
    save([outDir fn '_labels.mat'],'X','Y','samples','-v7.3')
%     save([outDir fn '_labels.mat'],'Xexp','Xhist','Y','samples','-v7.3')
    
end

