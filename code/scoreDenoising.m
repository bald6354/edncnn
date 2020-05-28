%Script to generate denoising scores
clear, clc

% addpath('/home/wescomp/Dropbox/WesDocs/UD/Research/imuWork_updated/')
% addpath('/home/wescomp/Dropbox/WesDocs/UD/Research/IETS/')
mainDir = '/media/wescomp/WesDataDrive/UDVS_features1/'
cnnDir = '/home/wescomp/Dropbox/WesDocs/UD/ECE_595_Deep_Learning/ednn/'
dataDir = '/media/wescomp/WesDataDrive/UDVS_Labeled/'
% mainDir = 'D:\Wes\UDVS_features1\'
% cnnDir = 'D:\Wes\Dropbox\WesDocs\UD\ECE_595_Deep_Learning\ednn\'
% dataDir = 'D:\Wes\UDVS_Labeled\'

files = dir([mainDir '*.mat']);

justRunEDNnCNN = false;

vString = 'v7'
mkdir(vString)

% for grpID = [5 6 12 13 11 15 1 2 3 4 7 8 9 10 14 16]
%     for grpID = [9 16] %v9 (10,14 worse)
% for grpID = [2 3 5 6 12 13 11 15 1] %v8
% for grpID = [4 7 8 9 10 14] %v4
for grpID = [11] %v8 (check)
% for grpID = 1
    
    testSet = (grpID-1).*3 + 2;
    
    %Load a CNN and results
    load([cnnDir num2str(grpID) '_' vString '.mat'])
    
    %where are the .mat files
    labeledMatFileForTestData = [dataDir files(testSet).name(1:end-11) '.mat']
    
    load(labeledMatFileForTestData, 'aedat');
    load(labeledMatFileForTestData, 'inputVar');
    
    %Only score events where EPM can generate valid data
    validEventsWithinFrameIdx = ~isnan(YPred) & (aedat.data.polarity.duringAPS>0) & aedat.data.polarity.apsIntGood;
    
    %YPred==2 good event from edncnn
    
    %Filtered surface of active events
    k = 50000; %50msec in paper
    %Since only part of each files is evaluated just process that data
    a = find(~isnan(YPred),1,'first');
    b = find(~isnan(YPred),1,'last');
    
    st = aedat.data.polarity.timeStamp(a);
    et = aedat.data.polarity.timeStamp(b);
    frmsWithScores = find(aedat.data.frame.frameStart>st & aedat.data.frame.frameEnd<et);
    midFrame = round(median(frmsWithScores));
    eventTimeToFrame = abs(aedat.data.polarity.timeStamp - aedat.data.frame.timeStamp(midFrame));
    
    %if just making images make the window just a few frames
    if false
        a = find((aedat.data.polarity.closestFrame==(midFrame-1)),1,'first');
        b = find((aedat.data.polarity.closestFrame==(midFrame+1)),1,'last');
    end
    
    if ~justRunEDNnCNN
        
        if false
            %old 3/2/2020
%             %FSAE
%             %     old
%             %     passesFilteredSAE = runFSAE(aedat.data.polarity.x(a:b),aedat.data.polarity.y(a:b),aedat.data.polarity.timeStamp(a:b),k);
%             tou = 50e3;
%             FSAE = Denoise_fsae_updated(aedat.data.polarity.x(a:b),aedat.data.polarity.y(a:b),aedat.data.polarity.timeStamp(a:b),aedat.data.polarity.polarity(a:b),tou);
%             aedat.data.polarity.fsae = nan(size(aedat.data.polarity.x));
%             aedat.data.polarity.fsae(a:b) = ~FSAE;
%             
%             %BA filter
%             tou = 25e3;
%             %     jearIdx(aedat.data.polarity.x==1 | aedat.data.polarity.x==346 | aedat.data.polarity.y==1 | aedat.data.polarity.y==260) = false;
%             BA = Denoise_jEAR_updated(aedat.data.polarity.x(a:b),aedat.data.polarity.y(a:b),aedat.data.polarity.timeStamp(a:b),aedat.data.polarity.polarity(a:b),tou);
%             aedat.data.polarity.BA = nan(size(aedat.data.polarity.x));
%             aedat.data.polarity.BA(a:b) = ~BA;
%             
%             %nn filter
%             tou = 25e3;
%             tou2 = 1;
%             k = 3;
%             nn = Nearest_Neighbor(aedat.data.polarity.x(a:b),aedat.data.polarity.y(a:b),aedat.data.polarity.timeStamp(a:b),aedat.data.polarity.polarity(a:b),tou,tou2,k);
%             aedat.data.polarity.nn = nan(size(aedat.data.polarity.x));
%             aedat.data.polarity.nn(a:b) = nn;
%             
%             %nn2 filter
%             tou = 25e3;
%             tou2 = 2;
%             k = 3;
%             nn2 = Nearest_Neighbor(aedat.data.polarity.x(a:b),aedat.data.polarity.y(a:b),aedat.data.polarity.timeStamp(a:b),aedat.data.polarity.polarity(a:b),tou,tou2,k);
%             aedat.data.polarity.nn2 = nan(size(aedat.data.polarity.x));
%             aedat.data.polarity.nn2(a:b) = nn2;
%             
%             %density filter
%             tou = 25e3;
%             tou2 = 3;
%             density = Density_filter(aedat.data.polarity.x(a:b),aedat.data.polarity.y(a:b),aedat.data.polarity.timeStamp(a:b),aedat.data.polarity.polarity(a:b),tou,tou2);
%             aedat.data.polarity.density = nan(size(aedat.data.polarity.x));
%             aedat.data.polarity.density(a:b) = density;
        else
            %FSAE
            %     old
            %     passesFilteredSAE = runFSAE(aedat.data.polarity.x(a:b),aedat.data.polarity.y(a:b),aedat.data.polarity.timeStamp(a:b),k);
            tou = 50e3;
            FSAE = Denoise_fsae_updated(aedat.data.polarity.x(a:b),aedat.data.polarity.y(a:b),aedat.data.polarity.timeStamp(a:b),aedat.data.polarity.polarity(a:b),tou);
            aedat.data.polarity.fsae = nan(size(aedat.data.polarity.x));
            aedat.data.polarity.fsae(a:b) = ~FSAE;
            rpsData = [aedat.data.polarity.x(validEventsWithinFrameIdx) aedat.data.polarity.y(validEventsWithinFrameIdx) aedat.data.polarity.polarity(validEventsWithinFrameIdx) aedat.data.polarity.duringAPS(validEventsWithinFrameIdx) aedat.data.polarity.fsae(validEventsWithinFrameIdx)];
            [rpsData,sIndex] = sortrows(rpsData,5,'descend');
            [~,ia,~] = unique(rpsData(:,1:4),'first','rows');
            onePerLocation = find(validEventsWithinFrameIdx);
            onePerLocation = onePerLocation(sIndex(ia));
            validEventsWithinFrameIdxOnlyOne.fsae = false(size(validEventsWithinFrameIdx));
            validEventsWithinFrameIdxOnlyOne.fsae(onePerLocation) = true;
            
            %BA filter
            tou = 25e3;
            %     jearIdx(aedat.data.polarity.x==1 | aedat.data.polarity.x==346 | aedat.data.polarity.y==1 | aedat.data.polarity.y==260) = false;
            BA = Denoise_jEAR_updated(aedat.data.polarity.x(a:b),aedat.data.polarity.y(a:b),aedat.data.polarity.timeStamp(a:b),aedat.data.polarity.polarity(a:b),tou);
            aedat.data.polarity.BA = nan(size(aedat.data.polarity.x));
            aedat.data.polarity.BA(a:b) = ~BA;
            rpsData = [aedat.data.polarity.x(validEventsWithinFrameIdx) aedat.data.polarity.y(validEventsWithinFrameIdx) aedat.data.polarity.polarity(validEventsWithinFrameIdx) aedat.data.polarity.duringAPS(validEventsWithinFrameIdx) aedat.data.polarity.BA(validEventsWithinFrameIdx)];
            [rpsData,sIndex] = sortrows(rpsData,5,'descend');
            [~,ia,~] = unique(rpsData(:,1:4),'first','rows');
            onePerLocation = find(validEventsWithinFrameIdx);
            onePerLocation = onePerLocation(sIndex(ia));
            validEventsWithinFrameIdxOnlyOne.BA = false(size(validEventsWithinFrameIdx));
            validEventsWithinFrameIdxOnlyOne.BA(onePerLocation) = true;
            
            %nn filter
            tou = 25e3;
            tou2 = 1;
            k = 3;
            nn = Nearest_Neighbor(aedat.data.polarity.x(a:b),aedat.data.polarity.y(a:b),aedat.data.polarity.timeStamp(a:b),aedat.data.polarity.polarity(a:b),tou,tou2,k);
            aedat.data.polarity.nn = nan(size(aedat.data.polarity.x));
            aedat.data.polarity.nn(a:b) = nn;
            rpsData = [aedat.data.polarity.x(validEventsWithinFrameIdx) aedat.data.polarity.y(validEventsWithinFrameIdx) aedat.data.polarity.polarity(validEventsWithinFrameIdx) aedat.data.polarity.duringAPS(validEventsWithinFrameIdx) aedat.data.polarity.nn(validEventsWithinFrameIdx)];
            [rpsData,sIndex] = sortrows(rpsData,5,'descend');
            [~,ia,~] = unique(rpsData(:,1:4),'first','rows');
            onePerLocation = find(validEventsWithinFrameIdx);
            onePerLocation = onePerLocation(sIndex(ia));
            validEventsWithinFrameIdxOnlyOne.nn = false(size(validEventsWithinFrameIdx));
            validEventsWithinFrameIdxOnlyOne.nn(onePerLocation) = true;

            
            %nn2 filter
            tou = 25e3;
            tou2 = 2;
            k = 3;
            nn2 = Nearest_Neighbor(aedat.data.polarity.x(a:b),aedat.data.polarity.y(a:b),aedat.data.polarity.timeStamp(a:b),aedat.data.polarity.polarity(a:b),tou,tou2,k);
            aedat.data.polarity.nn2 = nan(size(aedat.data.polarity.x));
            aedat.data.polarity.nn2(a:b) = nn2;
            rpsData = [aedat.data.polarity.x(validEventsWithinFrameIdx) aedat.data.polarity.y(validEventsWithinFrameIdx) aedat.data.polarity.polarity(validEventsWithinFrameIdx) aedat.data.polarity.duringAPS(validEventsWithinFrameIdx) aedat.data.polarity.nn2(validEventsWithinFrameIdx)];
            [rpsData,sIndex] = sortrows(rpsData,5,'descend');
            [~,ia,~] = unique(rpsData(:,1:4),'first','rows');
            onePerLocation = find(validEventsWithinFrameIdx);
            onePerLocation = onePerLocation(sIndex(ia));
            validEventsWithinFrameIdxOnlyOne.nn2 = false(size(validEventsWithinFrameIdx));
            validEventsWithinFrameIdxOnlyOne.nn2(onePerLocation) = true;
            
            %density filter
            tou = 25e3;
            tou2 = 3;
            density = Density_filter(aedat.data.polarity.x(a:b),aedat.data.polarity.y(a:b),aedat.data.polarity.timeStamp(a:b),aedat.data.polarity.polarity(a:b),tou,tou2);
            aedat.data.polarity.density = nan(size(aedat.data.polarity.x));
            aedat.data.polarity.density(a:b) = density;
            rpsData = [aedat.data.polarity.x(validEventsWithinFrameIdx) aedat.data.polarity.y(validEventsWithinFrameIdx) aedat.data.polarity.polarity(validEventsWithinFrameIdx) aedat.data.polarity.duringAPS(validEventsWithinFrameIdx) aedat.data.polarity.density(validEventsWithinFrameIdx)];
            [rpsData,sIndex] = sortrows(rpsData,5,'descend');
            [~,ia,~] = unique(rpsData(:,1:4),'first','rows');
            onePerLocation = find(validEventsWithinFrameIdx);
            onePerLocation = onePerLocation(sIndex(ia));
            validEventsWithinFrameIdxOnlyOne.density = false(size(validEventsWithinFrameIdx));
            validEventsWithinFrameIdxOnlyOne.density(onePerLocation) = true;
            
            %IE
            rpsData = [aedat.data.polarity.x(validEventsWithinFrameIdx) aedat.data.polarity.y(validEventsWithinFrameIdx) aedat.data.polarity.polarity(validEventsWithinFrameIdx) aedat.data.polarity.duringAPS(validEventsWithinFrameIdx) aedat.data.polarity.IE(validEventsWithinFrameIdx)];
            [rpsData,sIndex] = sortrows(rpsData,5,'descend');
            [~,ia,~] = unique(rpsData(:,1:4),'first','rows');
            onePerLocation = find(validEventsWithinFrameIdx);
            onePerLocation = onePerLocation(sIndex(ia));
            validEventsWithinFrameIdxOnlyOne.IE = false(size(validEventsWithinFrameIdx));
            validEventsWithinFrameIdxOnlyOne.IE(onePerLocation) = true;
            
            %IE+TE
            rpsData = [aedat.data.polarity.x(validEventsWithinFrameIdx) aedat.data.polarity.y(validEventsWithinFrameIdx) aedat.data.polarity.polarity(validEventsWithinFrameIdx) aedat.data.polarity.duringAPS(validEventsWithinFrameIdx) (aedat.data.polarity.IE(validEventsWithinFrameIdx) | aedat.data.polarity.TE(validEventsWithinFrameIdx))];
            [rpsData,sIndex] = sortrows(rpsData,5,'descend');
            [~,ia,~] = unique(rpsData(:,1:4),'first','rows');
            onePerLocation = find(validEventsWithinFrameIdx);
            onePerLocation = onePerLocation(sIndex(ia));
            validEventsWithinFrameIdxOnlyOne.IETE = false(size(validEventsWithinFrameIdx));
            validEventsWithinFrameIdxOnlyOne.IETE(onePerLocation) = true;

        end
    end
        
    if false
        %old
%         N = sum(validEventsWithinFrameIdx & ~isnan(YPred));
%         cnt(grpID) = N;
%         logOptimalScore(grpID) = 1/N.*(sum(log(aedat.data.polarity.Prob(validEventsWithinFrameIdx & aedat.data.polarity.Prob>0.5))) + ...
%             sum(log(1-aedat.data.polarity.Prob(validEventsWithinFrameIdx & aedat.data.polarity.Prob<=0.5))));
%         
%         logAllScore(grpID) = 1/N.*sum(log(max(1-aedat.data.polarity.Prob(validEventsWithinFrameIdx),realmin)));
%         disp(['ALL: ' num2str(logAllScore(grpID)-logOptimalScore(grpID))])
    else
        %update 3/2/2020 - need to group all events at the same location
        %and frame to the same label
        rpsData = [aedat.data.polarity.x(validEventsWithinFrameIdx) aedat.data.polarity.y(validEventsWithinFrameIdx) aedat.data.polarity.polarity(validEventsWithinFrameIdx) aedat.data.polarity.duringAPS(validEventsWithinFrameIdx) YPred(validEventsWithinFrameIdx)];
        [rpsData,sIndex] = sortrows(rpsData,5,'descend');
        [~,ia,~] = unique(rpsData(:,1:4),'first','rows');
        onePerLocation = find(validEventsWithinFrameIdx);
        onePerLocation = onePerLocation(sIndex(ia));
        validEventsWithinFrameIdxOnlyOne.edn = false(size(validEventsWithinFrameIdx));
        validEventsWithinFrameIdxOnlyOne.edn(onePerLocation) = true;
        
        %         ind = sub2ind(size(aedat.data.frame.apsIntGood),rpsData(:,2),rpsData(:,1),rpsData(:,3));
        %         wes = accumarray(ind,double(rpsData(:,4)), [], @min, 0, true);
        
        N = numel(onePerLocation);
        cnt(grpID) = N;
        logOptimalScore(grpID) = 1/N.*(sum(log(aedat.data.polarity.Prob(validEventsWithinFrameIdxOnlyOne.edn & aedat.data.polarity.Prob>0.5))) + ...
            sum(log(1-aedat.data.polarity.Prob(validEventsWithinFrameIdxOnlyOne.edn & aedat.data.polarity.Prob<=0.5))));
        
        logAllScore(grpID) = 1/N.*sum(log(max(1-aedat.data.polarity.Prob(validEventsWithinFrameIdxOnlyOne.edn),realmin)));
        disp(['ALL: ' num2str(logAllScore(grpID)-logOptimalScore(grpID))])
    end
        
    if ~justRunEDNnCNN
        if false
            %old - 3/2/2020
%             logFSAEScore(grpID) = 1/N.*(sum(log(max(aedat.data.polarity.Prob(validEventsWithinFrameIdx & (aedat.data.polarity.fsae==1)),realmin))) + ...
%                 sum(log(max(1-aedat.data.polarity.Prob(validEventsWithinFrameIdx & (aedat.data.polarity.fsae==0)),realmin))));
%             disp(['FSAE: ' num2str(logFSAEScore(grpID)-logOptimalScore(grpID))])
%             
%             logIEScore(grpID) = 1/N.*(sum(log(max(aedat.data.polarity.Prob(validEventsWithinFrameIdx & aedat.data.polarity.IE),realmin))) + ...
%                 sum(log(max(1-aedat.data.polarity.Prob(validEventsWithinFrameIdx & ~aedat.data.polarity.IE),realmin))));
%             disp(['IE: ' num2str(logIEScore(grpID)-logOptimalScore(grpID))])
%             
%             logIETEScore(grpID) = 1/N.*(sum(log(max(aedat.data.polarity.Prob(validEventsWithinFrameIdx & (aedat.data.polarity.IE | aedat.data.polarity.TE)),realmin))) + ...
%                 sum(log(max(1-aedat.data.polarity.Prob(validEventsWithinFrameIdx & ~(aedat.data.polarity.IE | aedat.data.polarity.TE)),realmin))));
%             disp(['IE/TE: ' num2str(logIETEScore(grpID)-logOptimalScore(grpID))])
%             
%             logBAScore(grpID) = 1/N.*(sum(log(max(aedat.data.polarity.Prob(validEventsWithinFrameIdx & (aedat.data.polarity.BA==1)),realmin))) + ...
%                 sum(log(max(1-aedat.data.polarity.Prob(validEventsWithinFrameIdx & (aedat.data.polarity.BA==0)),realmin))));
%             disp(['BA: ' num2str(logBAScore(grpID)-logOptimalScore(grpID))])
%             
%             logNNScore(grpID) = 1/N.*(sum(log(max(aedat.data.polarity.Prob(validEventsWithinFrameIdx & (aedat.data.polarity.nn==1)),realmin))) + ...
%                 sum(log(max(1-aedat.data.polarity.Prob(validEventsWithinFrameIdx & (aedat.data.polarity.nn==0)),realmin))));
%             disp(['NN: ' num2str(logNNScore(grpID)-logOptimalScore(grpID))])
%             
%             logNN2Score(grpID) = 1/N.*(sum(log(max(aedat.data.polarity.Prob(validEventsWithinFrameIdx & (aedat.data.polarity.nn2==1)),realmin))) + ...
%                 sum(log(max(1-aedat.data.polarity.Prob(validEventsWithinFrameIdx & (aedat.data.polarity.nn2==0)),realmin))));
%             disp(['NN2: ' num2str(logNN2Score(grpID)-logOptimalScore(grpID))])
%             
%             logSTScore(grpID) = 1/N.*(sum(log(max(aedat.data.polarity.Prob(validEventsWithinFrameIdx & (aedat.data.polarity.density==1)),realmin))) + ...
%                 sum(log(max(1-aedat.data.polarity.Prob(validEventsWithinFrameIdx & (aedat.data.polarity.density==0)),realmin))));
%             disp(['Density: ' num2str(logSTScore(grpID)-logOptimalScore(grpID))])
        else
            logFSAEScore(grpID) = 1/N.*(sum(log(max(aedat.data.polarity.Prob(validEventsWithinFrameIdxOnlyOne.fsae & (aedat.data.polarity.fsae==1)),realmin))) + ...
                sum(log(max(1-aedat.data.polarity.Prob(validEventsWithinFrameIdxOnlyOne.fsae & (aedat.data.polarity.fsae==0)),realmin))));
            disp(['FSAE: ' num2str(logFSAEScore(grpID)-logOptimalScore(grpID))])
            
            logIEScore(grpID) = 1/N.*(sum(log(max(aedat.data.polarity.Prob(validEventsWithinFrameIdxOnlyOne.IE & aedat.data.polarity.IE),realmin))) + ...
                sum(log(max(1-aedat.data.polarity.Prob(validEventsWithinFrameIdxOnlyOne.IE & ~aedat.data.polarity.IE),realmin))));
            disp(['IE: ' num2str(logIEScore(grpID)-logOptimalScore(grpID))])
            
            logIETEScore(grpID) = 1/N.*(sum(log(max(aedat.data.polarity.Prob(validEventsWithinFrameIdxOnlyOne.IETE & (aedat.data.polarity.IE | aedat.data.polarity.TE)),realmin))) + ...
                sum(log(max(1-aedat.data.polarity.Prob(validEventsWithinFrameIdxOnlyOne.IETE & ~(aedat.data.polarity.IE | aedat.data.polarity.TE)),realmin))));
            disp(['IE/TE: ' num2str(logIETEScore(grpID)-logOptimalScore(grpID))])
            
            logBAScore(grpID) = 1/N.*(sum(log(max(aedat.data.polarity.Prob(validEventsWithinFrameIdxOnlyOne.BA & (aedat.data.polarity.BA==1)),realmin))) + ...
                sum(log(max(1-aedat.data.polarity.Prob(validEventsWithinFrameIdxOnlyOne.BA & (aedat.data.polarity.BA==0)),realmin))));
            disp(['BA: ' num2str(logBAScore(grpID)-logOptimalScore(grpID))])
            
            logNNScore(grpID) = 1/N.*(sum(log(max(aedat.data.polarity.Prob(validEventsWithinFrameIdxOnlyOne.nn & (aedat.data.polarity.nn==1)),realmin))) + ...
                sum(log(max(1-aedat.data.polarity.Prob(validEventsWithinFrameIdxOnlyOne.nn & (aedat.data.polarity.nn==0)),realmin))));
            disp(['NN: ' num2str(logNNScore(grpID)-logOptimalScore(grpID))])
            
            logNN2Score(grpID) = 1/N.*(sum(log(max(aedat.data.polarity.Prob(validEventsWithinFrameIdxOnlyOne.nn2 & (aedat.data.polarity.nn2==1)),realmin))) + ...
                sum(log(max(1-aedat.data.polarity.Prob(validEventsWithinFrameIdxOnlyOne.nn2 & (aedat.data.polarity.nn2==0)),realmin))));
            disp(['NN2: ' num2str(logNN2Score(grpID)-logOptimalScore(grpID))])
            
            logSTScore(grpID) = 1/N.*(sum(log(max(aedat.data.polarity.Prob(validEventsWithinFrameIdxOnlyOne.density & (aedat.data.polarity.density==1)),realmin))) + ...
                sum(log(max(1-aedat.data.polarity.Prob(validEventsWithinFrameIdxOnlyOne.density & (aedat.data.polarity.density==0)),realmin))));
            disp(['Density: ' num2str(logSTScore(grpID)-logOptimalScore(grpID))])
        end
    end
    
    if false
        %old 3/2/2020
        logEDNScore(grpID) = 1/N.*(sum(log(max(aedat.data.polarity.Prob(validEventsWithinFrameIdx & YPred==2),realmin))) + ...
            sum(log(max(1-aedat.data.polarity.Prob(validEventsWithinFrameIdx & YPred==1),realmin))));
        disp(['EDn: ' num2str(logEDNScore(grpID)-logOptimalScore(grpID))])
    else
        logEDNScore(grpID) = 1/N.*(sum(log(max(aedat.data.polarity.Prob(validEventsWithinFrameIdxOnlyOne.edn & YPred==2),realmin))) + ...
            sum(log(max(1-aedat.data.polarity.Prob(validEventsWithinFrameIdxOnlyOne.edn & YPred==1),realmin))));
        disp(['EDn: ' num2str(logEDNScore(grpID)-logOptimalScore(grpID))])
    end
    
    end
    
    %write out multiple images to ensure we get one with events (i.e.
    %motion)
    stepSize = round(range(frmsWithScores)/4);
    
    for midFrameLoop = [midFrame-stepSize:stepSize:midFrame+stepSize]
        
        eventTimeToFrame = abs(aedat.data.polarity.timeStamp - aedat.data.frame.timeStamp(midFrameLoop));

        %Write out imagesc
        im = uint8(aedat.data.frame.samples{midFrameLoop});
        im = imadjust(im,stretchlim(im,.05));
        
        %EPM
        epm = aedat.data.frame.Jt(:,:,midFrameLoop);
        gamma = zeros(size(epm));
        gamma(epm>0) = aedat.cameraSetup.estGammaP;
        gamma(epm<=0) = aedat.cameraSetup.estGammaN;
        epm = epm./gamma;
        clf
        set(gcf, 'Position',  [100, 100, 792, 620])
        imagesc(epm,[-1 1])
        colormap gray
        hold on
        axis image
        view(0,-90)
        axis([14 335 14 249])
        drawnow
        frame = getframe(gca);
        imwrite(frame.cdata, [vString filesep num2str(grpID) '_' num2str(midFrameLoop) '_epm.png']);
        
        %EPM - no polarity
        epm = aedat.data.frame.Jt(:,:,midFrameLoop);
        gamma = zeros(size(epm));
        gamma(epm>0) = aedat.cameraSetup.estGammaP;
        gamma(epm<=0) = aedat.cameraSetup.estGammaN;
        epm = abs(epm./gamma);
        clf
        set(gcf, 'Position',  [100, 100, 792, 620])
        imagesc(epm,[0 1])
        colormap gray
        hold on
        axis image
        view(0,-90)
        axis([14 335 14 249])
        drawnow
        frame = getframe(gca);
        imwrite(frame.cdata, [vString filesep num2str(grpID) '_' num2str(midFrameLoop) '_epmGray.png']);
        
        clf
        imagesc(im,[0 255])
        colormap gray
        hold on
        axis image
        view(0,-90)
        axis([14 335 14 249])
        drawnow
        frame = getframe(gca);
        imwrite(frame.cdata, [vString filesep num2str(grpID) '_' num2str(midFrameLoop) '_image.png']);
        
        clf
        imagesc(im,[0 255])
        colormap gray
        hold on
        axis image
        tmp = sort(eventTimeToFrame);
        timeDelta = tmp(20e3); %capture 10,000 events closest to frame for display
        idx = eventTimeToFrame<=timeDelta;
        scatter(aedat.data.polarity.x(idx)+1,aedat.data.polarity.y(idx)+1,10,'y','filled')
        %     idx = aedat.data.polarity.closestFrame == midFrame & aedat.data.polarity.polarity<=0;
        %     scatter(aedat.data.polarity.x(idx)+1,aedat.data.polarity.y(idx)+1,10,'r','filled')
        view(0,-90)
        axis([14 335 14 249])
        drawnow
        frame = getframe(gca);
        imwrite(frame.cdata, [vString filesep num2str(grpID) '_' num2str(midFrameLoop) '_noisy.png']);
        
        if ~justRunEDNnCNN
            clf
            imagesc(im,[0 255])
            colormap gray
            hold on
            axis image
            isEvent = aedat.data.polarity.fsae==1;
            %     idx = (aedat.data.polarity.closestFrame == midFrame) & ~aedat.data.polarity.IE;
            %     scatter(aedat.data.polarity.x(idx)+1,aedat.data.polarity.y(idx)+1,10,'r','filled')
            idx = isEvent & (eventTimeToFrame<=timeDelta);
            scatter(aedat.data.polarity.x(idx)+1,aedat.data.polarity.y(idx)+1,10,'y','filled')
            view(0,-90)
            axis([14 335 14 249])
            drawnow
            frame = getframe(gca);
            imwrite(frame.cdata, [vString filesep num2str(grpID) '_' num2str(midFrameLoop) '_FSAE.png']);
            
            clf
            imagesc(im,[0 255])
            colormap gray
            hold on
            axis image
            isEvent = aedat.data.polarity.IE;
            %     idx = (aedat.data.polarity.closestFrame == midFrame) & ~aedat.data.polarity.IE;
            %     scatter(aedat.data.polarity.x(idx)+1,aedat.data.polarity.y(idx)+1,10,'r','filled')
            idx = isEvent & (eventTimeToFrame<=timeDelta);
            scatter(aedat.data.polarity.x(idx)+1,aedat.data.polarity.y(idx)+1,10,'y','filled')
            view(0,-90)
            axis([14 335 14 249])
            drawnow
            frame = getframe(gca);
            imwrite(frame.cdata, [vString filesep num2str(grpID) '_' num2str(midFrameLoop) '_IE.png']);
            
            clf
            imagesc(im,[0 255])
            colormap gray
            hold on
            axis image
            isEvent = (aedat.data.polarity.IE | aedat.data.polarity.TE);
            %     idx = (aedat.data.polarity.closestFrame == midFrame) & ~(aedat.data.polarity.IE | aedat.data.polarity.TE);
            %     scatter(aedat.data.polarity.x(idx)+1,aedat.data.polarity.y(idx)+1,10,'r','filled')
            idx = isEvent & (eventTimeToFrame<=timeDelta);
            scatter(aedat.data.polarity.x(idx)+1,aedat.data.polarity.y(idx)+1,10,'y','filled')
            view(0,-90)
            axis([14 335 14 249])
            drawnow
            frame = getframe(gca);
            imwrite(frame.cdata, [vString filesep num2str(grpID) '_' num2str(midFrameLoop) '_IETE.png']);
            
            clf
            imagesc(im,[0 255])
            colormap gray
            hold on
            axis image
            isEvent = aedat.data.polarity.BA==1;
            %     idx = (aedat.data.polarity.closestFrame == midFrame) & ~(aedat.data.polarity.IE | aedat.data.polarity.TE);
            %     scatter(aedat.data.polarity.x(idx)+1,aedat.data.polarity.y(idx)+1,10,'r','filled')
            idx = isEvent & (eventTimeToFrame<=timeDelta);
            scatter(aedat.data.polarity.x(idx)+1,aedat.data.polarity.y(idx)+1,10,'y','filled')
            view(0,-90)
            axis([14 335 14 249])
            drawnow
            frame = getframe(gca);
            imwrite(frame.cdata, [vString filesep num2str(grpID) '_' num2str(midFrameLoop) '_BA.png']);
            
            clf
            imagesc(im,[0 255])
            colormap gray
            hold on
            axis image
            isEvent = aedat.data.polarity.nn==1;
            %     idx = (aedat.data.polarity.closestFrame == midFrame) & ~(aedat.data.polarity.IE | aedat.data.polarity.TE);
            %     scatter(aedat.data.polarity.x(idx)+1,aedat.data.polarity.y(idx)+1,10,'r','filled')
            idx = isEvent & (eventTimeToFrame<=timeDelta);
            scatter(aedat.data.polarity.x(idx)+1,aedat.data.polarity.y(idx)+1,10,'y','filled')
            view(0,-90)
            axis([14 335 14 249])
            drawnow
            frame = getframe(gca);
            imwrite(frame.cdata, [vString filesep num2str(grpID) '_' num2str(midFrameLoop) '_NN.png']);
            
            clf
            imagesc(im,[0 255])
            colormap gray
            hold on
            axis image
            isEvent = aedat.data.polarity.nn2==1;
            %     idx = (aedat.data.polarity.closestFrame == midFrame) & ~(aedat.data.polarity.IE | aedat.data.polarity.TE);
            %     scatter(aedat.data.polarity.x(idx)+1,aedat.data.polarity.y(idx)+1,10,'r','filled')
            idx = isEvent & (eventTimeToFrame<=timeDelta);
            scatter(aedat.data.polarity.x(idx)+1,aedat.data.polarity.y(idx)+1,10,'y','filled')
            view(0,-90)
            axis([14 335 14 249])
            drawnow
            frame = getframe(gca);
            imwrite(frame.cdata, [vString filesep num2str(grpID) '_' num2str(midFrameLoop) '_NN2.png']);
            
            clf
            imagesc(im,[0 255])
            colormap gray
            hold on
            axis image
            isEvent = aedat.data.polarity.density==1;
            %     idx = (aedat.data.polarity.closestFrame == midFrame) & ~(aedat.data.polarity.IE | aedat.data.polarity.TE);
            %     scatter(aedat.data.polarity.x(idx)+1,aedat.data.polarity.y(idx)+1,10,'r','filled')
            idx = isEvent & (eventTimeToFrame<=timeDelta);
            scatter(aedat.data.polarity.x(idx)+1,aedat.data.polarity.y(idx)+1,10,'y','filled')
            view(0,-90)
            axis([14 335 14 249])
            drawnow
            frame = getframe(gca);
            imwrite(frame.cdata, [vString filesep num2str(grpID) '_' num2str(midFrameLoop) '_Density.png']);
        end
        
        clf
        imagesc(im,[0 255])
        colormap gray
        hold on
        axis image
        isEvent = YPred==2;
        %     idx = (aedat.data.polarity.closestFrame == midFrame) & YPred==1;
        %     scatter(aedat.data.polarity.x(idx)+1,aedat.data.polarity.y(idx)+1,10,'r','filled')
        idx = isEvent & (eventTimeToFrame<=timeDelta);
        scatter(aedat.data.polarity.x(idx)+1,aedat.data.polarity.y(idx)+1,10,'y','filled')
        view(0,-90)
        axis([14 335 14 249])
        drawnow
        frame = getframe(gca);
        imwrite(frame.cdata, [vString filesep num2str(grpID) '_' num2str(midFrameLoop) '_EDnCNN.png']);
        
    end
    
end

save([vString filesep 'scores2.mat'],'-regexp', 'log.*')
