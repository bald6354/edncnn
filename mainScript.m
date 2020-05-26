% Script to denoise data from event camera using CNN
% R. Wes Baldwin
% University of Dayton
% August 2019

% Please cite the CVPR 2020 paper: 
%  "Event Probability Mask (EPM) and Event Denoising Convolutional 
%   Neural Network (EDnCNN) for Neuromorphic Cameras"

% DVSNOISE20 data can be downloaded from:
% (https://sites.google.com/a/udayton.edu/issl/software/dataset)

% This script assumes you have the data in MAT format (2_mat.zip) from
% DVSNOISE20 site. If you need to convert your data use one of these two
% methods...
% 1. Convert AEDAT data into a MAT file use AEDAT-TOOLS
% (https://gitlab.com/inivation/AedatTools)
% 2. Convert AEDAT4 data into a MAT file use aedat4tomat
% (https://github.com/bald6354/aedat4tomat)

% The folder "camera" contains the camera parameters to remove lens distortion
% and fixed pattern noise for the DAVIS 346 camera we used for the
% DVSNOISE20 dataset. You can generate your own distortion correction for a
% different camera using the built-in MATLAB calibration application.

%% Setup and Variables
clear, clc, close all
set(0,'DefaultFigureWindowStyle','docked')

% Path to data files in MAT format
mainDir = '/media/wescomp/WesDataDrive/2_mat/'

% Output directory - results go here
outDir = '/media/wescomp/WesDataDrive/edncnn_output/'

% Settings
inputVar.depth = 2; %feature depth per polarity (k in paper)
inputVar.neighborhood = 12; %feature neighborhood (m in paper)(0=1x1, 1=3x3, 2=5x5, etc.)
inputVar.maxNumSamples = 100e3; %Only sample up to this many events each for pos and neg polarities combined
inputVar.waitBuffer = 2; %time in seconds to wait before sampling an event - early events have no history & dvs tends to drop the feed briefly in the first second or so
inputVar.minTime = 150; %any amount less than 150 microseconds can be ignored (helps with log scaling)
inputVar.maxTime = 5e6; %any amount greater than 5 seconds can be ignored (put data on fixed output size)
inputVar.maxProb = 1; %"probability" score capped at this num
inputVar.nonCausal = false; %if true, double feature size by creating surface both back in time AND forward in time
inputVar.removeGyroBias = 0.5; %use the first 0.5 seconds of data to remove a bias in the gyro

% Camera/Lens details - DAVIS346
inputVar.focalLength = 4; %mm (4.5=240C, 4=346(wide), 12=346(zoom)
inputVar.Wx = 6.4;  %width of focal plane in mm
inputVar.Wy = 4.8;  %height of focal plane in mm
load('camera/fpn_346.mat')
inputVar.fpn = fpn; %fixed pattern noise
clear fpn


%% Gather a list of files

files = dir([mainDir '*.mat']);

if ~exist(outDir, 'dir')
    mkdir(outDir)
end
    
%How many samples to randomly grab from test dataset for validation
numValidSamples = 2e5;

%% Process each file
processingErrors = false(numel(files),1);
% for fLoop = 1:numel(files)

for fLoop = 1:numel(files)
    
    close all
    
    file = [mainDir files(fLoop).name]
    [fp,fn,fe] = fileparts(file);

    % Read in aedat data (Requires events, frames, and IMU data)
    try
        if strcmp(fe,'.aedat')
            aedat = loadAedatWithAttributes(file);
        else
            %aedat4 converted to .mat via aedat4tomat.py
            aedat = loadMatWithAttributes(file);
        end
    catch
        processingErrors(fLoop) = true;
        continue
    end

%     aedat.cameraSetup.focalLength = inputVar.focalLength;
       
    %labelTimeThresh = 1e-2;  %Consider only events within this time window of APS frame (sec)
        
    aedat = eventTiming(aedat, inputVar);

    %Match each event to an APS intensity
    aps = reshape(cell2mat(aedat.data.frame.samples),aedat.data.frame.size(1),aedat.data.frame.size(2),[]);
    aedat.data.frame.sizeCube = size(aps);
    apsIdx = sub2ind(size(aps),aedat.data.polarity.y,aedat.data.polarity.x,aedat.data.polarity.closestFrame);
    aedat.data.polarity.apsIntensity = aps(apsIdx);

    %Is the aps data near zero or saturation
    minAps = 5;
    maxAps = 250;
    apsIntGood = (aps>=minAps) & (aps<=maxAps);
    for apsLoop = 1:size(aps,3)
        apsIntGood(:,:,apsLoop) = imdilate(medfilt2(apsIntGood(:,:,apsLoop)),ones(3));
    end
    aedat.data.polarity.apsIntGood = apsIntGood(apsIdx);
    aedat.data.frame.apsIntGood = apsIntGood;
    clear aps

    %Time window around APS frame to use to estimate sensor thresholds
    %(larger=more total measurements, smaller=more accurate measurements)
    %     tau = 1e4;
    
    timeBetweenFrames = diff(cell2mat(reshape(aedat.data.frame.diffImTime,1,1,numel(aedat.data.frame.diffImTime))),1,3);
    estFrameRate = 1e6./(median(timeBetweenFrames(:)));
    clear timeBetweenFrames
    %     tau = aedat.cameraSetup.estIntegrationTime; %Most accurate during APS intergration window (BUT dont use because gamma estimate gets bad near zero!!!)
%     tau = 1/estFrameRate; %Gamma estimate is best when we use ~1/framerate
    inputVar.tau = aedat.cameraSetup.estIntegrationTime;
    
    %Label inceptive events
%     multiTriggerWindow = 50e3;
    multiTriggerWindow = 20e3;
    aedat = IE(aedat, multiTriggerWindow);
    
% 
%     offSets = 0:5:255;
%     [offsetFitP3, offsetFitN3] = deal(nan(numel(offSets),1));
%     [gammaP3, gammaN3] = deal(nan(numel(offSets),1));
%     for oLoop = 1:numel(offSets)
%         
%         inputVar.intensityOffset = offSets(oLoop);
%         
%         % Using APS/IMU calculate a temporal change for each frame/event
%         aedat = assignJt2Events(aedat, inputVar);
%         
%         aedat = calcSensorThresholdViaMaxLike(aedat, tau);
%         
%         gammaP3(oLoop) = aedat.cameraSetup.estGammaP;
%         gammaN3(oLoop) = aedat.cameraSetup.estGammaN;
%         
%         offsetFitP3(oLoop) = aedat.cameraSetup.estGammaP_val;
%         offsetFitN3(oLoop) = aedat.cameraSetup.estGammaN_val;
%         
%         clf
%         subplot(211)
%         title('cost')
%         hold on
%         yyaxis left
%         plot(offSets(1:oLoop),offsetFitP3(1:oLoop),'.-')
%         yyaxis right
%         plot(offSets(1:oLoop),offsetFitN3(1:oLoop),'.-')
%         legend('pos','neg')
%         
%         subplot(212)
%         title('gamma')
%         hold on
%         yyaxis left
%         plot(offSets(1:oLoop),gammaP3(1:oLoop),'.-')
%         yyaxis right
%         plot(offSets(1:oLoop),gammaN3(1:oLoop),'.-')
%         legend('pos','neg')
% 
%         pause(.1)
%         
%     end
    
    %New Method
    [eP, eN, eEmpty] = deal(zeros(aedat.data.frame.sizeCube));
    
    for loop = 1:aedat.data.frame.numDiffImages
%         eventsWithinTauOfFrameIdx = (aedat.data.polarity.closestFrame == loop) & ...
%             (abs(aedat.data.polarity.frameTimeDelta) <= (tau*1e6/2)) & ...
%             (aedat.data.polarity.apsIntensity > 10) & ...
%             (aedat.data.polarity.apsIntensity < 245);
%         eventsWithinTauOfFrameIdx = (aedat.data.polarity.closestFrame == loop) & ...
%             (abs(aedat.data.polarity.frameTimeDelta) <= (inputVar.tau*1e6/2)) & ...
%             (aedat.data.polarity.apsIntensity > 0) & ...
%             (aedat.data.polarity.apsIntensity < 255);
        validEventsWithinFrameIdx = (aedat.data.polarity.duringAPS == loop) & ...
            aedat.data.polarity.apsIntGood;
%         validEventsWithinFrameIdx = (aedat.data.polarity.duringAPS == loop) & ...
%             aedat.data.polarity.IE & aedat.data.polarity.apsIntGood;
        eventFreeFrameIdx = aedat.data.polarity.closestFrame == loop;

        ewtofIdxP = validEventsWithinFrameIdx & (aedat.data.polarity.polarity > 0);
        ewtofIdxN = validEventsWithinFrameIdx & (aedat.data.polarity.polarity <= 0);
        eP(:,:,loop) = 1==accumarray([aedat.data.polarity.y(ewtofIdxP) aedat.data.polarity.x(ewtofIdxP)], ...
            1, [aedat.data.frame.size], @max,0);
        eN(:,:,loop) = 1==accumarray([aedat.data.polarity.y(ewtofIdxN) aedat.data.polarity.x(ewtofIdxN)], ...
            1, [aedat.data.frame.size], @max,0);
        eEmpty(:,:,loop) = 0==accumarray([aedat.data.polarity.y(eventFreeFrameIdx) aedat.data.polarity.x(eventFreeFrameIdx)], ...
            1, [aedat.data.frame.size], @max,0);
    end
    aedat.data.frame.eP = eP .* aedat.data.frame.apsIntGood;
    aedat.data.frame.eN = eN .* aedat.data.frame.apsIntGood;
    aedat.data.frame.eEmpty = eEmpty .* aedat.data.frame.apsIntGood;

    clear eP eN eEmpty
    
% % % %     %Orig Method
% % % %     [eP, eN] = deal(zeros(aedat.data.frame.sizeCube));
% % % %     
% % % %     for loop = 1:aedat.data.frame.numDiffImages
% % % % %         eventsWithinTauOfFrameIdx = (aedat.data.polarity.closestFrame == loop) & ...
% % % % %             (abs(aedat.data.polarity.frameTimeDelta) <= (tau*1e6/2)) & ...
% % % % %             (aedat.data.polarity.apsIntensity > 10) & ...
% % % % %             (aedat.data.polarity.apsIntensity < 245);
% % % % %         eventsWithinTauOfFrameIdx = (aedat.data.polarity.closestFrame == loop) & ...
% % % % %             (abs(aedat.data.polarity.frameTimeDelta) <= (inputVar.tau*1e6/2)) & ...
% % % % %             (aedat.data.polarity.apsIntensity > 0) & ...
% % % % %             (aedat.data.polarity.apsIntensity < 255);
% % % %         eventsWithinTauOfFrameIdx = (aedat.data.polarity.closestFrame == loop) & ...
% % % %             (abs(aedat.data.polarity.frameTimeDelta) <= (inputVar.tau*1e6/2));
% % % % 
% % % %         ewtofIdxP = eventsWithinTauOfFrameIdx & (aedat.data.polarity.polarity > 0);
% % % %         ewtofIdxN = eventsWithinTauOfFrameIdx & (aedat.data.polarity.polarity <= 0);
% % % %         eP(:,:,loop) = 1==accumarray([aedat.data.polarity.y(ewtofIdxP) aedat.data.polarity.x(ewtofIdxP)], ...
% % % %             1, [aedat.data.frame.size], @max,0);
% % % %         eN(:,:,loop) = 1==accumarray([aedat.data.polarity.y(ewtofIdxN) aedat.data.polarity.x(ewtofIdxN)], ...
% % % %             1, [aedat.data.frame.size], @max,0);
% % % %     end
% % % %     aedat.data.frame.eP = eP;
% % % %     aedat.data.frame.eN = eN;
% % % % 
% % % %     clear eP eN

    %% Try using fminsearch with multiple variables
    tic
    
    %     fun4d = @(b) minOffsetAndGamma(b(1), b(2), b(3), b(4), aedat, inputVar);
    %     opts = optimset('PlotFcns','optimplotfval','MaxIter',5000,'MaxFunEvals',5000);%,'TolFun',1e3,'TolX',1e-2);,    fun2d = @(b) minOffsetAndGamma(b(1), b(2), b(3), b(4), aedat, inputVar);
    %     b_guess = [20 20 70 70]; %1=APS offset(pos), 2=APS offset(neg), 3=Gamma(pos), 4=Gamma(neg)
    %     b_min = fminsearch(fun4d, b_guess, opts);
    %     apsOffsetPHat = b_min(1)
    %     apsOffsetNHat = b_min(2)
    %     gammaPHat = b_min(3)
    %     gammaNHat = b_min(4)
    
    fun3d = @(b) minSingleOffsetAndGamma(b(1), b(2), b(3), aedat, inputVar);
    opts = optimset('PlotFcns','optimplotfval','MaxIter',100,'MaxFunEvals',5000,'TolFun',10,'TolX',.1);
    b_guess = [50 20 20]; %1=APS offset(pos), 2=APS offset(neg), 3=Gamma(pos), 4=Gamma(neg)
    [b_min,cost_fit,flag_fit,output_fit] = fminsearch(fun3d, b_guess, opts);
    inputVar.intensityOffset = b_min(1)
    aedat.cameraSetup.estGammaP = b_min(2)
    aedat.cameraSetup.estGammaN = b_min(3)
% 
%     fun3d = @(b) minSingleOffsetAndGammaFitting(b(1), b(2), b(3), aedat, inputVar);
%     opts = optimset('PlotFcns','optimplotfval','MaxIter',100,'MaxFunEvals',5000,'TolFun',10,'TolX',.1);
%     b_guess = [100 20 20]; %1=APS offset(pos), 2=APS offset(neg), 3=Gamma(pos), 4=Gamma(neg)
%     [b_min,cost_fit,flag_fit,output_fit] = fminsearch(fun3d, b_guess, opts);
%     inputVar.intensityOffset = b_min(1)
%     aedat.cameraSetup.estGammaP = b_min(2)
%     aedat.cameraSetup.estGammaN = b_min(3)
    
    toc
    
%     aedat = halfNormProb(apsOffsetHat, aedat, inputVar);
    
    % Using APS/IMU calculate a temporal change for each frame/event
    aedat = assignJt2Events(aedat, inputVar);

    %% other
    
    if false
        %test linear relationship for tau/gamma
        t = logspace(log(.001)/log(10),log(.04)/log(10),100)
        for tLoop = 1:numel(t)
            aedat = calcSensorThresholdViaMaxLike(aedat, t(tLoop));
            gp(tLoop) = aedat.cameraSetup.estGammaP;
            gn(tLoop) = aedat.cameraSetup.estGammaN;
            clf
            hold on
            plot(t(1:tLoop),gp(1:tLoop).*t(1:tLoop))
            plot(t(1:tLoop),gn(1:tLoop).*t(1:tLoop))
            ylabel('est. epsilon')
            xlabel('tau')
            grid on
            title(strrep(fn,'_','\_'))
            pause(.01)
        end
    end

    if genRpt
        import mlreportgen.report.* 
        import mlreportgen.dom.* 
        rpt = Report([outDir fn], 'pdf');
        specs = struct();
        specs.Name = string(file);
        specs.Length = range(double(aedat.data.frame.timeStamp))/1e6;
        specs.EventCount = double(aedat.data.polarity.numEvents);
        specs.EventRate =  specs.EventCount/specs.Length;
        specs.PercentPos = 100.*mean(double(aedat.data.polarity.polarity));
        specs.FrameCount = double(aedat.data.frame.numDiffImages);
        specs.FrameRate = estFrameRate;
        specs.IntegrationTime = aedat.cameraSetup.estIntegrationTime;
        specs.DutyCycle = 100.*(specs.IntegrationTime*specs.FrameCount)/specs.Length;
        specs.PercentDuringAPS = 100.*mean(aedat.data.polarity.duringAPS~=0);
        specs.Cost = cost_fit;
        specs.FitFlag = flag_fit;
        specs.FitIterations = output_fit.iterations;
        specs.Offset = inputVar.intensityOffset;
        specs.EstGammaP = aedat.cameraSetup.estGammaP;
        specs.EstGammaN = aedat.cameraSetup.estGammaN;
        specs.EstEpsilonP = specs.EstGammaP * specs.IntegrationTime;
        specs.EstEpsilonN = specs.EstGammaN * specs.IntegrationTime;
        specs.TauUsed = inputVar.tau;
        specs.MultiTriggerWindow = multiTriggerWindow;
        specs.PercentIE = 100.*mean(aedat.data.polarity.IE);
        specs.PercentTE = 100.*mean(aedat.data.polarity.TE);
        specs.PercentIsolated = 100.*mean(~aedat.data.polarity.IE & ~aedat.data.polarity.TE);
        rpt = addTable(rpt,struct2table(specs));
    end
    
%     %If all data is collected around the same time with the same settings,
%     %lets use it all to generate the maximum likelyhood of epsilon before
%     %processing into probabilities
%     en(fLoop) = aedat.cameraSetup.estGammaN*tau;
%     ep(fLoop) = aedat.cameraSetup.estGammaP*tau;
%     
%     %Update the saved matfile with calculated data
%     save(file, 'aedat')
%     
% end
% 
% %Est threshold from all collects
% epsilonNHAT = median(en)
% epsilonPHAT = median(ep)
% 
% for fLoop = 1:numel(files)
%     
%     file = [mainDir files(fLoop).name]
%     
%     load(file)
    
%     aedat.cameraSetup.epsilonNHAT_AllFiles = epsilonNHAT;
%     aedat.cameraSetup.epsilonPHAT_AllFiles = epsilonPHAT;
%     
%     tau = aedat.cameraSetup.estIntegrationTime;
%     
%     gammaNHAT = epsilonNHAT/tau;
%     gammaPHAT = epsilonPHAT/tau;
%     
%     dirIdx = aedat.data.polarity.Jt>0;
%     aedat.data.polarity.Prob(dirIdx,1) = aedat.data.polarity.Jt(dirIdx)./gammaPHAT;
%     dirIdx = aedat.data.polarity.Jt<=0;
%     aedat.data.polarity.Prob(dirIdx,1) = -1.*aedat.data.polarity.Jt(dirIdx)./gammaNHAT;

    dirIdx = aedat.data.polarity.polarity>0;
    aedat.data.polarity.Prob(dirIdx,1) = aedat.data.polarity.Jt(dirIdx)./aedat.cameraSetup.estGammaP;
    dirIdx = aedat.data.polarity.polarity<=0;
    aedat.data.polarity.Prob(dirIdx,1) = -1.*aedat.data.polarity.Jt(dirIdx)./aedat.cameraSetup.estGammaN;

%     dirIdx = aedat.data.polarity.Jt>0;
%     aedat.data.polarity.Prob(dirIdx,1) = aedat.data.polarity.Jt(dirIdx)./aedat.cameraSetup.estGammaP;
%     dirIdx = aedat.data.polarity.Jt<=0;
%     aedat.data.polarity.Prob(dirIdx,1) = -1.*aedat.data.polarity.Jt(dirIdx)./aedat.cameraSetup.estGammaN;

    %Fix prob to range 0-inputVar.maxProb
    aedat.data.polarity.Prob(aedat.data.polarity.Prob<0) = 0;
    aedat.data.polarity.Prob(aedat.data.polarity.Prob>inputVar.maxProb) = inputVar.maxProb;
    
%     makeAnimatedGifMaxLikeIE(aedat, outDir)
%     makeAnimatedGifMaxLikeFalseIE(aedat, outDir)
    makeAnimatedGifMaxLike(aedat, outDir)
%     makeAnimatedGifMaxLikeFalse(aedat, outDir)
    
    if genRpt
        weCanLabelIdx = (aedat.data.polarity.duringAPS>0) & (aedat.data.polarity.apsIntGood);
%         & ~aedat.data.polarity.TE;
    
        clf
        histogram(aedat.data.polarity.Prob(weCanLabelIdx & aedat.data.polarity.polarity==1),100)
        hold on
        histogram(-1.*aedat.data.polarity.Prob(weCanLabelIdx & aedat.data.polarity.polarity==0),100)
        rpt = add2rpt(rpt, 'Histogram of Probabilities');
        
        clf
        hold on
        plot(aedat.data.imu6.gyroX)
        plot(aedat.data.imu6.gyroY)
        plot(aedat.data.imu6.gyroZ)
        legend('gyroX','gyroY','gyroZ')
        rpt = add2rpt(rpt, 'IMU');
        
        clf
        histogram(aedat.data.frame.Jt(:),[-100:100])
        rpt = add2rpt(rpt, 'Jt');
        
        clf
        histogram(aedat.data.frame.Jt(aedat.data.frame.eP(:)==1),[-100:100])
        rpt = add2rpt(rpt, 'Jt Pos Events');
        
%         clf
%         histogram(aedat.data.polarity.Prob(aedat.data.polarity.polarity==1)>=.5)
%         rpt = add2rpt(rpt, 'Pos Events > 50% Confidence');

        clf
        histogram(aedat.data.frame.Jt(aedat.data.frame.eN(:)==1),[-100:100])
        rpt = add2rpt(rpt, 'Jt Neg Events');

%         clf
%         histogram(aedat.data.polarity.Prob(aedat.data.polarity.polarity==0)>=.5)
%         rpt = add2rpt(rpt, 'Neg Events > 50% Confidence');

        clf
        histogram(aedat.data.frame.Jt(aedat.data.frame.eEmpty(:)==1),[-100:100])
        rpt = add2rpt(rpt, 'Jt No Events');
        
        clf
        histogram(aedat.data.polarity.frameTimeDelta)
        rpt = add2rpt(rpt, 'Frame Time Deltas');

        close(rpt)
        
    end
    
    save([outDir fn '.mat'],'-v7.3')
    
end

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
    [X, Y, samples] = events2FeatML(aedat, inputVar);
    
    % %% Visualize a feature
    % sampleIdx = randperm(size(X,4),1);
    % sList = find(samples);
    % pol = aedat.data.polarity.polarity(sList(sampleIdx));
    % wes = X(:,:,:,sampleIdx);
    % [x,y] = meshgrid(1:size(X,1),1:size(X,2));
    % x = repmat(x,1,1,size(X,3));
    % y = repmat(y,1,1,size(X,3));
    % c=repmat(reshape([1:size(X,3)],1,1,size(X,3)),size(X,1),size(X,2),1);
    % scatter3(x(:),y(:),wes(:),15,c(:),'filled')
    % title([num2str(Y(sampleIdx)) ' - ' num2str(pol)])
    %
    %
    % %% Train CNN to learn probabilities from feature vectors
    % [net, trainMask] = trainDenoiseNetwork(X, Y, inputVar);
    %
    % makeAnimatedGifResults(aedat, samples, trainMask)
    
    %% Write out images for use with pytorch
    
    [~,fn,~] = fileparts(aedat.importParams.filePath);
    folderDir = [outDir fn filesep];
    
    if ~exist(folderDir,'dir')
        mkdir(folderDir)
    end
    
    %TIFF tag setup
    tagstruct.ImageLength = size(X,1);
    tagstruct.ImageWidth = size(X,2);
    tagstruct.Photometric = Tiff.Photometric.MinIsBlack;
    tagstruct.BitsPerSample = 32;
    tagstruct.SamplesPerPixel = size(X,3);
    tagstruct.SampleFormat = Tiff.SampleFormat.IEEEFP;
    tagstruct.PlanarConfiguration = Tiff.PlanarConfiguration.Chunky;
    tagstruct.Software = 'MATLAB';
    
    %Write train data
    sList = find(samples);
    dList = 1:numel(sList);
    file_name = cell(numel(sList),1);
    est_probability = zeros(numel(sList),1);
    polarity = zeros(numel(sList),1);
    for loop = 1:numel(sList)
        clc, loop/numel(sList)
        file_name{loop} = [fn filesep fn '_' num2str(loop) '.tif'];
        est_probability(loop) = Y(dList(loop));
        polarity(loop) = aedat.data.polarity.polarity(sList(loop));
        t = Tiff([outDir file_name{loop}],'w');
        setTag(t,tagstruct)
        write(t,X(:,:,:,dList(loop)));
        close(t);
    end
    T{fLoop} = table(file_name, est_probability, polarity);
    writetable(T{fLoop}, [outDir fn '_gt.csv'])
    
    save([outDir fn '.mat'],'-v7.3')

end


%% Split the data into train/test randomly - put images from the same collect in the same folder train or test
trainPercent = 90;

%Histogram of all values
% allTable = T{1};
% for loop = 1:numel(files)
%     allTable = cat(1,allTable,T{loop});
% end

rng('shuffle')
rIdx = randperm(numel(files));

trainIdx = rIdx <= (floor(numel(files)*trainPercent/100))

mkdir([outDir 'train'])
mkdir([outDir 'test'])

%Move data into train/test subfolder
for fLoop = 1:numel(files)
    [~,fn,~] = fileparts(files(fLoop).name);
    folderDir = [outDir fn];
    folderFile = [outDir fn '_gt.csv'];
    if trainIdx(fLoop)
        moveDir = [outDir 'train' filesep fn]
        moveFile = [outDir 'train' filesep fn '_gt.csv']
    else
        moveDir = [outDir 'test' filesep fn]
        moveFile = [outDir 'test' filesep fn '_gt.csv']
    end
%     unix(['mv ' folderDir ' ' moveDir])
    movefile(folderDir,moveDir)
    movefile(folderFile,moveFile)
    
end

%Copy some test data into validation

%Build the labels

testTruth = dir([outDir filesep 'test' filesep '*_gt.csv']);
for loop = 1:numel(testTruth)
    if loop == 1
        testTable = readtable([outDir filesep 'test' filesep testTruth(loop).name],'Delimiter',',');
    else
        testTable = cat(1,testTable,readtable([outDir filesep 'test' filesep testTruth(loop).name],'Delimiter',','));
    end
end
writetable(testTable, [outDir 'gt_test.csv'])


trainTruth = dir([outDir filesep 'train' filesep '*_gt.csv']);
for loop = 1:numel(trainTruth)
    if loop == 1
        trainTable = readtable([outDir filesep 'train' filesep trainTruth(loop).name],'Delimiter',',');
    else
        trainTable = cat(1,trainTable,readtable([outDir filesep 'train' filesep trainTruth(loop).name],'Delimiter',','));
    end
end
writetable(trainTable, [outDir 'gt_train.csv'])


% idx = find(trainIdx);
% trainTable = T{idx(1)};
% for loop = 2:numel(idx)
%     trainTable = cat(1,trainTable,T{idx(loop)});
% end
% writetable(trainTable, [outDir 'gt_train.csv'])
% 
% idx = find(~trainIdx);
% testTable = T{idx(1)};
% for loop = 2:numel(idx)
%     testTable = cat(1,testTable,T{idx(loop)});
% end
% writetable(testTable, [outDir 'gt_test.csv'])

%For valid dataset use a max of N samples
validIdx = randperm(size(testTable,1),numValidSamples);
writetable(testTable(validIdx,:), [outDir 'gt_valid.csv'])
unix(['ln -s ' outDir 'test ' outDir 'valid']) %no need to reproduce all the files for validation, just make a symlink

%For training iteration dataset use a max of N samples
smallTrainIdx = randperm(size(trainTable,1),numValidSamples);
writetable(trainTable(smallTrainIdx,:), [outDir 'gt_smalTrain.csv'])
unix(['ln -s ' outDir 'train ' outDir 'smallTrain']) %no need to reproduce all the files for small train, just make a symlink

%