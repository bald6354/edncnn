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
% DVSNOISE20 website. If you need to convert your data use one of these two
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

% Path to data files in MAT format (put files you wish to process here)
mainDir = '/media/wescomp/WesDataDrive/2_mat/'

% Path to output directory (results go here)
outDir = '/media/wescomp/WesDataDrive/edncnn_output/'

% Settings
inputVar.depth = 2; %feature depth per polarity (k in paper)
inputVar.neighborhood = 12; %feature neighborhood (m in paper)(0=1x1, 1=3x3, 2=5x5, etc.)
inputVar.maxNumSamples = 100e3; %Only sample up to this many events per file for pos and neg polarities combined
inputVar.waitBuffer = 2; %time in seconds to wait before sampling an event - early events have no history & dvs tends to drop the feed briefly in the first second or so
inputVar.minTime = 150; %any amount less than 150 microseconds can be ignored (helps with log scaling)
inputVar.maxTime = 5e6; %any amount greater than 5 seconds can be ignored (put data on fixed output size)
inputVar.maxProb = 1; %"probability" score capped at this num
inputVar.nonCausal = false; %if true, double feature size by creating surface both back in time AND forward in time
inputVar.removeGyroBias = 0.5; %use the first 0.5 seconds of data to remove a bias in the gyro
inputVar.numValidSamples = 2e5; %How many samples to randomly grab from test dataset for validation
inputVar.writeOutGIF = false; %write out an animated gif of the EPM labels assigned to the events

% Camera/Lens details - DAVIS346
inputVar.focalLength = 4; %mm (4.5=240C, 4=346(wide), 12=346(zoom)
inputVar.Wx = 6.4;  %width of focal plane in mm
inputVar.Wy = 4.8;  %height of focal plane in mm
load('camera/fpn_346.mat')
inputVar.fpn = fpn; %fixed pattern noise
clear fpn

% Add EDnCNN code to path
addpath('code')

    
%% Process each file

% Gather a list of files and 
files = dir([mainDir '*.mat']);

% Make output directory (if needed)
if ~exist(outDir, 'dir')
    mkdir(outDir)
end

for fLoop = 1:numel(files)
    
    close all
    
    file = [mainDir files(fLoop).name]
    [fp,fn,fe] = fileparts(file);

    % Read in aedat data (Requires events, frames, and IMU data) (aedat4 converted to .mat via aedat4tomat.py)
    aedat = loadMatWithAttributes(file);

    % Match DVS events to APS frames
    aedat = eventTiming(aedat, inputVar);

    %Match each DVS event to an APS intensity
    aps = reshape(cell2mat(aedat.data.frame.samples),aedat.data.frame.size(1),aedat.data.frame.size(2),[]);
    aedat.data.frame.sizeCube = size(aps);
    apsIdx = sub2ind(size(aps),aedat.data.polarity.y,aedat.data.polarity.x,aedat.data.polarity.closestFrame);
    aedat.data.polarity.apsIntensity = aps(apsIdx);

    %Is the APS data near zero or saturation (APS near extremes is not good for training)
    minAps = 5;
    maxAps = 250;
    apsIntGood = (aps>=minAps) & (aps<=maxAps);
    %For each APS frame
    for apsLoop = 1:size(aps,3)
        apsIntGood(:,:,apsLoop) = imdilate(medfilt2(apsIntGood(:,:,apsLoop)),ones(3));
    end
    aedat.data.polarity.apsIntGood = apsIntGood(apsIdx);
    aedat.data.frame.apsIntGood = apsIntGood;
    clear aps apsIntGood

    %Set tau to integration time
    inputVar.tau = aedat.cameraSetup.estIntegrationTime;
    
    %Find pixels with pos/neg DVS events during APS frame
    [eP, eN, eEmpty] = deal(zeros(aedat.data.frame.sizeCube));
    
    for loop = 1:aedat.data.frame.numDiffImages

        validEventsWithinFrameIdx = (aedat.data.polarity.duringAPS == loop) & ...
            aedat.data.polarity.apsIntGood;
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
    
    % Estimate camera contrast thresholds using fminsearch with multiple variables (slow)
    fun3d = @(b) minSingleOffsetAndGamma(b(1), b(2), b(3), aedat, inputVar);
    opts = optimset('PlotFcns','optimplotfval','MaxIter',100,'MaxFunEvals',5000,'TolFun',10,'TolX',.1);
    b_guess = [50 20 20]; %1=APS offset, 2=Gamma(pos), 3=Gamma(neg)
    [b_min,cost_fit,flag_fit,output_fit] = fminsearch(fun3d, b_guess, opts);
    inputVar.intensityOffset = b_min(1);
    aedat.cameraSetup.estGammaP = b_min(2);
    aedat.cameraSetup.estGammaN = b_min(3);
    
    % Using APS/IMU calculate a temporal change for each frame/event
    aedat = assignJt2Events(aedat, inputVar);

    % Assign probability to each event based on temporal derivative and gamma (EPM)
    dirIdx = aedat.data.polarity.polarity>0;
    aedat.data.polarity.Prob(dirIdx,1) = aedat.data.polarity.Jt(dirIdx)./aedat.cameraSetup.estGammaP;
    dirIdx = aedat.data.polarity.polarity<=0;
    aedat.data.polarity.Prob(dirIdx,1) = -1.*aedat.data.polarity.Jt(dirIdx)./aedat.cameraSetup.estGammaN;

    %Fix prob to range 0-inputVar.maxProb
    aedat.data.polarity.Prob(aedat.data.polarity.Prob<0) = 0;
    aedat.data.polarity.Prob(aedat.data.polarity.Prob>inputVar.maxProb) = inputVar.maxProb;
    
    if inputVar.writeOutGIF
        %make an animated gif of the file 
        makeAnimatedGifMaxLike(aedat, outDir)
    end
    
    save([outDir fn '_epm.mat'],'aedat','inputVar','-v7.3')
    
end


%% Write out features for EDnCNN network training/testing

%Create features with labels from each dataset
writeOutData(outDir)

%Combine data from each dataset into one train/test dataset
buildTrainTestData(outDir)


%% Train/test EDnCNN network



