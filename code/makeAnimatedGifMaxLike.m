function makeAnimatedGifMaxLike(aedat, outDir)

h = figure;
axis tight manual % this ensures that getframe() returns a consistent size
[fp,fn,fe] = fileparts(aedat.importParams.filePath)
filename = [outDir filesep fn '_maxLikeLabelsColor.gif'];
% cm = spring(256);
%cm = [linspace(1,0,256)' linspace(0,1,256)' zeros(256,1)];
% cm = hsv2rgb([linspace(0,1/3,256)' ones(256,1) cat(1,linspace(1,0.6,128)',linspace(0.6,1,128)')]);
load('cm.mat')
cm = cm256;

% epsilonNHAT = aedat.cameraSetup.epsilonNHAT_AllFiles;
% epsilonPHAT = aedat.cameraSetup.epsilonPHAT_AllFiles;
%     
% tau = aedat.cameraSetup.estIntegrationTime;
%     
% gammaNHAT = epsilonNHAT/tau;
% gammaPHAT = epsilonPHAT/tau;
%     
% posProb = max(aedat.data.frame.Jt,0)./gammaPHAT;
% negProb = max(-1.*aedat.data.frame.Jt,0)./gammaNHAT;

posProb = max(aedat.data.frame.Jt,0)./aedat.cameraSetup.estGammaP;
negProb = max(-1.*aedat.data.frame.Jt,0)./aedat.cameraSetup.estGammaN;

% while(1)
for loop = 1:(aedat.data.frame.numDiffImages)
    %     for loop = 52:64
    clf
    %     tmp = Jt(:,:,loop) - 5;
    % tmp = tmp./10;
    % tmp(tmp<0)=0;tmp(tmp>1)=1;
    % image(255.*tmp)
    %         imagesc(flipud(Jt(:,:,loop)),[-10 10])
    % pause(.1)
    %         imagesc(fliplr(mat2gray(aedat.data.frame.samples{loop})),[0 1])
%     imagesc(posProb(:,:,loop) - negProb(:,:,loop),[-1 1])
    imagesc(aedat.data.frame.samples{loop},[0 255])
    colormap gray
    hold on
%     idx = (aedat.data.polarity.closestFrame==loop) &aedat.data.polarity.duringAPS == loop;% & (aedat.data.polarity.Prob<0.01 | aedat.data.polarity.Prob>.99);
%     idx = (aedat.data.polarity.closestFrame==loop) &aedat.data.polarity.duringAPS == loop & (aedat.data.polarity.Prob>=.1);
    %         idx = (aedat.data.polarity.closestFrame==loop) & abs(aedat.data.polarity.frameTimeDelta)<(labelTimeThresh./3.*1e6);
%             idx = aedat.data.polarity.duringAPS == loop & aedat.data.polarity.Prob>.9;
%             scatter(aedat.data.polarity.x(idx)+1,aedat.data.polarity.y(idx)+1,10,'g','filled')
%             idx = aedat.data.polarity.duringAPS == loop & aedat.data.polarity.Prob<.1;
%             scatter(aedat.data.polarity.x(idx)+1,aedat.data.polarity.y(idx)+1,10,'r','filled')
            idx = (aedat.data.polarity.duringAPS == loop) & (aedat.data.polarity.apsIntGood);
    ptColorIdx = 256-double(uint8(127.5+(127.5*aedat.data.polarity.Prob(idx).*(2.*aedat.data.polarity.polarity(idx)-1))));
%     ptColorIdx = 256-double(uint8(127.5+(127.5*aedat.data.polarity.Prob(idx).*(2.*aedat.data.polarity.polarity(idx)-1))));
%     ptColorIdx = double(uint8(255.*aedat.data.polarity.polarity(idx)))+1;
    %         ptColorIdx = double(uint8(12.*abs(aedat.data.polarity.Jt(idx))))+1;
    %         ptColorIdx = double(uint8(255.*(aedat.data.polarity.Jt(idx)==0)))+1;
    %use 0.5 offset to put event in middle of pixel
    scatter(aedat.data.polarity.x(idx),aedat.data.polarity.y(idx),5,cm(ptColorIdx,:),'filled')
    view(0,-90)
    pause(.01)
    % Capture the plot as an image
    frame = getframe(h);
    im = frame2im(frame);
    [imind,cm2] = rgb2ind(im,256);
    % Write to the GIF File
    if loop == 1
        imwrite(imind,cm2,filename,'gif', 'Loopcount',inf,'DelayTime',0);
    else
        imwrite(imind,cm2,filename,'gif','WriteMode','append','DelayTime',0);
    end
    pause(.1)
end
% end