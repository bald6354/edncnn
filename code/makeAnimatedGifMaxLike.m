function makeAnimatedGifMaxLike(aedat, outDir)

h = figure;
set(h, 'Position', [1 1 533 377]);
axis tight manual % this ensures that getframe() returns a consistent size
[~,fn,~] = fileparts(aedat.importParams.filePath)
filename = [outDir filesep fn '_maxLikeLabelsColor.gif'];
load('cm.mat')
cm = cm256;

% posProb = max(aedat.data.frame.Jt,0)./aedat.cameraSetup.estGammaP;
% negProb = max(-1.*aedat.data.frame.Jt,0)./aedat.cameraSetup.estGammaN;

for loop = 1:(aedat.data.frame.numDiffImages)
    clf
    imagesc(aedat.data.frame.samples{loop},[0 255])
    colormap gray
    hold on
    idx = (aedat.data.polarity.duringAPS == loop) & (aedat.data.polarity.apsIntGood);
    ptColorIdx = 256-double(uint8(127.5+(127.5*aedat.data.polarity.Prob(idx).*(2.*aedat.data.polarity.polarity(idx)-1))));
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
    pause(.01)
end
