function aedat = assignJt2Events(aedat, inputVar)

%set offset
if ~isfield(inputVar,'intensityOffset')
    inputVar.intensityOffset = 0;
end

% Camera Params
[X, Y, normU, normV] = loadDistortion(inputVar);

% Convert rotation motion into pixel shift
% need omega radians/second (davis reads in deg/sec)
% omega = [aedat.data.imu6.gyroX aedat.data.imu6.gyroY aedat.data.imu6.gyroZ];

numRows = double(aedat.data.frame.size(1));
numCols = double(aedat.data.frame.size(2));

Dawn = 20;
X_deci = X(1:Dawn:end, 1:Dawn:end);
Y_deci = Y(1:Dawn:end, 1:Dawn:end);
% scale = 4.25;
% focalLength = 50;  %mm
% Wx = 4.44;  %width of focal plane in mm
% Wy = 3.33;  %height of focal plane in mm
afovX = 2*atand(inputVar.Wx/2/inputVar.focalLength);  %deg
afovY = 2*atand(inputVar.Wy/2/inputVar.focalLength);  %deg
% radPerPixX = deg2rad(52.3)/240;
% radPerPixY = deg2rad(40.4)/180;
radPerPixX = deg2rad(afovX)/numCols;
radPerPixY = deg2rad(afovY)/numRows;
% [G,Gx,Gy] = Spatial_Gauss(1,3);
%labelTimeThresh = 1e-2;  %Consider only events within this time window of APS frame (sec)

%Put this inside the load_distortion function later
load('jacobian_346.mat')

drawPlot = false;
[Vx, Vy, Jt, JtNull, JtP, JtN, Jx_out, Jy_out, Gx_out, Gy_out] = deal(zeros(numRows, numCols, aedat.data.frame.numDiffImages));
%     profile on
for fLoop = 1:aedat.data.frame.numDiffImages
    
    %     if inputVar.globalShutter
    %         ind = sub2ind(size(allMagicTimes),1,cLoop,fLoop);
    %         omega = deg2rad([imu.gyroX(ind) imu.gyroY(ind) imu.gyroZ(ind)]);
    %
    %         [Vx(:,:,fLoop), Vy(:,:,fLoop)] = repmat(ground_truth_motion(normU(:,cLoop), normV(:,cLoop), -1.*omega),1,numCols);
    %     else
    if aedat.cameraSetup.isRollingShutter
        %per column imu interp
        for cLoop = 1:aedat.data.frame.size(2)
            %             clc,cLoop
            
            ind = sub2ind(size(allMagicTimes),1,cLoop,fLoop);
            omega = deg2rad([imu.gyroX(ind) imu.gyroY(ind) imu.gyroZ(ind)]);
            
            [Vx(:,cLoop,fLoop), Vy(:,cLoop,fLoop)] = ground_truth_motion(normU(:,cLoop), normV(:,cLoop), -1.*omega);
            %             Vx(:,cLoop,fLoop) = Vx_tmp;
            %             Vy(:,:,loop) = reshape(Vy_tmp,size(U));
        end
    else
        omega = deg2rad([aedat.data.frame.imu.gyroX(fLoop) aedat.data.frame.imu.gyroY(fLoop) 0.*aedat.data.frame.imu.gyroZ(fLoop)]);
        [Vx(:,:,fLoop), Vy(:,:,fLoop)] = ground_truth_motion(normU, normV, -1.*omega);
    end
    
    %Use Jacobian transformation to convert the velocities in ideal pixel
    %to distorted (real) pixels
    Vu = Vx(:,:,fLoop);
    Vv = Vy(:,:,fLoop);
    Vx(:,:,fLoop) = dx_du.*Vu + dx_dv.*Vv;
    Vy(:,:,fLoop) = dy_du.*Vu + dy_dv.*Vv;
    
    %Convert motion from rad to pix space - original scalar approx (replaced with Jacobian above)
%     Vx(:,:,fLoop) = Vx(:,:,fLoop) ./ radPerPixX;
%     Vy(:,:,fLoop) = Vy(:,:,fLoop) ./ radPerPixY;
    
    %     end
    %     logIm = real(log(im(:,:,loop)));
    im = double(aedat.data.frame.samples{fLoop});
    
    %Correct fixed pattern noise
    if isfield(inputVar, 'fpn')
        im = im.*inputVar.fpn.slope;
    end
    
    %         im = medfilt2(im); %KEEP OR REMOVE??? TALK TO KEIGO!
    
    %         Jx = conv2(im,Gx,'same')./im;
    %         Jy = conv2(im,Gy,'same')./im;
    
%     %Ok, but estimates high Jt for non-edge pixels, need to be more
%     %specific
%     [Gx, Gy] = imgradientxy(im,'central');

    %Since we have direction, we can use a more accurate estimate.
    [Gx1, Gy1] = imgradientxy(rot90(im,2),'intermediate');
    Gx1 = -1.*rot90(Gx1,2);
    Gy1 = -1.*rot90(Gy1,2);
    [Gx2, Gy2] = imgradientxy(im,'intermediate');
    Gx = (Vx(:,:,fLoop)>0).*Gx1 + (Vx(:,:,fLoop)<=0).*Gx2;
    Gy = (Vy(:,:,fLoop)>0).*Gy1 + (Vy(:,:,fLoop)<=0).*Gy2;
%     
% % % % % % % % % % %     %Mohammeds method
% % % % % % % % % % %     [~,GxK,GyK] = Spatial_Gauss(1,3);
% % % % % % % % % % %     Gx = conv2(im,GxK,'same');
% % % % % % % % % % %     Gy = conv2(im,GyK,'same');
    
    
    %Gsharp
%     Gx = Gx.*abs(Vx(:,:,fLoop).*aedat.cameraSetup.estIntegrationTime);
%     Gy = Gy.*abs(Vy(:,:,fLoop).*aedat.cameraSetup.estIntegrationTime);
    
    %Gsharp(v2 - set min to one(Vx can be very low during slow motion)
    Gx = Gx.*max(1,abs(Vx(:,:,fLoop).*aedat.cameraSetup.estIntegrationTime));
    Gy = Gy.*max(1,abs(Vy(:,:,fLoop).*aedat.cameraSetup.estIntegrationTime));

    %Check
    if isfield(inputVar, 'intensityOffsetP')
        JxP = Gx./(max(im+inputVar.intensityOffsetP,1));   %Divide by zero (can we add one to im???)
        JyP = Gy./(max(im+inputVar.intensityOffsetP,1)); %This is that thing i cant remember
        JxN = Gx./(max(im+inputVar.intensityOffsetN,1));   %Divide by zero (can we add one to im???)
        JyN = Gy./(max(im+inputVar.intensityOffsetN,1)); %This is that thing i cant remember
        JtP(:,:,fLoop) = (JxP.*Vx(:,:,fLoop) + JyP.*Vy(:,:,fLoop));% .* estIntegrationTime;
        JtN(:,:,fLoop) = (JxN.*Vx(:,:,fLoop) + JyN.*Vy(:,:,fLoop));% .* estIntegrationTime;

        Jx = Gx./(max(im,1));   %Divide by zero (can we add one to im???)
        Jy = Gy./(max(im,1)); %This is that thing i cant remember
    elseif isfield(inputVar, 'intensityOffset')
        Jx = Gx./(max(im+inputVar.intensityOffset,1));   %Divide by zero (can we add one to im???)
        Jy = Gy./(max(im+inputVar.intensityOffset,1)); %This is that thing i cant remember
    else
        Jx = Gx./(max(im,1));   %Divide by zero (can we add one to im???)
        Jy = Gy./(max(im,1)); %This is that thing i cant remember
    end
    Jt(:,:,fLoop) = (Jx.*Vx(:,:,fLoop) + Jy.*Vy(:,:,fLoop));
    JtNull(:,:,fLoop) = (Jx.*Vx(:,:,fLoop) + Jy.*-1.*Vy(:,:,fLoop));
    Jx_out(:,:,fLoop) = Jx;
    Jy_out(:,:,fLoop) = Jy;
    Gx_out(:,:,fLoop) = Gx;
    Gy_out(:,:,fLoop) = Gy;
    

    %         Jx = conv2(log(im),Gx,'same'); %similar to Jx above
    
    %     [Jx, Jy] = imgradientxy(medfilt2(im),'central');
    %     Jx = Jx./im;
    %     Jy = Jy./im;

    %add multiply by delta t (aka integration time)
    %     Jt(:,:,fLoop) = -1 .* scale .* (Jx.*Vx(:,:,fLoop) + Jy.*Vy(:,:,fLoop));% .* estIntegrationTime;
    
    if drawPlot
        clf
        subplot(4,2,1)
        imagesc(Jx,[-5 5])
        axis image
        title('Jx')
        colorbar
        subplot(4,2,2)
        imagesc(Jy,[-5 5])
        axis image
        title('Jy')
        colorbar
        subplot(4,2,3)
        imagesc(Vx(:,:,fLoop),[-150 150])
        axis image
        title('Vx')
        colorbar
        subplot(4,2,4)
        imagesc(Vy(:,:,fLoop),[-150 150])
        axis image
        title('Vy')
        colorbar
    
        labelTimeThresh = aedat.cameraSetup.estIntegrationTime;
        subplot(4,2,5:8)
        imagesc(Jt(:,:,fLoop).*labelTimeThresh,[-.1 .1]);colormap gray
        colorbar
        axis image
        hold on
        %             quiver(X_deci, Y_deci,1/estFramesPerSec.*Vx_deci./radPerPixX,1/estFramesPerSec.*Vy_deci./radPerPixY,0, 'g','LineWidth',1)
        scaleQuiver = 80;    %magnify quiver by this amount
        
        Vx_deci = [Vx(1:Dawn:end, 1:Dawn:end, fLoop)];
        Vy_deci = [Vy(1:Dawn:end, 1:Dawn:end, fLoop)];
        
        quiver(X_deci, Y_deci,Vx_deci.*labelTimeThresh.*scaleQuiver,Vy_deci.*labelTimeThresh.*scaleQuiver,0, 'g','LineWidth',1)
        view(0,-90)
        pause(.01)
        
        drawnow()
    end
    %         else
    %             clf
    %             imagesc(Jt(:,:,fLoop).*labelTimeThresh,[-.1 .1]);colormap gray
    %             colorbar
    %             axis image
    %             hold on
    %             scaleQuiver = 5;    %magnify quiver by this amount
    %             quiver(X_deci, Y_deci,Vx_deci./radPerPixX.*labelTimeThresh.*scaleQuiver,Vy_deci./radPerPixY.*labelTimeThresh.*scaleQuiver,0, 'g','LineWidth',1)
    %             view(0,-90)
    %             pause(.01)
    %         end
end

aedat.data.frame.imu.quantiles = [0.01 0.025 0.1 0.25 0.50 0.75 0.9 0.975 0.99];
aedat.data.frame.imu.quantilesVx = quantile(Vx(:),aedat.data.frame.imu.quantiles);
aedat.data.frame.imu.quantilesVy = quantile(Vy(:),aedat.data.frame.imu.quantiles);

disp('Jt Assignment Complete')

%     profile viewer
% if drawPlot
%     clf
%     imagesc(Jt(:,:,loop),[-5 5]);colormap gray
%     hold on
%     quiver(X_deci, Y_deci,1/estFramesPerSec.*scale.*Vx_deci,1/estFramesPerSec.*scale.*Vy_deci,0, 'g','LineWidth',1)
%     %     imagesc(reshape(Vx(:,:,loop),size(U)),[-.1 .1])
%     pause(.3)
%     drawnow
% end
%     [spatialGX(:,:,imIdx),spatialGY(:,:,imIdx)]
%Calculate probabity for each event

%     % Capture the plot as an image
%     frame = getframe(h);
%     imF = frame2im(frame);
%     [imind,cm] = rgb2ind(imF,256);
%     % Write to the GIF File
%     if loop == 2
%         imwrite(imind,cm,filename,'gif', 'Loopcount',inf,'DelayTime',0);
%     else
%         imwrite(imind,cm,filename,'gif','WriteMode','append','DelayTime',0);
%     end
% end

% %orig
% refractoryPeriod = 4.7e-9;
% diff = 7.4e-9;
% diffOn = 95.7e-9;
% diffOff = 189.9e-12;
%
% %19mar
% refractoryPeriod = 4.9e-9;
% diff = 6.8e-9;
% diffOn = 278.1e-9;
% diffOff = 189.9e-12;
%
% %20 comes from inivation website
% threshInc = (diffOn/diff)/20
% threshDec = -1.*(diff/diffOff)/20

% %Calculate bias diff setting values
% courseBiasLookupTable = fliplr(logspace(log10(12e-12),log10(25e-6),8));
% idx = find(cellfun(@(x) strcmp(x,'DAVIS240C.AddressedIPotCF.DiffBn.BitValueCoarse'),{aedat.info.Attributes.key}));
% diffC = str2num(aedat.info.Attributes(idx).value);
% idx = find(cellfun(@(x) strcmp(x,'DAVIS240C.AddressedIPotCF.DiffBn.BitValueFine'),{aedat.info.Attributes.key}));
% diffF = str2num(aedat.info.Attributes(idx).value);
% diffFineTable = linspace(0,courseBiasLookupTable(diffC+1),256);
% diffB = diffFineTable(diffF+1);
%
% idx = find(cellfun(@(x) strcmp(x,'DAVIS240C.AddressedIPotCF.OnBn.BitValueCoarse'),{aedat.info.Attributes.key}));
% diffOnC = str2num(aedat.info.Attributes(idx).value);
% idx = find(cellfun(@(x) strcmp(x,'DAVIS240C.AddressedIPotCF.OnBn.BitValueFine'),{aedat.info.Attributes.key}));
% diffOnF = str2num(aedat.info.Attributes(idx).value);
% diffOnFineTable = linspace(0,courseBiasLookupTable(diffOnC+1),256);
% diffOn = diffOnFineTable(diffF+1);
%
% idx = find(cellfun(@(x) strcmp(x,'DAVIS240C.AddressedIPotCF.OffBn.BitValueCoarse'),{aedat.info.Attributes.key}));
% diffOffC = str2num(aedat.info.Attributes(idx).value);
% idx = find(cellfun(@(x) strcmp(x,'DAVIS240C.AddressedIPotCF.OffBn.BitValueFine'),{aedat.info.Attributes.key}));
% diffOffF = str2num(aedat.info.Attributes(idx).value);
% diffOffFineTable = linspace(0,courseBiasLookupTable(diffOffC+1),256);
% diffOff = diffOffFineTable(diffOffF+1);

% %Temperature and Parasitic Photocurrent Effects in Dynamic Vision Sensors
% threshInc = 1/20.*log(aedat.info.diffOn/aedat.info.diffB)
% threshDec = 1/20.*log(aedat.info.diffOff/aedat.info.diffB)

% Jt = fliplr(Jt);
IND = sub2ind(size(Jt),aedat.data.polarity.y, aedat.data.polarity.x, aedat.data.polarity.closestFrame);
if isfield(inputVar, 'intensityOffsetP')
    aedat.data.polarity.Jt(aedat.data.polarity.polarity>0) = JtP(IND(aedat.data.polarity.polarity>0));
    aedat.data.polarity.Jt(aedat.data.polarity.polarity<=0) = JtN(IND(aedat.data.polarity.polarity<=0));
    aedat.data.frame.JtP = JtP;
    aedat.data.frame.JtN = JtN;
else
    aedat.data.polarity.Jt = Jt(IND);
    aedat.data.polarity.JtNull = JtNull(IND);
end

%Save out Jt values
aedat.data.frame.Jt = Jt;
aedat.data.frame.JtNull = JtNull;
aedat.data.frame.Vx = Vx;
aedat.data.frame.Vy = Vy;
aedat.data.frame.Jx = Jx_out;
aedat.data.frame.Jy = Jy_out;
aedat.data.frame.Gx = Gx_out;
aedat.data.frame.Gy = Gy_out;


% threshInc = 0.263
% threshDec = -0.258
%Calculate a prob of firing given temporal derivative
% dirIdx = aedat.data.polarity.Jt>0;
% aedat.data.polarity.Prob(dirIdx,1) = (threshInc./(aedat.data.polarity.Jt(dirIdx)))./refractoryPeriod;
% dirIdx = aedat.data.polarity.Jt<=0;
% aedat.data.polarity.Prob(dirIdx,1) = (threshDec./(aedat.data.polarity.Jt(dirIdx)))./refractoryPeriod;
% dirIdx = aedat.data.polarity.Jt>0;
% aedat.data.polarity.Prob(dirIdx,1) = aedat.data.polarity.Jt(dirIdx).*labelTimeThresh./threshInc;
% dirIdx = aedat.data.polarity.Jt<=0;
% aedat.data.polarity.Prob(dirIdx,1) = aedat.data.polarity.Jt(dirIdx).*labelTimeThresh./threshDec;

%max prob at 1
% aedat.data.polarity.Prob(aedat.data.polarity.Prob>1) = 1;