function aedat = assignJt2Events(aedat, inputVar)

%set offset
if ~isfield(inputVar,'intensityOffset')
    inputVar.intensityOffset = 0;
end

% Camera Params
[~, ~, normU, normV, dx_du, dx_dv, dy_du, dy_dv] = loadDistortion();
numRows = double(aedat.data.frame.size(1));
numCols = double(aedat.data.frame.size(2));

%Temporal derivative
Jt = zeros(numRows, numCols, aedat.data.frame.numDiffImages);

%For each APS image
for fLoop = 1:aedat.data.frame.numDiffImages
    
    %Velocity fields
    [Vx, Vy] = deal(zeros(numRows, numCols));
    
    if aedat.cameraSetup.isRollingShutter
        %per column imu interp
        for cLoop = 1:aedat.data.frame.size(2)
            ind = sub2ind(size(allMagicTimes),1,cLoop,fLoop);
            omega = deg2rad([imu.gyroX(ind) imu.gyroY(ind) 0.*imu.gyroZ(ind)]);
            [Vx(:,cLoop), Vy(:,cLoop)] = ground_truth_motion(normU(:,cLoop), normV(:,cLoop), -1.*omega);
        end
    else
        omega = deg2rad([aedat.data.frame.imu.gyroX(fLoop) aedat.data.frame.imu.gyroY(fLoop) 0.*aedat.data.frame.imu.gyroZ(fLoop)]);
        [Vx, Vy] = ground_truth_motion(normU, normV, -1.*omega);
    end
    
    %Use Jacobian transformation to convert the velocities in ideal pixel
    %to distorted (real) pixels
    Vu = Vx;
    Vv = Vy;
    Vx = dx_du.*Vu + dx_dv.*Vv;
    Vy = dy_du.*Vu + dy_dv.*Vv;
    
    %get the aps image
    im = double(aedat.data.frame.samples{fLoop});
    
    %Correct fixed pattern noise
    if isfield(inputVar, 'fpn')
        im = im.*inputVar.fpn.slope;
    end
    
    %Calculate the spatial derivative
    %Since we have direction, we can use a more accurate estimate.
    [Gx1, Gy1] = imgradientxy(rot90(im,2),'intermediate');
    Gx1 = -1.*rot90(Gx1,2);
    Gy1 = -1.*rot90(Gy1,2);
    [Gx2, Gy2] = imgradientxy(im,'intermediate');
    Gx = (Vx>0).*Gx1 + (Vx<=0).*Gx2;
    Gy = (Vy>0).*Gy1 + (Vy<=0).*Gy2;

    %Gsharp(v2 - set min to one(Vx can be very low during slow motion)
    Gx = Gx.*max(1,abs(Vx.*aedat.cameraSetup.estIntegrationTime));
    Gy = Gy.*max(1,abs(Vy.*aedat.cameraSetup.estIntegrationTime));

    %Linear to log domain
    Jx = Gx./(max(im+inputVar.intensityOffset,1));   %Divide by zero (can we add one to im???)
    Jy = Gy./(max(im+inputVar.intensityOffset,1)); %This is that thing i cant remember
    
    %Optical flow equation
    Jt(:,:,fLoop) = (Jx.*Vx + Jy.*Vy);
    
end

%Assign Jt to each DVS event
IND = sub2ind(size(Jt),aedat.data.polarity.y, aedat.data.polarity.x, aedat.data.polarity.closestFrame);
aedat.data.polarity.Jt = Jt(IND);

%Save out Jt values per APS frame
aedat.data.frame.Jt = Jt;

