function [X, Y, normU, normV] = loadDistortion(inputVar)

if inputVar.cameraID == 240
    load('cameraParameters_240.mat')
    
    [X, Y] = meshgrid(1:cameraParams.ImageSize(2), 1:cameraParams.ImageSize(1));
    
    if exist('UandV_240.mat','file')
        load('UandV_240.mat')
    else
        U = zeros(size(X));
        V = zeros(size(X));
        for loop = 1:numel(X)
            upd = undistortPoints([X(loop) Y(loop)],cameraParams);%points = Mx2 [X Y]
            U(loop) = upd(1);
            V(loop) = upd(2);
        end
        save('UandV_240.mat','U','V')
    end
    
    %Normalize U and V
    %Normalized image coordinates are calculated from pixel coordinates by translating to the optical center and dividing by the focal length in pixels. Thus, x and y are dimensionless.
    normU = (U - cameraParams.PrincipalPoint(1)) ./ cameraParams.FocalLength(1);
    normV = (V - cameraParams.PrincipalPoint(2)) ./ cameraParams.FocalLength(2);
    
elseif inputVar.cameraID == 346
    
    load('cameraParameters_346.mat')
    
    [X, Y] = meshgrid(1:cameraParams.ImageSize(2), 1:cameraParams.ImageSize(1));
    
    if exist('UandV_346.mat','file')
        load('UandV_346.mat')
    else
        U = zeros(size(X));
        V = zeros(size(X));
        for loop = 1:numel(X)
            upd = undistortPoints([X(loop) Y(loop)],cameraParams);%points = Mx2 [X Y]
            U(loop) = upd(1);
            V(loop) = upd(2);
        end
        save('UandV_346.mat','U','V')
    end
    
    %Normalize U and V
    %Normalized image coordinates are calculated from pixel coordinates by translating to the optical center and dividing by the focal length in pixels. Thus, x and y are dimensionless.
    normU = (U - cameraParams.PrincipalPoint(1)) ./ cameraParams.FocalLength(1);
    normV = (V - cameraParams.PrincipalPoint(2)) ./ cameraParams.FocalLength(2);
    
end