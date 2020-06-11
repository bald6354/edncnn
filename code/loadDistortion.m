function [X, Y, normU, normV, dx_du, dx_dv, dy_du, dy_dv] = loadDistortion()

load('camera/cameraParameters_346.mat')

[X, Y] = meshgrid(1:cameraParams.ImageSize(2), 1:cameraParams.ImageSize(1));

if exist('camera/UandV_346.mat','file')
    load('camera/UandV_346.mat')
else
    U = zeros(size(X));
    V = zeros(size(X));
    for loop = 1:numel(X)
        upd = undistortPoints([X(loop) Y(loop)],cameraParams);%points = Mx2 [X Y]
        U(loop) = upd(1);
        V(loop) = upd(2);
    end
    save('camera/UandV_346.mat','U','V')
end

%Normalize U and V
%Normalized image coordinates are calculated from pixel coordinates by translating to the optical center and dividing by the focal length in pixels. Thus, x and y are dimensionless.
normU = (U - cameraParams.PrincipalPoint(1)) ./ cameraParams.FocalLength(1);
normV = (V - cameraParams.PrincipalPoint(2)) ./ cameraParams.FocalLength(2);

load('camera/jacobian_346.mat','dx_du','dx_dv','dy_du','dy_dv')
