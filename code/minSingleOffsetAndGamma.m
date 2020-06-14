function cost = minSingleOffsetAndGamma(apsOffsetHat, gammaPHat, gammaNHat, aedat, inputVar)

disp([apsOffsetHat gammaPHat gammaNHat])

if isfield(inputVar, 'intensityOffsetP')
    inputVar = rmfield(inputVar, 'intensityOffsetP');
end

inputVar.intensityOffset = apsOffsetHat;

% Using APS/IMU calculate a temporal change for each frame/event
aedat = assignJt2Events(aedat, inputVar);

if isfield(inputVar, 'intensityOffsetP')
    Jt = aedat.data.frame.JtP;
    fP = -1.*sum(sum(sum(eP.*log(max(min(Jt./gammaPHat,1-eps),0+eps))+~eP.*log(1-max(min(Jt./gammaPHat,1-eps),0+eps)))));
    Jt = aedat.data.frame.JtN;
    fN = -1.*sum(sum(sum(eN.*log(max(min(-1.*Jt./gammaNHat,1-eps),0+eps))+~eN.*log(1-max(min(-1.*Jt./gammaNHat,1-eps),0+eps)))));
else
    fP = -1.*sum(sum(sum(mean(aedat.data.frame.eEmpty(:)).*aedat.data.frame.eP.*log(max(min(    aedat.data.frame.Jt./gammaPHat,1-eps),0+eps))+mean(aedat.data.frame.eP(:)).*aedat.data.frame.eEmpty.*log(1-max(min(    aedat.data.frame.Jt./gammaPHat,1-eps),0+eps)))));
    fN = -1.*sum(sum(sum(mean(aedat.data.frame.eEmpty(:)).*aedat.data.frame.eN.*log(max(min(-1.*aedat.data.frame.Jt./gammaNHat,1-eps),0+eps))+mean(aedat.data.frame.eN(:)).*aedat.data.frame.eEmpty.*log(1-max(min(-1.*aedat.data.frame.Jt./gammaNHat,1-eps),0+eps)))));
end

cost = fP + fN;
