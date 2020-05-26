function cost = minSingleOffsetAndGamma(apsOffsetHat, gammaPHat, gammaNHat, aedat, inputVar)

disp([apsOffsetHat gammaPHat gammaNHat])

if isfield(inputVar, 'intensityOffsetP')
    inputVar = rmfield(inputVar, 'intensityOffsetP');
end

inputVar.intensityOffset = apsOffsetHat;

% Using APS/IMU calculate a temporal change for each frame/event
aedat = assignJt2Events(aedat, inputVar);

% eV = aedat.data.frame.apsIntGood; %ONLY FIT IN AREAS WHERE APS IS NOT NEAR EXTREMES

if isfield(inputVar, 'intensityOffsetP')
    Jt = aedat.data.frame.JtP;
    fP = -1.*sum(sum(sum(eP.*log(max(min(Jt./gammaPHat,1-eps),0+eps))+~eP.*log(1-max(min(Jt./gammaPHat,1-eps),0+eps)))));
    Jt = aedat.data.frame.JtN;
    fN = -1.*sum(sum(sum(eN.*log(max(min(-1.*Jt./gammaNHat,1-eps),0+eps))+~eN.*log(1-max(min(-1.*Jt./gammaNHat,1-eps),0+eps)))));
else
    
    %Normalize Jt
%     Jt = aedat.data.frame.Jt;
%     randSample = randperm(numel(Jt),1e5);
%     [pjt,XI] = ksdensity(Jt(randSample),[-10:.01:10]);
%     %     [pjt,XI] = ksdensity(aedat.data.frame.Jt(:),[-10:.01:10]);
%     %     pjt = max(pjt,1e-7); %Add a minimum number to avoid value that dominates ML(reverse problem)
%     JtNorm = 1./(interp1(XI,pjt,Jt,'nearest','extrap'));
%     

    %IE filtered (weighted)
    fP = -1.*sum(sum(sum(mean(aedat.data.frame.eEmpty(:)).*aedat.data.frame.eP.*log(max(min(    aedat.data.frame.Jt./gammaPHat,1-eps),0+eps))+mean(aedat.data.frame.eP(:)).*aedat.data.frame.eEmpty.*log(1-max(min(    aedat.data.frame.Jt./gammaPHat,1-eps),0+eps)))));
    fN = -1.*sum(sum(sum(mean(aedat.data.frame.eEmpty(:)).*aedat.data.frame.eN.*log(max(min(-1.*aedat.data.frame.Jt./gammaNHat,1-eps),0+eps))+mean(aedat.data.frame.eN(:)).*aedat.data.frame.eEmpty.*log(1-max(min(-1.*aedat.data.frame.Jt./gammaNHat,1-eps),0+eps)))));

%     %IE filtered
%     fP = -1.*sum(sum(sum(aedat.data.frame.eP.*log(max(min(    aedat.data.frame.Jt./gammaPHat,1-eps),0+eps))+    aedat.data.frame.eEmpty.*log(1-max(min(    aedat.data.frame.Jt./gammaPHat,1-eps),0+eps)))));
%     fN = -1.*sum(sum(sum(aedat.data.frame.eN.*log(max(min(-1.*aedat.data.frame.Jt./gammaNHat,1-eps),0+eps))+aedat.data.frame.eEmpty.*log(1-max(min(-1.*aedat.data.frame.Jt./gammaNHat,1-eps),0+eps)))));
%     
    %     keigo - uniform ML
    %     JtNorm = aedat.data.frame.JtNorm;
%     fP = -1.*sum(sum(sum(eV.*JtNorm.*eP.*log(max(min(Jt./gammaPHat,1-eps),0+eps))+eV.*JtNorm.*~eP.*log(1-max(min(Jt./gammaPHat,1-eps),0+eps)))));
%     fN = -1.*sum(sum(sum(eV.*JtNorm.*eN.*log(max(min(-1.*Jt./gammaNHat,1-eps),0+eps))+eV.*JtNorm.*~eN.*log(1-max(min(-1.*Jt./gammaNHat,1-eps),0+eps)))));
    
    %     keigo - orig ML
    %     fP = -1.*sum(sum(sum(eP.*log(max(min(Jt./gammaPHat,1-eps),0+eps))+~eP.*log(1-max(min(Jt./gammaPHat,1-eps),0+eps)))));
    %     fN = -1.*sum(sum(sum(eN.*log(max(min(-1.*Jt./gammaNHat,1-eps),0+eps))+~eN.*log(1-max(min(-1.*Jt./gammaNHat,1-eps),0+eps)))));
    
    %     %wes - modified
    %     fP = -1.*sum(sum(sum(eP.*log(max(min(Jt./gammaPHat,1-eps),0+eps))+eN.*log(1-max(min(Jt./gammaPHat,1-eps),0+eps)))));
    %     fN = -1.*sum(sum(sum(eN.*log(max(min(-1.*Jt./gammaNHat,1-eps),0+eps))+eP.*log(1-max(min(-1.*Jt./gammaNHat,1-eps),0+eps)))));
    
    %     %wes - modified (10X)
    %     fP = -1.*sum(sum(sum(eP.*log(max(min(Jt./gammaPHat,1-eps),0+eps))+10.*eN.*log(1-max(min(Jt./gammaPHat,1-eps),0+eps)))));
    %     fN = -1.*sum(sum(sum(eN.*log(max(min(-1.*Jt./gammaNHat,1-eps),0+eps))+10.*eP.*log(1-max(min(-1.*Jt./gammaNHat,1-eps),0+eps)))));
    
end

cost = fP + fN;
