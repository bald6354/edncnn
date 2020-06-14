function compressEPMfiles(outDir)

%Significantly reduces file size for dataset publication

files =  dir([outDir '*_epm.mat']);

for fLoop = 1:numel(files)
  
    [fp,fn,fe] = fileparts([outDir files(fLoop).name]);
    load([outDir files(fLoop).name], 'aedat');
    load([outDir files(fLoop).name], 'inputVar');

    makeAnimatedGifMaxLike(aedat, outDir)
    
    epm = aedat.data.frame.Jt;
    epm(epm>0) = epm(epm>0)./aedat.cameraSetup.estGammaP;
    epm(epm<0) = epm(epm<0)./aedat.cameraSetup.estGammaN;
    epm = int16(epm.*32767);
    
    save([outDir fn '_array.mat'],'epm','-v7.3')
    
end