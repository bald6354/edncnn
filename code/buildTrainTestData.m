function buildTrainTestData(outDir)

%Combine all features for training/testing
files = dir([outDir '*_labels.mat']);

for loop = 1:numel(files)
    
    clc, loop

    load([outDir files(loop).name],'X','Y')
    
    if (loop==1)
        X_all = X;
        Y_all = Y;
    else
        X_all = cat(4,X_all,X);
        Y_all = cat(1,Y_all,Y);
    end
    
    %dataset label where feature originated
    setLabel = cat(1,setLabel,loop.*ones(size(X,4),1));
    
end

X = X_all;
Y = Y_all;

%DVSNOISE20 has 3 datasets per scene (group)
grpLabel = floor((setLabel-1)/3) + 1;

save([outDir 'all_labels.mat'],'X','Y','setLabel','grpLabel','-append')
