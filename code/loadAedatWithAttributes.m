function aedat = loadAedatWithAttributes(file)

aedat = loadAedat(file);
dbclear if error

aedat.info.Attributes = aedat.info.xmlStruct.preferences{2}.root.node.node.node.node.node.map.entry{1}.Attributes;
for aLoop = 2:numel(aedat.info.xmlStruct.preferences{2}.root.node.node.node.node.node.map.entry)
    aedat.info.Attributes = [aedat.info.Attributes aedat.info.xmlStruct.preferences{2}.root.node.node.node.node.node.map.entry{aLoop}.Attributes];
end

%Convert to 1-based indexing
aedat.data.polarity.x = aedat.data.polarity.x + 1;
aedat.data.polarity.y = aedat.data.polarity.y + 1;

%Ensure events are sorted by time
if ~issorted(aedat.data.polarity.timeStamp)
    [aedat.data.polarity.timeStamp,idx] = sort(aedat.data.polarity.timeStamp);
    aedat.data.polarity.y = aedat.data.polarity.y(idx);
    aedat.data.polarity.x = aedat.data.polarity.x(idx);
    aedat.data.polarity.polarity = aedat.data.polarity.polarity(idx);
end

%Calculate bias diff setting values
courseBiasLookupTable = fliplr(logspace(log10(12e-12),log10(25e-6),8));
idx = find(cellfun(@(x) strcmp(x,'DAVIS240C.AddressedIPotCF.DiffBn.BitValueCoarse'),{aedat.info.Attributes.key}));
diffC = str2num(aedat.info.Attributes(idx).value);
idx = find(cellfun(@(x) strcmp(x,'DAVIS240C.AddressedIPotCF.DiffBn.BitValueFine'),{aedat.info.Attributes.key}));
diffF = str2num(aedat.info.Attributes(idx).value);
diffFineTable = linspace(0,courseBiasLookupTable(diffC+1),256);
aedat.info.diffB = diffFineTable(diffF+1);

idx = find(cellfun(@(x) strcmp(x,'DAVIS240C.AddressedIPotCF.OnBn.BitValueCoarse'),{aedat.info.Attributes.key}));
diffOnC = str2num(aedat.info.Attributes(idx).value);
idx = find(cellfun(@(x) strcmp(x,'DAVIS240C.AddressedIPotCF.OnBn.BitValueFine'),{aedat.info.Attributes.key}));
diffOnF = str2num(aedat.info.Attributes(idx).value);
diffOnFineTable = linspace(0,courseBiasLookupTable(diffOnC+1),256);
aedat.info.diffOn = diffOnFineTable(diffOnF+1);

idx = find(cellfun(@(x) strcmp(x,'DAVIS240C.AddressedIPotCF.OffBn.BitValueCoarse'),{aedat.info.Attributes.key}));
diffOffC = str2num(aedat.info.Attributes(idx).value);
idx = find(cellfun(@(x) strcmp(x,'DAVIS240C.AddressedIPotCF.OffBn.BitValueFine'),{aedat.info.Attributes.key}));
diffOffF = str2num(aedat.info.Attributes(idx).value);
diffOffFineTable = linspace(0,courseBiasLookupTable(diffOffC+1),256);
aedat.info.diffOff = diffOffFineTable(diffOffF+1);

