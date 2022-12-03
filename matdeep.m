path = fullfile("D:\\spec_image1");
f = imageDatastore(path,"IncludeSubfolders",true,LabelSource='foldernames');
trainfiles = 4000;
[train,valid] = splitEachLabel(f,trainfiles,"randomized");
classes = 3;
layers = [
    imageInputLayer([100,100,3])
    convolution2dLayer(5,20)
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    fullyConnectedLayer(3)
    softmaxLayer
    classificationLayer
];

options = trainingOptions("sgdm",'MaxEpochs',10,'Plots','training-progress','ValidationData',valid,InitialLearnRate=0.00001,Verbose=false);
model = trainNetwork(train,layers,options);


