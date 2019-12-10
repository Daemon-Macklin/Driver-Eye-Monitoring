% Create a cascade detector object.
faceDetector = vision.CascadeObjectDetector('eyepairsmall');

faces={};
values=[1:1];
index = 1;

% Read images from closed eyes data set
x = dir('closedEyes');
pointsList = {};
for K = 3 : length(x)
    
    % Get path to image
    image = fullfile('closedEyes/' , x(K).name);
    
    % Read image
    imageReader = imread(image);
    
    % Draw a box around the eyes in image
    bbox = step(faceDetector, imageReader);
    
    % Detect Eigen Features in image
    face = detectMinEigenFeatures(rgb2gray(imageReader),'ROI', bbox);
    
    % Check if there are features
    if(~isempty(face))
        
     % If it is not empty then add the values to the list
     faces{end + 1} = face; 
     
     % Add an entry to the measured result list
     values = [values; 0];
    end
end


y = dir('openEyes');

% This loop is nested as the open eyes dataset has folders inside the main
% folder But other than thay works the same
for K = 3 : length(y)
   dict = y(K).name;
   z = dir(fullfile('openEyes/', dict));
   for J = 3 : length(z)
    image = fullfile('openEyes/', dict , z(J).name);
    imageReader = imread(image);
    bbox = step(faceDetector, imageReader);
    face = detectMinEigenFeatures(rgb2gray(imageReader),'ROI', bbox);
    if(~isempty(face))
     faces{end + 1} = selectStrongest(face,10); 
     values = [values; 1];
   end
  end
end

metrics =[];
mainIndex = 1;
pointCount = 0;

% For each of the cornerpoint objects in the faces list
for K = 1: length(faces)
 metric = [1:10];
 index = 1;
 count = faces{K}.Count;
 
 % If there are more than 10 readings
 if(count >= 10)
     
  % Add the first 10 metric readings
  for J = 1 : 10
  try
   metric(index) = faces{K}.Metric(J);
  catch ME
   metric(index) = 0; 
  end
  index = index + 1;
 end
 metrics = [metrics; metric];
 pointCount = pointCount + 1;
 end
end


avgDists = [];
mainIndex = 1;
pointCount = 0;

% For each of the cornerpoint objects in the faces list
for K = 1: length(faces)
 avgDist = [1:10];
 index = 1;
 count = faces{K}.Count;
 
 % If there are more than 10 readings
 if(count >= 10)
     
  % Add the first 10 distances between points
  for J = 1 : 10
  try
   avg = faces{K}.Metric(J: 1) - faces{K}.Metric(J: 1);
   avgDist(index) = faces{K}.Metric(J);
  catch ME
   avgDist(index) = 0; 
  end
  index = index + 1;
 end
 avgDists = [avgDists; avgDist];
 pointCount = pointCount + 1;
 end
end

% Concat the metric and location data
items=[];
for K = 1 : length(metrics)
   item=[1:20];
   item = [metrics(K,:), avgDists(K,:)];
   items = [items; item];  
end

% Create a new random list of indexes to randomise order of data 
ix = randperm(pointCount);
ranItems = [];
ranValues = [];


item = [1:10];
% For each data point
for K = 1 : pointCount
 
 % Get the item at the first random index
 item = items(ix(K),:);
 
 % Add the item to the random list
 ranItems = [ranItems; item];
 
 % Add the corresponding measured values to the random values list
 ranValues = [ranValues ; values(ix(K),:)];
end

% Split the data sets into two subsets. The first subset is to be used as
% the training set and the second is to test the model
PD = 0.5;
cvMet = cvpartition(size(ranItems, 1), 'HoldOut', PD);
cvVal = cvpartition(size(ranValues, 1), 'HoldOut', PD);

Mtrain = ranItems(cvMet.training, :);
Vtrain = ranValues(cvVal.training, :);
Mtest = ranItems(cvMet.test, :);
Vtest = ranValues(cvVal.test, :);

% Create a support vector machine model with the traningng data
SVMModel = fitcsvm(Mtrain, Vtrain);
disp(SVMModel);

% Predict the test data with the model
label = predict(SVMModel, Mtest);

% Count the number of correct predictions
correctPredictions = 0;
total = length(label);
for K = 1 : length(label)
    prediction = label(K, :);
    actual = Vtest(K, :);
    
    if(prediction == actual)
        correctPredictions = correctPredictions + 1;
    end
end

% Display the accuracy score
disp(correctPredictions / total);

% Read features of sample image
imageReader = imread('MeClosed.jpg');
bbox = step(faceDetector, imageReader);
face = detectMinEigenFeatures(rgb2gray(imageReader),'ROI', bbox);
 
index = 1;
for J = 1 : 10
 item(index) = face.Metric(J);
 index = index + 1;
end

% Predict image
testLabel = predict(SVMModel, item);

% Display prediction 
disp(testLabel);


