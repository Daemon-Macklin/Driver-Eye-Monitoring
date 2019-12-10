% Create a cascade detector object that will look for eyes.
faceDetector = vision.CascadeObjectDetector('eyepairsmall');

% Load the test video
videoFileReader = vision.VideoFileReader('closed.webm');

% Counter for time spent with eyes closed
closedEyes = 0;

% While there is a frame to process
while ~isDone(videoFileReader)
  
  % Get the frame
  videoFrame      = step(videoFileReader);
  
  % Draw a box around the detected eyes
  bbox            = step(faceDetector, videoFrame);

  % Expand the box for a better view of the eyes
  bbox(1) = bbox(1) - 50;
  bbox(2) = bbox(2) - 50;
  bbox(3) = bbox(3) + 50;
  bbox(4) = bbox(4) + 50;
    
  % Try to convert the image to gray scale
  % And crop out everything not in the box around the eyes
  try
   BW1 = rgb2gray(imcrop(videoFrame, bbox));
  catch ME
  end
  
  % Use the sobel method for edge detection 
  BW2 = edge(BW1,'sobel');
  
  % Create a figure and show the original cropped image
  % Uncomment this to view circles on images. 
  % WARNING: this will put up a figure for each frame in video
  %figure;
  %imshow(BW1)
  
  % Find all of the circles in the cropped image
  [centers, radii, metric] = imfindcircles(BW2,[7 15]);
  
  % If circles are found
  if(~isempty(centers))
    
    % Make the title of the figure open and draw the circles
    title('Open');
    viscircles(centers, radii,'EdgeColor','b');
    
    % Display open and reset the closedEyes Counter
    disp('open')
    closedEyes = 0;
  else
    
    % Make the title of the figure closed
    title('Closed');
    
    % Display closed and increment the closedEyes Counter
    disp('closed');
    closedEyes = closedEyes + 1;
  end
 
  % Check the state of the closedEyes var
  if(closedEyes == 10)
      
    % If it is 10 then the driver should be warned to be more aleart
    disp ('Warn Driver');
  elseif(closedEyes == 30)
    
    % If it is 30 the car needs to be pulled over
    disp('Pull Car Over');
    
    % Stop the program to represent the vehicle stopping
    return
  end
  
  % Wait .25 seconds to give matlab a chance to process the image
  pause(.25);
end
