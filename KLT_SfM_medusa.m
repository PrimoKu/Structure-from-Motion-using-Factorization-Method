
% Machine Perception Project 2
% Structure-from-motion using Factorization Approach (medusa dataset)
% team members: Juo-Tung Chen, Yu-chun Ku
% advisor: Professor Rama Chellappa

close all; clear; clc;

% Read a video frame.
videoFileReader = vision.VideoFileReader('./medusa_data.dv');

% Create a KLT Feature point tracker and enable the bidirectional error 
% constraint to make it more robust in the presence of noise and clutter.
pointTracker = vision.PointTracker('MaxBidirectionalError', 2);

% Initializing the parameters of the video frames 
startFrame = 140;
endFrame = 180;
tFrames = 300;
noFrames = 1;
detectFlag = 0;
F = endFrame - startFrame;

%% Perform KLT Feature Tracking on the video
while noFrames <= tFrames 
  videoFrame = step(videoFileReader);
  if(noFrames >= startFrame && noFrames <= endFrame)
    % interactive GUI -> select region of interest(ROI) on the startFrame
    if (detectFlag == 0)
        figure(1);imshow(videoFrame); drawnow; title('Use mouse to select ROI');
        ROI = drawrectangle('StripeColor','r');
        % crop the image according to the ROI
        ROIFrame = imcrop(videoFrame,ROI.Position);
        ROIFrame = imresize(ROIFrame, [size(videoFrame,1) size(videoFrame,1)]);
        % Detect feature points in the frame
        points = detectMinEigenFeatures(rgb2gray(ROIFrame),'MinQuality',0.002); %, 'ROI', ROI);
        points = points.Location;
        initialize(pointTracker, points, ROIFrame);
        oldPoints = points;
        featurePointsX = zeros(size(points,1),(endFrame-startFrame));
        featurePointsY = zeros(size(points,1),(endFrame-startFrame));
        ROIFrame = insertMarker(ROIFrame, points, '+', 'Color', 'white');
        figure; imshow(ROIFrame); drawnow;
        detectFlag = 1;
    end
    if detectFlag == 1
        % crop the image according to the ROI
        ROIFrame = imcrop(videoFrame,ROI.Position);
        ROIFrame = imresize( ROIFrame,[size(videoFrame,1) size(videoFrame,1)]);
        videoFrame = ROIFrame;
        % Track the points. Note that some points may be lost.
        [points, isFound] = step(pointTracker, videoFrame);
        visiblePoints = points(isFound, :);
        oldInliers = oldPoints(isFound, :);
        if size(visiblePoints, 1) >= 2     % need at least 2 points
            % Display tracked points
            videoFrame = insertMarker(videoFrame, visiblePoints, '+', 'Color', 'white');
            featurePointsX(isFound,(noFrames-startFrame)+1) = points(isFound,1);
            featurePointsY(isFound,(noFrames-startFrame)+1) = points(isFound,2);
            size(visiblePoints)
        end
        figure(2);imshow(videoFrame);drawnow;
    end
  end
  % increase the iteration count
  noFrames = noFrames + 1
end

%% Processing the Feature Points
featureU = [];
featureV = [];
for i = 1:size(points,1)
    if nnz(featurePointsX(i,:)) == (endFrame - startFrame)+1 % tFrames  
        featureU = [featureU; featurePointsX(i,:)];
    end
    if nnz(featurePointsY(i,:)) == (endFrame - startFrame)+1 % tFrames  
        featureV = [featureV; featurePointsY(i,:)];
    end
end
% Nornmalize the feature coordinates using their mean
meanU = mean(featureU);
meanV = mean(featureV);
for i = 1: (endFrame - startFrame)                   % tFrames-1
    objFeatureU(:,i) = featureU(:,i) - meanU(1,i);
    objFeatureV(:,i) = featureV(:,i) - meanV(1,i);
end

%% Calculating for R and S
% 1. construct measurement matrix objW
objW = [objFeatureU'; objFeatureV'];     

% 2. perform SVD on objW
[O1,S,O2] = svd(objW);                     

% 3. make M', S' 
M_hat = O1(:, 1:3)*sqrt(S(1:3, 1:3)); 
S_hat = sqrt(S(1:3, 1:3))*O2(:, 1:3)';

% 4. impose the metric constraints
Is = M_hat(1:F, :);
Js = M_hat(F+1:end, :);

fun_g = @(a, b)[ a(1)*b(1), a(1)*b(2)+a(2)*b(1), a(1)*b(3)+a(3)*b(1), ...
              a(2)*b(2), a(2)*b(3)+a(3)*b(2), a(3)*b(3)] ;
G = zeros(3*F, 6);
for f = 1:3*F
    if f <= F
        G(f, :) = fun_g(Is(f,:), Is(f,:));
    elseif f <= 2*F
        G(f, :) = fun_g(Js(mod(f, F+1)+1, :), Js(mod(f, F+1)+1, :));
    else
        G(f, :) = fun_g(Is(mod(f, 2*F),:), Js(mod(f, 2*F),:));
    end
end

% solving Gl = c by SVD and mldivide and compare the results
c = [ones(2*F, 1); zeros(F, 1)];
[U, S, V] = svd(G);
hatl = U'*c;
y = [hatl(1)/S(1,1); hatl(2)/S(2,2); hatl(3)/S(3,3); hatl(4)/S(4,4); ...
    hatl(5)/S(5,5); hatl(6)/S(6,6)];
l = V*y;
fprintf('resid with SVD= Gl - c, %g\n', norm(G*l - c));
l2 = G\c;
fprintf('resid with mldivide = Gl - c, %g\n', norm(G*l2 - c));
% they give the same result because matlab is optimized

% 5. compute Q
L = [l(1) l(2) l(3);
     l(2) l(4) l(5);
     l(3) l(5) l(6)] ;
Q = chol(L); % finally!

% 6. Find R and S
R = M_hat * Q;
S = inv(Q) * S_hat;

% 7. Align the first camera reference system with the global coordinate system
i1 = R(1,:)';
i1 = i1 / norm(i1);
j1 = R(F+1,:)';
j1 = j1 / norm(j1);
k1 = cross(i1, j1);
k1 = k1 / norm(k1);
R0 = [i1 j1 k1];
R = R * R0;
S = inv(R0) * S;

%%
% Display Shape

% depth_threshold_1 = -200;
% depth_threshold_2 = 300;
% for i = 1:size(S, 2)-1
% 
% if (S(3,i) < depth_threshold_1)
%     S(:,i) = [];
% 
% elseif (S(3,i) > depth_threshold_2)
%     S(:,i) = [];
% 
% end
% end

fig = figure; plot3(S(1, :), S(2, :), S(3, :), '*');
fig.Position = [0 0 1000 1000];    % set plot window size

xlin = linspace(min(S(1,:)),max(S(1,:)),500);
ylin = linspace(min(S(2,:)),max(S(2,:)),500); 
[X,Y] = meshgrid(xlin,ylin);
Z = griddata(S(1,:),S(2,:),S(3,:),X,Y,'cubic');
mesh(X,Y,Z);
axis tight; hold on;
plot3(S(1,:),S(2,:),S(3,:),'.','MarkerSize',15);
view(-140,60)
% title("medusa shape (increased min quality)", fontsize=18)


%% flip the depth
% fig = figure; plot3(S(1, :), S(2, :), -S(3, :), '*');
% fig.Position = [0 0 1000 1000];    % set plot window size
% 
% xlin = linspace(min(S(1,:)),max(S(1,:)),500);
% ylin = linspace(min(S(2,:)),max(S(2,:)),500); 
% [X,Y] = meshgrid(xlin,ylin);
% Z = griddata(S(1,:),S(2,:),-S(3,:),X,Y,'cubic');
% mesh(X,Y,Z);
% axis tight; hold on;
% plot3(S(1,:),S(2,:),-S(3,:),'.','MarkerSize',15);
% view(-140,60)


release(pointTracker);