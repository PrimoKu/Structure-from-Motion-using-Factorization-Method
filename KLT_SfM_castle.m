
% Machine Perception Project 2
% Structure-from-motion using Factorization Approach (castle sequence)
% team members: Juo-Tung Chen, Yu-chun Ku
% advisor: Professor Rama Chellappa

close all; clear; clc;

% set the order when running through the images 
direction = -1;
if direction > 0       % 1: increasing order
    noFrames = 1;
else
    noFrames = 28;     % -1: decreasing order
end

num_of_img = 28;  % total num of images

for n=1:num_of_img     % store the images into a cell using imread
  images{n} = imread(sprintf('./castlejpg_matlab/castle_%d.jpg',n-1));
end

pointTracker = vision.PointTracker('MaxBidirectionalError', 5);

startFrame = 1;endFrame = 28;tFrames = 28;
detectFlag = 0;


while (noFrames <= tFrames && noFrames >= startFrame)
  videoFrame = images{noFrames};
  if(noFrames >= startFrame && noFrames <= endFrame)
    % get the next frame  
    if (detectFlag == 0)
        figure(1);imshow(videoFrame); drawnow;title('Video Frame');
        ROI = drawrectangle('StripeColor','r');

        % Detect feature points in the frame
        ROIFrame = imcrop(videoFrame,ROI.Position);
        ROIFrame = imresize( ROIFrame, [size(videoFrame,1) size(videoFrame,1)]);
        points = detectMinEigenFeatures(rgb2gray(ROIFrame),'MinQuality',0.001); %, 'ROI', ROI);
        points = points.Location;
        initialize(pointTracker, points, ROIFrame);
        oldPoints = points;
        featurePointsX = zeros(size(points,1),(endFrame-startFrame));
        featurePointsY = zeros(size(points,1),(endFrame-startFrame));
        ROIFrame = insertMarker(ROIFrame, points, '+', ...
                'Color', 'white');
        figure;imshow(ROIFrame);drawnow;
        detectFlag = 1;
    end
    if detectFlag == 1
        % Track the points. Note that some points may be lost.
        ROIFrame = imcrop(videoFrame,ROI.Position);
        ROIFrame = imresize( ROIFrame,[size(videoFrame,1) size(videoFrame,1)]);
        videoFrame = ROIFrame;
        [points, isFound] = step(pointTracker, videoFrame);
        visiblePoints = points(isFound, :);
        oldInliers = oldPoints(isFound, :);
        if size(visiblePoints, 1) >= 2 % need at least 2 points
            % Display tracked points
            videoFrame = insertMarker(videoFrame, visiblePoints, '+', ...
                'Color', 'white');
            % Reset the points
            featurePointsX(isFound,(noFrames-startFrame)+1) = points(isFound,1);
            featurePointsY(isFound,(noFrames-startFrame)+1) = points(isFound,2);
            size(visiblePoints)
        end
        figure(2);imshow(videoFrame);drawnow;
    end
  else
  end
  if direction > 0
    noFrames = noFrames + 1
  else
    noFrames = noFrames - 1
  end

end

featureU = [];
featureV = [];
for i = 1:size(points,1)
    if nnz(featurePointsX(i,:)) == (endFrame - startFrame)+1 %tFrames  
        featureU = [featureU; featurePointsX(i,:)];
        i;
    end
    if nnz(featurePointsY(i,:)) == (endFrame - startFrame)+1 % tFrames  
        featureV = [featureV; featurePointsY(i,:)];
    end
    i = i+1;
end

meanU = mean(featureU);
meanV = mean(featureV);
for i = 1: (endFrame - startFrame) %tFrames-1
    objFeatureU(:,i) = featureU(:,i) - meanU(1,i);
    objFeatureV(:,i) = featureV(:,i) - meanV(1,i);
end
%%
objW = [ objFeatureU'; objFeatureV'];
[O1,S,O2T] = svd(objW);
O2 = O2T';

O1P = O1(:,1:3);
O2P = O2(1:3,:);
SP = S(1:3,1:3);

objR = O1P*(SP^0.5);
objS = (SP^0.5)*O2P;

F = endFrame-startFrame;
[U, D, V] = svd(objW);

%%% 3. make M', S' 
Mhat = U(:, 1:3)*sqrt(D(1:3, 1:3)); 
Shat = sqrt(D(1:3, 1:3))*V(:, 1:3)';

%%% 4. Compute Q, impose the metric constraints
Is = Mhat(1:F, :);
Js = Mhat(F+1:end, :);


gfun = @(a, b)[ a(1)*b(1), a(1)*b(2)+a(2)*b(1), a(1)*b(3)+a(3)*b(1), ...
              a(2)*b(2), a(2)*b(3)+a(3)*b(2), a(3)*b(3)] ;
G = zeros(3*F, 6);
for f = 1:3*F
    if f <= F
        G(f, :) = gfun(Is(f,:), Is(f,:));
    elseif f <= 2*F        
        G(f, :) = gfun(Js(mod(f, F+1)+1, :), Js(mod(f, F+1)+1, :));
    else
        G(f, :) = gfun(Is(mod(f, 2*F),:), Js(mod(f, 2*F),:));
    end
end

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

% could be a programatic way, but hey we "see" 3D or 2D

L = [l(1) l(2) l(3);...
     l(2) l(4) l(5);...
     l(3) l(5) l(6)] ;
if all(eig(L) > 0)
    Q = chol(L);
else

L1 = [l2(1) l2(2) l2(3);...
     l2(2) l2(4) l2(5);...
     l2(3) l2(5) l2(6)] ;

Q = chol(L1); % finally!
end


% (iv) Find R and S [3.14]
R = objR * Q;
S = inv(Q) * objS;

% (v) Align the first camera reference system with the world reference
% system
F = endFrame - startFrame; %noFrames;
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

depth_threshold_1 = -500;
depth_threshold_2 = 400;
S1 = zeros(size(S));

for i = 1:size(S, 2)-1

if (S(3,i) > depth_threshold_1 && S(3,i) < depth_threshold_2 )
    S1(:,i) = S(:,i);

end
end

figure; plot3(S(1, :), S(2, :), S(3, :), '*');
xlin = linspace(min(S(1,:)),max(S(1,:)),500);
ylin = linspace(min(S(2,:)),max(S(2,:)),500);
[X,Y] = meshgrid(xlin,ylin);
Z = griddata(S(1,:),S(2,:),S(3,:),X,Y,'cubic');
mesh(X,Y,Z);
axis tight; hold on;
plot3(S(1,:),S(2,:),S(3,:),'.','MarkerSize',15);
title("shape")

%%
figure; plot3(S1(1, :), S1(2, :), S1(3, :), '*');
xlin = linspace(min(S1(1,:)),max(S1(1,:)),500);
ylin = linspace(min(S1(2,:)),max(S1(2,:)),500);
[X,Y] = meshgrid(xlin,ylin);
Z = griddata(S1(1,:),S1(2,:),S1(3,:),X,Y,'cubic');
mesh(X,Y,Z);
axis tight; hold on;
plot3(S1(1,:),S1(2,:),S1(3,:),'.','MarkerSize',15);
title("castle shape (filtered)")


%% flip to negative depth
% figure; plot3(S1(1, :), S1(2, :), -S1(3, :), '*');
% xlin = linspace(min(S1(1,:)),max(S1(1,:)),500);
% ylin = linspace(min(S1(2,:)),max(S1(2,:)),500);
% [X,Y] = meshgrid(xlin,ylin);
% Z = griddata(S1(1,:),S1(2,:),-S1(3,:),X,Y,'cubic');
% mesh(X,Y,Z);
% axis tight; hold on;
% plot3(S1(1,:),S1(2,:),-S1(3,:),'.','MarkerSize',15);
% title("castle shape (filtered)", fontsize=18)

release(pointTracker);