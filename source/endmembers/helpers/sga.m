function [endmemberindex, duration] = SGA(imagecube,p)
start_time=cputime();
n = 1
% Simplex Growing Algorithm
% - - - - - - - Input variables - - - - - - - - - - - -
% ’imagecube’ - The data transformed components [row column band]
% ’p’ - The number of endmembers to be generated
%
% if band > p, then the program will automatically use Singular Value Decomposition to calculate the volume
% - - - - - - - Output variables - - - - - - - - - - -
% ’endmemberindex - The locations of the final endmembers (x,y)
% ’duration - The number of seconds used to run this program
% Set initial condition
initial=0;
[row, column, band]=size(imagecube);
Y = reshape(imagecube,[row*column,band]);
% Determine to use SVD to calculate the volume or not
if(band > p),
use_svd=1;
else
use_svd=0;
end
% Start to count the CPU computing time

% Randomly Select a point as the initial point
endmemberindex=[ceil(row*rand);ceil(column*rand)];
% The main algorithm
while n<p 
% if get enough endmember group, it stops
% Generate endmember vector from reduced cub
endmember=[];
for i=1:n
if(use_svd)
endmember=[endmember squeeze(imagecube(endmemberindex(1,i),endmemberindex(2,i),:))];
else
endmember=[endmember squeeze(imagecube(endmemberindex(1,i),endmemberindex(2,i),1:n))];
end
end
% Use each sample vector to calculate new volume
newendmemberindex=[];
maxvolume=0;
for i=1:row,
for j=1:column,
if(use_svd)
jointpoint=[endmember squeeze(imagecube(i,j,:))];
s=svd(jointpoint);
volume=1;
for z=1:n+1,
volume=volume*s(z);
end
else
jointpoint=[endmember squeeze(imagecube(i,j,1:n))];
jointmatrix=[ones(1,n+1);jointpoint];
volume=abs(det(jointmatrix))/factorial(n); % The formula of a simplex volume
end
if volume > maxvolume,
maxvolume=volume;
newendmemberindex=[i;j];
end
end
end
endmemberindex=[endmemberindex newendmemberindex]; % Add this pixel into the endmember group
%nfinder_plot(endmemberindex);
n=n+1;
if initial==0, % Use new pixel as the initial pixel
n=1;
endmemberindex(:,1)=[];
initial=initial+1;
end
end
% Switch the results back to X and Y
endmemberindex(3,:)=endmemberindex(1,:);
endmemberindex(1,:)=[];
endmemberindex=endmemberindex';
endmemberindex = endmemberindex(:,1) .* endmemberindex(:,2);
duration=cputime()-start_time;
