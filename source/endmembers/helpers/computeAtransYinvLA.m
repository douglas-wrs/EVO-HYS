function [AtransYinvLA] = computeAtransYinvLA(Y,sinvl)

[p,N] = size(Y);
Yaug = zeros(p*p, N);
SinvL = reshape(sinvl,[p,N]);
SinvLaug = repmat(SinvL,p,1);
for i=1:p
    Yaug((i-1)*p+1:i*p,:) = repmat(Y(i,:),p,1);
end

AtransYinvLcompact = Yaug .* SinvLaug;

AtransYinvLAcompact = AtransYinvLcompact*Y';

AtransYinvLA = zeros(p*p, p*p);

for i = 1:p
    for k = 1:p
        for j = 1:p
            AtransYinvLA((i-1)*p+k,k+(j-1)*p) = AtransYinvLAcompact((i-1)*p+k,j);
        end
    end
end



