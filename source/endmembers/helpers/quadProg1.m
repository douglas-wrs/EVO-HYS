function [ x ] = quadProg1( G, c, Y, b, Aeq, beq, x0, throtle)

[p,N] = size(Y);
neq = size(Aeq, 1);
%n = size(A, 1);
%m = size(A, 2);
m = p*p;
mu = ones(neq, 1);
y0 = ones(N*p,1);
l0 = ones(N*p,1);
n = N*p;
mu0 = mu;
xnew = x0;
ynew = y0;
lnew = l0;
munew = mu0;

%if throtle == 1
    Q0 = reshape(x0,[p,p]);
    previous = -log(abs(det(Q0)));
    below_threshold = 0;
%end
nniter = 150;
if throtle == 1
    nniter = 300
end
for k = 1:nniter
   
    x = xnew; y = ynew; mu = munew; l = lnew;

    %%%%%%%%%%%%%% DEBUG %%%%%%%%%%%%%
    k;
    Q = reshape(x,[p,p]);
    true_value = -log(abs(det(Q)));
    quadratic_value = c'*x + 0.5*x'*G*x;
    threshold = previous;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    yinvl = ((y.^-1).*l);
    %AtransYinvL = A'.*repmat(yinvl, m, 1);
    temp = reshape(l,[p,N])*Y';
    rd = G*x + c - temp(:) + Aeq'*mu;
    
    temp = reshape(x,[p,p])*Y;
    rb = temp(:) - y - b;
    %size(rb)
    rbeq = Aeq*x - beq;
       
    AtransYinvLA = computeAtransYinvLA(Y,yinvl);
    K      = G + AtransYinvLA;
    Aug    = [K Aeq'; Aeq zeros(neq, neq)];
    %size(yinvl)
    %size(y)
    rh = yinvl .* (rb + y);
    temp = reshape(rh, [p,N])*Y';
    rhaug1 = -rd - temp(:);

    rhaug  = [rhaug1; -rbeq];
    
    DxDm   = Aug\rhaug; 
    DxAff  =  DxDm(1:m);    
    
    temp = reshape(DxAff,[p,p])*Y;
    DyAff  = temp(:) + rb;
    DlAff  = -yinvl.*(y + DyAff);  
 
    mi = y'*l/n;
    
    aDymin = min(-y(DyAff < 0)./DyAff(DyAff < 0));
    if (size(aDymin, 1) == 0 || aDymin > 1) 
        aDymin = 1;
    end
    
    aDlmin = min(-l(DlAff < 0)./DlAff(DlAff < 0));
    if (size(aDlmin, 1) == 0 || aDlmin > 1) 
        aDlmin = 1;
    end
   
    aAff = min(aDymin, aDlmin);
    
    miAff = (y + aAff*DyAff)'*(l + aAff*DlAff)/n;
    
    sigma = (miAff/mi)^3;
    ycorrected = y + (l.^-1).*DlAff.*DyAff - sigma*mi*(l.^-1);
    rh = yinvl.*(rb + ycorrected);
    temp = reshape(rh, [p,N])*Y'; 
    %rhaug1 = -rd - AtransYinvL*(rb + y + (l.^-1).*DlAff.*DyAff - sigma*mi*(l.^-1));
    rhaug1 = -rd -temp(:); 
    
    rhaug  = [rhaug1; -rbeq];
    DxDm   = Aug\rhaug;
    Dx     = DxDm(1:m);
    Dmu    = DxDm(m+1:end);
    
    temp = reshape(Dx,[p,p])*Y;
    Dy     = temp(:) + rb;
    
    Dl     = -yinvl.*(Dy + ycorrected);
    
    tk = 1 - 1/(k+1);
    
    apri  = min(-tk*y(Dy < 0)./Dy(Dy < 0));
    if (size(apri, 1) == 0 || apri > 1) 
        apri = 1;
    end
    
    adual = min(-tk*l(Dl < 0)./Dl(Dl < 0));
    if (size(adual, 1) == 0 || adual > 1) 
        adual = 1;
    end
   
    a = min (apri, adual);
    if throtle == 1 
        a = a/10;
    end
        
    xnew  = x  + a * Dx;
    %if throtle == 1
        Qnew = reshape(xnew, [p,p]);
        true_value_new = -log(abs(det(Qnew)));
    %end
    ynew  = y  + a * Dy;
    lnew  = l  + a * Dl;
    munew = mu + a * Dmu;
    
    if sigma < 1e-8 && mi < 1e-8 
        break
    end
    %if throtle == 1 
        if true_value_new < previous
            previous = true_value_new;
            below_threshold = 1;
        else
            if below_threshold == 1
                xnew = x;
                break;
            end
        end
    %end     
end
x = xnew;
end

