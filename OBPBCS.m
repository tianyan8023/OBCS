function [w] = OBPBCS(data,y,tau_w,tau_d,K,L,B,l_max,t_max)	%OBPBCS algorithm function
% data is the traininng data
% y is the labels corresponding to the data
% tau_w and tao_d is the step size 
% K is the sparsity of feature selection vector
% S is the number for scattered feature selection vector
% L is the length of block
% l_max and t_max are the number of the iterations for obtaining w and D
% w is the reconstructed feature selection vector
   
    [M,N] = size(data);
    D = data;
    block_num = (K-L)/B;
    w0 = zeros(N,1);
    w1 = zeros(N,1);

    % Change the size of data to block data
    l = rem(N-L,B);
    n = (N-l)/B;
    if l==0
        b = zeros(n,1);
    else
        b = zeros(n+1,1);
    end

    iterN = 10;
    for iteri = iterN:-1:1
        if iteri > 1
            s_lim = 0;
            s = Inf;
            s_limE = 0;
            s_lim1 = 0;
            s1 = Inf;
            i1=0;
            sE = Inf;
            i = 0;
            iE = 0;

            % Iterations for obtaining block feature selection vector
            while (s_lim < s) && (i < l_max)
                % Compute Gradient
                g = D'*(sign(D*w0) - y);
                % Step
                w0_temp = w0 - tau_w.*g;
                % Select best K-term block(threshold)
                for j = 1:n
                   b(j,1) = norm(w0_temp(((B*(j-1)+1):B*j),1),2);
                end
                if l~=0
                   b(n+1,1) = norm(w0_temp(((N-l+1):N),1),2); 
                end
                % Threshold
                [~, aidx] = sort(abs(b), 'descend');           
                m = aidx(block_num+1:end);        
               for j = 1:length(m)
                    if m(j) == n+1
                        w0_temp(B*m(j)-(B-1):N) = 0;
                    elseif m(j)~=n+1
                        w0_temp(B*m(j)-(B-1):B*m(j)) = 0;
                    end
                end           
                % Update w0
                w0 = w0_temp;
                % Threshold
                s = nnz(y - sign(D*w0));
                i = i+1;
            end
            
            % Iterations for obtaining scattered feature selection vector
            while (s_lim1 < s1) && (i1 < l_max)
                % Compute Gradient
               g1 = D'*(sign(D*w1) - y);
                % Step
               w1_temp = w1 - tau_w.*g1;
               w1_temp(find(w0~=0)) = 0;
                % Best K-term (threshold)
               [~, aidx] = sort(abs(w1_temp), 'descend');
                   w1_temp(aidx(L+1:end)) = 0;
                % Update w0
               w1 = w1_temp;
                % Threshold
               s1 = nnz(y - sign(D*w1));
               i1 = i1+1;
            end
             

            w = w1+w0; % Combine block feature selection vector and scattered one

            % Iterations for removing noise in D 
            while (s_limE < sE) && (iE < t_max)
                for rowi = 1:M
                    % Compute Gradient
                    gE = (sign(D(rowi,:)*w) - y(rowi))*w;
                    % Step
                    D(rowi,:) = D(rowi,:) - tau_d.*gE.';
                end
                % Threshold
                sE = nnz(y - sign(D*w));
                iE = iE+1;
            end        
        else
            break;
        end
    end
end    


