import numpy as _np


class Som:
    def __init__(self, data, N):
        """

        :param data: data to be clustered, represented as a matrix of n rows,
            as inputs and m cols as input features
        :param N: map shape, can be:
            float - total number of cells
            [float, float] - dimensions
        """
        self.codebook, self.mshape = self.lininit(data, N)
        self.inputs = self._inputSpace()
        
    def lininit(self, x, N):
        mn = x.mean(0)
        d = mn.size
        MN = _np.tile(mn, (x.shape[0], 1))
        A = x-MN
        E = _np.dot(A.T,A)/(x.shape[0]-1)
        q = _np.linalg.eig(E)
        L = _np.sort(_np.real(q[0]))
        L1 = _np.sqrt(L[L.size-1])
        L2 = _np.sqrt(L[L.size-2])
        if type(N)() == 0:
            N1 = _np.int(_np.round_(_np.sqrt(N*_np.sqrt(L1/L2))))
            N2 = _np.int(_np.round_(_np.ceil(N/N1)))
        else:
            if len(N) == 2:
                N1 = N[0]
                N2 = N[1]
            elif len(N) > 2:
                print("incorrect map shape")
                return
        N1 = max(N1, 2)
        N2 = max(N2, 2)
        if N2 > N1:
            N1, N2 = N2, N1

        V = (_np.real(q[1].T[_np.flipud(_np.argsort(_np.real(q[0])))])).T
        line=_np.zeros(N1)
        for i in range(N1):
            line[i]=L1/2-i*L1/(N1-1)
        net=_np.zeros((N2*N1,d))
        for i in range(_np.int(N2)):
            net.T[d-1][i*N1:(i+1)*N1]=line
            for j in range(_np.int(N1)):
                net.T[d-2][i*N1+j]=-L2/2+L2/(N2-1)*i+_np.power(-1,j+1)*L2/(N2-1)/4
        shape=[N1,N2]
        net=_np.dot(net,V)+_np.tile(mn,(N1*N2,1))
        return net, shape
    
    def _inputSpace(self):
        d=_np.zeros((self.mshape[0]*self.mshape[1],2))
        for j in range(self.mshape[1]):
            for i in range(self.mshape[0]):
                d[j*self.mshape[0]+i][0]=i
                d[j*self.mshape[0]+i][1]=j+(1-_np.power(-1,i+1))/4
        return d

##    def winner(self,x):
##        win=_np.zeros(x.shape[0])
##        dist=_np.zeros(self.codebook.shape[0])
##        D=0;#
##        for i in range(x.shape[0]):
##            for j in range(self.codebook.shape[0]):
##                Q=self.codebook[j]-x[i]
##                dist[j]=_np.sum(Q*Q)
##            win[i]=_np.argmin(dist)
##            D+=_np.min(dist)#
##        return win, D

    def winner(self, x):
        win = _np.zeros(x.shape[0])
        dist = _np.zeros(x.shape[0])
        w2 = _np.zeros(self.codebook.shape[0])
        for i in range(self.codebook.shape[0]):
            w2[i]=_np.dot(self.codebook[i],self.codebook[i])
        for i in range(x.shape[0]):
            distances=w2-2*_np.dot(self.codebook,x[i])+_np.dot(x[i],x[i])
            win[i] = _np.argmin(distances)
            dist[i] = _np.min(distances)
        return win, dist
    
    def hits(self,x):
        win, D=self.winner(x)
        num=self.mshape[0]*self.mshape[1]
        h=_np.zeros(num)
        for i in range(win.size):
            for j in range(num):
                if win[i]==j:
                    h[j]=h[j]+1
                    break
        return h
        
    def neighborhood(self, sigma):
        if sigma<=0:
            H=_np.ones(self.codebook.shape[0])
            h=_np.diag(H)
            return h
        h=_np.zeros((self.codebook.shape[0],self.codebook.shape[0]))
        for i in range(h.shape[0]):
            h[i][i]=1
            for j in range(h.shape[0]-i-1):
                d=self.inputs[i]-self.inputs[j+i+1]
                h[i][j+i+1]=_np.exp(-_np.sum(d*d)/sigma/sigma/2)
                h[j+i+1][i]=h[i][j+i+1]
        return h
                
    def train(self, x, T):
        sigma=_np.float32(self.mshape[1])
        for t in range(T):
            win, D =self.winner(x)
            print('step ',t,'  error ',_np.sum(D))#
            h=self.neighborhood(sigma)
            s=_np.zeros((self.codebook.shape))
            count=_np.zeros(self.codebook.shape[0])
            for i in range(x.shape[0]):
                s[int(win[i])]+=x[i]
                count[int(win[i])]+=1
            for i in range(s.shape[0]):
                denom=_np.sum(count*h[i])
                if denom>0:
                    self.codebook[i]=_np.dot(s.T,h[i])/denom
            sigma/=1.5
        win, D =self.winner(x)
        print('step ',T,'  error ',_np.sum(D))
        return None
