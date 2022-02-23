# Filename: filter2.py
# Program to generate recursive filter coefficients
# Version 2.1 
from math import pi,sqrt,log
from numpy import array, zeros, exp, cos, sin, tan, arctan, arcsin, arccos, conjugate

import matplotlib.pyplot as plt
fig = plt.figure()

def pause():
    junk=input("Press return")

def low():
    global RIPPLE, REJECTION, FPASS, FSTOP, FS
    RIPPLE=float(input("Maximum Passband Ripple (+-dB)\n"))
    REJECTION=float(input("Minimum Stopband Rejection (dB)\n"))
    FPASS=float(input("Upper Limit of Passband (Hz)\n"))
    FSTOP=float(input("Lower Limit of Stopband (Hz)\n"))
    FS=float(input("Sampling Frequency (Hz)\n"))
    (N,A,B)=delpfilt(RIPPLE, REJECTION, FPASS, FSTOP, FS)
    return (N,A,B)

def high():
    global RIPPLE, REJECTION, FPASS, FSTOP, FS
    RIPPLE=float(input("Maximum Passband Ripple (+-dB)\n"))
    REJECTION=float(input("Minimum Stopband Rejection (dB)\n"))
    FPASS=float(input("Lower Limit of Passband (Hz)\n"))
    FSTOP=float(input("Upper Limit of Stopband (Hz)\n"))
    FS=float(input("Sampling Frequency (Hz)\n"))
    (N,A,B)=dehpfilt(RIPPLE, REJECTION, FSTOP, FPASS, FS)
    return (N,A,B)

def output(N,A,B):
    print ("\nFilter Equation:\n")
    print ("y_n  = a_0 x_n + a_1 x_(n-1) + ... + a_n x_0 - B_1 y_(n-1) - B_2 y_(n-2) - ... -  B_n y_0")
    #print (" t    0 t    1 t-1           N t-N     1 t-1     2 t-2          N t-N")
    print ("\nFilter Coefficients:\n")
    print (" i \t A(i) \t\t\t B(i)")
    print (" 0 \t %.16f \t %.16f" %(0, A[0]))
    for i in range(1,N+1): print (" %d \t %.16f \t %.16f" %(i, A[i], B[i]))

def recurrence(POLE, ZERO, N):
    c=zeros(20, complex)
    A=zeros(20, float)
    B=zeros(20, float)
    c[1]=-POLE[1]
    for k in range(2,N+1):
        z=POLE[k]
        c[k]=-z*c[k-1]
        if k!=2: c=extra(z,c,k)
        c[1]=c[1]-z
    B[1:N+1]=c[1:N+1].real
    c[1]=-ZERO[1]
    for k in range(2,N+1):
        z=ZERO[k]
        c[k]=-z*c[k-1]
        if k!=2: c=extra(z,c,k)
        c[1]=c[1]-z
    A[1:N+1]=c[1:N+1].real
    return (A,B)

def extra(z, c, k):
    for l in range(2, k):
        m=k-l+1
        c[m]=c[m] - z*c[m-1]
    return c

def delpfilt(RIPPLE, REJECT, FPASS, FSTOP, FS):
    (N,POLE,ZERO)=delpz(RIPPLE, REJECT, FPASS, FSTOP, FS)
    (A,B)=recurrence(POLE, ZERO, N)
    a=1; b=1
    for i in range(1,N+1): a=a+A[i]; b=b+B[i]
    A[0]=1
    for i in range(0,N+1): A[i]=A[i]*b/a
    return (N,A,B)

def delpz(RIPPLE, REJECTION, FPASS, FSTOP, FS):
    x=pi/FS
    vc=tan(FPASS*x)
    vr=tan(FSTOP*x)
    (n,POLE,ZERO)=elpfilt(RIPPLE, REJECTION, vc, vr)
    for m in range(1, n+1):
        POLE[m]=(1.0+POLE[m])/(1.0-POLE[m])
        ZERO[m]=(1.0+ZERO[m])/(1.0-ZERO[m])
    if (n % 2)==1: ZERO[n]=-1.0
    return (n, POLE, ZERO)

def dehpfilt(RIPPLE, REJECT, FSTOP, FPASS, FS):
    (N, POLE, ZERO)=delpz(RIPPLE, REJECT, 0.5*FS-FPASS, 0.5*FS-FSTOP, FS)
    for m in range(1, N+1):
        POLE[m]=-POLE[m]
        ZERO[m]=-ZERO[m]

    (A,B)=recurrence(POLE, ZERO, N)
    a=A[N]; b=B[N]; A[0]=1; B[0]=1
    for i in range(1,N+1):
        a=-a+A[N-i]
        b=-b+B[N-i]
    for i in range(0,N+1): A[i]=A[i]*b/a
    return (N, A, B)

def elpfilt(RIPPLE, REJECTION, PASSBLIM, STOPBLIM):
    POLE=zeros(20, complex)
    ZERO=zeros(20, complex)
    k=PASSBLIM/STOPBLIM
    eps=sqrt(exp(0.1*RIPPLE*log(10.0))-1)
    x=exp(-0.1*REJECTION*log(10.0))
    k1=eps*sqrt(x/(1-x))

    p=KK(k*k)
    q=KK(k1*k1)
    r=KK(1-k*k)
    s=KK(1-k1*k1)

    n=int(p*s/(q*r)+1)
    k1=parameter(n*r/p)
    s=KK(1-k1*k1)
    u0=bodge(k1, 1/eps)*r/s
    p=p/n
    n2=n//2; m=2*n2+1
    for j in range(1,n2+1):
        a=sn(k, complex(p*(n+1-2*j), u0))
        POLE[j]=PASSBLIM*(1J)*a
        POLE[m-j]=conjugate(POLE[j])
        a=sn(k, complex(p*(n+1-2*j), r))
        ZERO[j]=PASSBLIM*(1J)*a
        ZERO[m-j]=conjugate(ZERO[j])

    if (n % 2) > 0:
        a=sn(k, complex(0,u0))
        POLE[n]=PASSBLIM*(1J)*a
        ZERO[n]=(0J)
    return (n, POLE, ZERO)

def bodge(k,x):
    return F(sqrt(1.0-k*k),arctan(x))

def sn(k, z):
    kk=KK(k*k)
    kkprime=KK(1.0-k*k)
    tau=(1.0J)*kkprime/kk
    u=1.5707963267*z/kk
    a=theta(1,u,tau)
    c=theta(4,u,tau)
    return a/(c*sqrt(k))

def parameter(w):
    tau=complex(0.0,w)
    a=theta(2,(0+0J),tau)
    c=theta(3,0+0J,tau)
    return ((a/c)**2).real

def theta(i, s, tau):
    z=s
    M=exp(0.25*pi*(1J)*tau + (1J)*z)
    f=1.0
    if   i==1: z=z + 0.5*pi + 0.5*pi*tau; f=(-1J)*M
    elif i==2: z=z + 0.5*pi*tau;          f=M
    elif i==4: z=z + 0.5*pi
    q=exp((1J)*pi*tau)
    m=int((z.imag/(pi*tau.imag))+0.5)
    z=z - pi*m*tau
    clast=1.0; c=2.0*cos(2.0*z); cn=0.5*c
    w=0; u=q; v=q
    while(True):
        t=u*cn
        w=w+t
        v=v*q*q; u=u*v
        cnext=cn*c - clast; clast=cn; cn=cnext
        if abs(t)< 1e-6*abs(w): break

    w=1+2*w
    w=w*exp(m*(-1J)*(2*z + pi*tau))*f
    return w

def KK(x):
    y=1-x
    z=1.38629436112+y*(0.09666344259+y*(0.03590092383+y*(0.03742563713+y* \
    0.01451196212)))-(0.5+y*(0.12498593597+y*(0.06880248576+y*(0.03328355346+y* \
    0.00441787012))))*log(y)
    return z

def EE(x):
    y=1-x
    z=1+y*(0.44325141463+y*(0.0626060122+y*(0.04757383546+y*0.01736506451))) \
    -(0.2499836831+y*(0.09200180037+y*(0.04069697526+y*0.00526449639)))*y*log(y)
    return z

def F(n,x):
    y=n; a=1.0; s=sin(x); s=s*s
    while(True):
        a=a*2.0/(1.0+y)
        s=0.5*(1.0 + y*s - sqrt((1.0-s)*(1.0 - y*y*s)))
        y=2.0*sqrt(y)/(1.0+y)
        if y>0.999999: break

    return a*log(tan(0.78539816339+0.5*arcsin(sqrt(s))))

def E(n,x):
    a=1; b=sqrt(1-n*n); z=0; n=1; t=tan(x)
    while(True):
        t=tan(x)
        t=(b/a-1)*t/(1+b*t*t/a)
        c=0.5*(a-b)
        s=0.5*(a+b)
        b=sqrt(a*b)
        a=s
        x=2*x+arctan(t)
        z=z+c*sin(x)
        n=2*n
        if c<1e-8: break

    return z+x*EE(n*n)/(a*n*KK(n*n))

def fplot2(RIPPLE, REJECTION, FPASS, FSTOP, FS, N, A, B):
    y=zeros(81,float)
    x=zeros(81,float)
    n=80
    x1=0.0001; x2=0.5*FS
    dx=(x2-x1)/n
    y1=-2*REJECTION; y2=2*RIPPLE
    #axes(x1, x2, y1, y2)
    for i in range(0,n+1):
        x[i]=x1+i*dx
        y[i]=db(x[i], FS, N, A, B)
    
    ax1 = fig.add_axes([0.1,0.6,0.8,0.35]) # left, bottom, width, height
    line1 = ax1.plot(x,y)
    ax1.set_xlabel('Frequency response' , fontsize=12)
    ax1.set_ylabel('Attenuation' , fontsize=12)
    #ax1.set_xlim((x1, x2))
    ax1.set_ylim((y1, y2))
    plt.setp(line1, color='green' , linewidth=2.0 )
    line_PASS = ax1.plot([FPASS,FPASS],[y1,y2])
    plt.setp(line_PASS, color='blue' , linewidth=1.0 )  # pass band
    line_STOP = ax1.plot([FSTOP,FSTOP],[y1,y2])
    plt.setp(line_STOP, color='red' , linewidth=1.0 )  # stop band
    line_RIPPLE1 = ax1.plot([x1,x2],[RIPPLE,RIPPLE])
    plt.setp(line_RIPPLE1, color='gray' , linewidth=1.0 )  # ripple band
    line_RIPPLE2 = ax1.plot([x1,x2],[-RIPPLE,-RIPPLE])
    plt.setp(line_RIPPLE2, color='gray' , linewidth=1.0 )  # ripple band
    line_REJ = ax1.plot([x1,x2],[-REJECTION,-REJECTION])
    plt.setp(line_REJ, color='red' , linewidth=1.0 )  # rejection band
    """
    line(FPASS, y1, FPASS, y2, 1)
    line(FSTOP, y1, FSTOP, y2, 1)
    line(x1,  RIPPLE, x2,  RIPPLE, 3)
    Y=-RIPPLE;    line(x1, Y, x2, Y, 3)
    Y=-REJECTION; line(x1, Y, x2, Y, 3)
    """

def db(f, fs, N, A, B):
    z=exp(-(2J)*pi*f/fs)
    B1=array(B)
    B1[0]=1.0
    t=A[N]
    b=B1[N]
    for i in range(1, N+1):
        t=t*z+A[N-i]
        b=b*z+B1[N-i]

    a=t/b
    return 10*log(abs(a)**2)/log(10.0)

def iresponse(N, A, B):
    y=zeros(81,float)
    n=80; y1=1e10; y2=-y1
    for t in range(N,n+1):
        Y=0
        for k in range(1,N+1): Y=Y-B[k]*y[t-k]
        if t<=N+N: Y=Y+A[t-N]
        y[t]=Y
        if y2<Y: y2=Y
        if y1>Y: y1=Y

    ax2 = fig.add_axes([0.1,0.1,0.8,0.35]) # left, bottom, width, height
    line2 = ax2.plot(y)
    ax2.set_xlabel('Impulse response' , fontsize=12)
    plt.setp(line2, color='blue' , linewidth=2.0 )
	

############################################################################

print("Digital Filter Design Program")

while(True):
    i=int(input("Low or High Pass Filter, Low=0, High=1\n"))
    if i==0: (N,A,B)=low()
    if i==1: (N,A,B)=high()
    if i==0 or i==1: break

output(N,A,B)
pause()
fplot2(RIPPLE, REJECTION, FPASS, FSTOP, FS, N, A, B)
pause()
iresponse(N, A, B)
plt.show()
