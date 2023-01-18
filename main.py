import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sounddevice as sd
from scipy.io import wavfile
from scipy import signal
# N=0, Q=0,  R=1, P=3
#𝑥[𝑛] = (n+1)*(𝑢[𝑛 + 2] − 𝑢[𝑛 − 3])


#Zadatak 1
start_n=-2
d_n=1
end_n=3
n=np.arange(start_n, end_n, d_n)
x =(n+1)*((n>=-2)*(n<3))


plt.figure()
plt.stem(n,x)
plt.xlabel('n')
plt.ylabel('x[n]')
plt.title('Originalni signal: x[n] = (n+1)*(u[n + 2] − u[n − 3])')
plt.grid(b=True,which="both",color='grey',linestyle='--')

plt.show()


#y1[𝑛] = 𝑥[3(𝑄 + 1) − (𝑅 + 1)𝑛] 
#y1[n]=x[3-2n]

#Pomeranje za 3:
n0=3

y1 = np.concatenate((x,np.zeros(n0))) 
n1 = np.concatenate((np.arange(start_n-n0, start_n, d_n),n))
plt.figure()
plt.stem(n1,y1)
plt.xlabel('n1')
plt.ylabel('y1[n]')
plt.title('Pomeren signal: y1[n]=x[n+3]')
plt.grid(b=True,which="both",color='grey',linestyle='--')

plt.show()

#Inverzij:
y2 = y1[::-1]
n2=-n1[::-1]
plt.figure()
plt.stem(n2,y2)
plt.xlabel('n2')
plt.ylabel('y2[n]')
plt.grid(b=True,which="both",color='grey',linestyle='--')

plt.title('Invertovan signal: y1[n]=x[-n+3]')
plt.show()

#Skaliranje

t=np.arange(0,len(n2),1)
skaliranje=2

y3 = y2[::skaliranje] # ubrzavanje
n3 = n2[::skaliranje]/skaliranje
plt.figure()
plt.figure()
plt.stem(n3,y3)
plt.xlabel('n2')
plt.grid(b=True,which="both",color='grey',linestyle='--')

plt.ylabel('y2[n]')
plt.title('Skaliran signal: y[n]=x[3-2n]')
plt.show()


#𝑦2[𝑛] = 𝑥[−2(𝑄 + 1) + 𝑛/2].
#y2[n]= x[-2*1+n/2]
#y2[n]=x[-2+n/2] =x[n/2-2]
#%%
#P: 
n0=-2

y1 = np.concatenate((np.zeros(-n0),x)) 
n1 = np.concatenate((n,np.arange(n[-1]+d_n, n[-1]-n0+d_n, d_n)))
plt.figure()
plt.stem(n1,y1)
plt.xlabel('n1')
plt.grid(b=True,which="both",color='grey',linestyle='--')

plt.ylabel('y1[n]')
plt.title('Pomeren signal: y[n]=x[n-2]')
plt.show()
#%%

#S:
skaliranje=1/2  
N = len(y1)
n2=np.arange(n1[0]/skaliranje, n1[-1]/skaliranje+d_n, d_n)
y2 = np.interp(n2, n1/skaliranje, y1)
plt.figure()
plt.stem(n2,y2)
plt.xlabel('n2')
plt.grid(b=True,which="both",color='grey',linestyle='--')

plt.ylabel('y2[n]')
plt.title('Skaliran signal: y[n]=x[n/2-2]')
plt.show()

 #%%

# snimanje i analiza zvucnog signala
  
duration = 3  # trajanje snimka
samplerate=8000  # ucestanost odabiranja

myrecording=sd.rec(int(duration*samplerate), samplerate=samplerate,channels=1)
sd.wait()
wavfile.write('./sekvenca.wav', samplerate, myrecording)

samplerate, data=wavfile.read('./sekvenca.wav')

dt=1/samplerate
t=np.arange(0,dt*len(data),dt)
chanel1=data
plt.figure()
plt.stem(t,chanel1, markerfmt=" ")
plt.xlabel('t')
plt.ylabel('y[t]')
plt.title('Audio signal')
plt.show()

sd.play(data,samplerate)
#%%
#n/2

skaliranje=1/2  
t2=np.arange(t[0]/skaliranje, t[-1]/skaliranje+dt, dt)
y2 = np.interp(t2, t/skaliranje, data)
plt.figure()
plt.stem(t2,y2, markerfmt=" ")
plt.xlabel('t2')
plt.grid(b=True,which="both",color='grey',linestyle='--')

plt.ylabel('y2[t]')
plt.title('Skaliran signal (1/2)')
plt.show()

scaled = np.int16(y2/ np.max (np.abs (y2)) * 32767)
sd.play(scaled, samplerate)


# =============================================================================
# samplerate, data = wavfile.read('./mic.wav')
# samplerate = 22050
# dt=1/samplerate
# t=np.arange(0,dt*len(data),dt)
# chanel1=data # chanel1=data[:,1] ako ima dva kanala
# plt.figure()
# plt.stem(t,chanel1, markerfmt=" ")
# plt.xlabel('t')
# plt.ylabel('y(t)')
# plt.title('Audio signal')
# plt.show()
# 
# sd.play(data,samplerate)
# 
# =============================================================================
#%%
# =============================================================================
# samplerate, data = wavfile.read('./mic.wav')
# samplerate = 88200
# dt=1/samplerate
# t=np.arange(0,dt*len(data),dt)
# chanel1=data # chanel1=data[:,1] ako ima dva kanala
# plt.figure()
# plt.stem(t,chanel1, markerfmt=" ")
# plt.xlabel('t')
# plt.ylabel('y(t)')
# plt.title('Audio signal')
# plt.show()
# 
# sd.play(data,samplerate)
# =============================================================================

#2n

samplerate, data = wavfile.read('./sekvenca.wav')
samplerate=8000
dt=1/samplerate
t=np.arange(0,dt*len(data),dt)
skaliranje=2

y3 = data[1:-1:skaliranje] # ubrzavanje, da li od 0 ili -1
t3 = t[1:-1:skaliranje]/skaliranje
plt.figure()
plt.stem(t3,y3, markerfmt=" ")
plt.xlabel('t')
plt.ylabel('y[t]')
plt.title('Skalirani signal (2)')
plt.show()
sd.play(y3,samplerate)


samplerate, data = wavfile.read('./sekvenca.wav')
samplerate = 4000
dt=1/samplerate
t=np.arange(0,dt*len(data),dt)
chanel1=data # chanel1=data[:,1] ako ima dva kanala
plt.figure()
plt.stem(t,chanel1, markerfmt=" ")
plt.xlabel('t')
plt.ylabel('y(t)')
plt.title('Audio signal 4kHZ')
plt.show()

sd.play(data,samplerate)
#%% generisati frekvencija 16kHz
samplerate, data = wavfile.read('./sekvenca.wav')
samplerate = 16000
dt=1/samplerate
t=np.arange(0,dt*len(data),dt)
chanel1=data # chanel1=data[:,1] ako ima dva kanala
plt.figure()
plt.stem(t,chanel1, markerfmt=" ")
plt.xlabel('t')
plt.ylabel('y(t)')
plt.title('Audio signal 16kHZ')
plt.show()
sd.play(data,samplerate)
#%% zakasniti signal za 2 sekunde

samplerate, data = wavfile.read('./sekvenca.wav')
samplerate=8000
dt=1/samplerate
t=np.arange(0,dt*len(data),dt)
t1=np.concatenate((t,np.arange(t[-1]+dt, t[-1]+dt+2, dt)))
data1=np.concatenate((np.zeros(samplerate*2),data))
plt.figure()
plt.stem(t1,data1, markerfmt=" ")
plt.xlabel('t')
plt.ylabel('y(t)')
plt.title('Audio signal pomeren za 2s')
plt.show()

scaled = np.int16(data1 / np.max (np.abs (data1)) * 32767)
sd.play(scaled, samplerate)

#%% 

data2 = data[::-1]


plt.figure()
plt.stem(t,data2, markerfmt=" ")
plt.xlabel('t')
plt.ylabel('y(t)')
plt.title('Audio signal invertovan')
plt.show()

sd.play(data2,samplerate)

#%%

duration = 3  # trajanje snimka
samplerate=8000  # ucestanost odabiranja

myrecording=sd.rec(int(duration*samplerate), samplerate=samplerate,channels=1)
sd.wait()
wavfile.write('./palindrom.wav', samplerate,myrecording)

samplerate, data=wavfile.read('./palindrom.wav')

dt=1/samplerate
t=np.arange(0,dt*len(data),dt)

chanel1=data 
plt.figure()
plt.stem(t,chanel1, markerfmt=" ")
plt.xlabel('t')
plt.ylabel('y(t)')
plt.title('Audio signal palindrom')
plt.show()

sd.play(data,samplerate)
data2 = data[::-1]
plt.figure()
plt.stem(t,data2, markerfmt=" ")
plt.xlabel('t')
plt.ylabel('y(t)')
plt.title('Audio signal palindrom invertovan')
plt.show()

sd.play(data2,samplerate)

#%%
#Zadatak 2
#𝑃 = 3 

#𝑥[𝑛] = (n+1)(𝑢[𝑛 + 2] − 𝑢[𝑛 − 3]) n=0



#ℎ[𝑛]=𝑢[𝑛+2]−𝑢[𝑛−2]

start_h=-2
d_n=1
end_n=2
n=np.arange(start_n, end_n, d_n)
h =((n>=-2)*(n<2))

plt.figure()
plt.stem(n,h)
plt.xlabel('n')
plt.ylabel('h[n]')
plt.title('Originalni signal: u[n + 2] - u[n-2]')
plt.grid(b=True,which="both",color='grey',linestyle='--')

#%%

start_n=-2
d_n=1
end_n=3
n=np.arange(start_n, end_n, d_n)
x =(n+1)*((n>=-2)*(n<3))


plt.figure()
plt.stem(n,x)
plt.xlabel('n')
plt.ylabel('x[n]')
plt.title('Originalni signal: x[n] = (n+1)*(u[n + 2] − u[n − 3])')
plt.grid(b=True,which="both",color='grey',linestyle='--')

plt.show()

#%%
odziv=np.convolve(h,x)
#odziv=odziv/max(np.absolute(odziv))
no=np.arange(-4,4,1)


plt.figure()
plt.stem(no,odziv,markerfmt=" ")
plt.xlabel('n')
plt.ylabel('y[n]')
plt.title('Konvolucija')
plt.grid(b=True,which="both",color='grey',linestyle='--')
plt.show()

#%% 
#konvolucija zvucni signal Snimiti govornu sekvencu trajanja 3 sekunde sa učestanošću
#odabiranja 𝑓𝑠 = 48kHz.



duration = 3  # trajanje snimka
samplerate=48000  # ucestanost odabiranja

myrecording=sd.rec(int(duration*samplerate), samplerate=samplerate,channels=1)
sd.wait()
wavfile.write('./sekvenca2.wav', samplerate,myrecording)

samplerate, data=wavfile.read('./sekvenca2.wav')

dt=1/samplerate
t=np.arange(0,dt*len(data),dt)
chanel1=data
plt.figure()
plt.stem(t,chanel1, markerfmt=" ")
plt.xlabel('t')
plt.ylabel('y[t]')
plt.title('Audio signal')
plt.show()

sd.play(data,samplerate)

samplerate_impulsni, impulsni_odziv = wavfile.read('\elveden_hall_impulse_response.wav')
impulsni_odziv=impulsni_odziv/max(np.absolute(impulsni_odziv))
dt_i=1/samplerate_impulsni
t_i=np.arange(0,dt_i*len(impulsni_odziv),dt_i)
plt.figure()
plt.plot(t_i,impulsni_odziv)
plt.xlabel('t')
plt.ylabel('y(t)')
plt.title('Impulsni odziv-Elveden Hall - Suffolk')
plt.show()

odziv=np.convolve(data,impulsni_odziv)
odziv=odziv/max(np.absolute(odziv))
dt_o=1/samplerate_impulsni
t_o=np.arange(0,dt_o*len(odziv),dt_o)
plt.figure()
plt.plot(t_o,odziv)
plt.xlabel('t')
plt.ylabel('y(t)')
plt.title('Konvolucija-Elveden Hall')
plt.show()
scaled = np.int16(odziv / np.max (np.abs (odziv)) * 32767)
wavfile.write('konv_ElvedenHall.wav',samplerate, scaled.astype(np.float32))
sd.play(scaled, samplerate)

#%% 

samplerate_impulsni, impulsni_odziv = wavfile.read('\central_hall_impulse_response.wav')
impulsni_odziv=impulsni_odziv/max(np.absolute(impulsni_odziv))
dt_i=1/samplerate_impulsni
t_i=np.arange(0,dt_i*len(impulsni_odziv),dt_i)
plt.figure()
plt.plot(t_i,impulsni_odziv)
plt.xlabel('t')
plt.ylabel('y(t)')
plt.title('Impulsni odziv-Central Hall')
plt.show()

odziv=np.convolve(data,impulsni_odziv)
odziv=odziv/max(np.absolute(odziv))
dt_o=1/samplerate_impulsni
t_o=np.arange(0,dt_o*len(odziv),dt_o)
plt.figure()
plt.plot(t_o,odziv)
plt.xlabel('t')
plt.ylabel('y(t)')
plt.title('Konvolucija-Central Hall')
plt.show()
scaled = np.int16(odziv / np.max (np.abs (odziv)) * 32767)
wavfile.write('konv_CentralHall.wav',samplerate, scaled.astype(np.float32))
sd.play(scaled, samplerate)
#%%
samplerate_impulsni, impulsni_odziv = wavfile.read('\koli_national_park_impulse_response.wav')
impulsni_odziv=impulsni_odziv/max(np.absolute(impulsni_odziv))
dt_i=1/samplerate_impulsni
t_i=np.arange(0,dt_i*len(impulsni_odziv),dt_i)
plt.figure()
plt.plot(t_i,impulsni_odziv)
plt.xlabel('t')
plt.ylabel('y(t)')
plt.title('konv-National Park')
plt.show()

odziv=np.convolve(data,impulsni_odziv)
odziv=odziv/max(np.absolute(odziv))
dt_o=1/samplerate_impulsni
t_o=np.arange(0,dt_o*len(odziv),dt_o)
plt.figure()
plt.plot(t_o,odziv)
plt.xlabel('t')
plt.ylabel('y(t)')
plt.title('konv-National Park')
plt.show()
scaled = np.int16(odziv / np.max (np.abs (odziv)) * 32767)
wavfile.write('kon_NationalPark.wav',samplerate, scaled.astype(np.float32))
sd.play(scaled, samplerate)

#%%

samplerate_impulsni, impulsni_odziv = wavfile.read('\mine_impulse_response.wav')
impulsni_odziv=impulsni_odziv/max(np.absolute(impulsni_odziv))
dt_i=1/samplerate_impulsni
t_i=np.arange(0,dt_i*len(impulsni_odziv),dt_i)
plt.figure()
plt.plot(t_i,impulsni_odziv)
plt.xlabel('t')
plt.ylabel('y(t)')
plt.title('Impulsni odziv-Mine')
plt.show()

odziv=np.convolve(data,impulsni_odziv)
odziv=odziv/max(np.absolute(odziv))
dt_o=1/samplerate_impulsni
t_o=np.arange(0,dt_o*len(odziv),dt_o)
plt.figure()
plt.plot(t_o,odziv)
plt.xlabel('t')
plt.ylabel('y(t)')
plt.title('Konvolucija-Mine')
plt.show()
scaled = np.int16(odziv / np.max (np.abs (odziv)) * 32767)
wavfile.write('konv_Mine.wav',samplerate, scaled.astype(np.float32))
sd.play(scaled, samplerate)



#%%




#%%

#SLIKA


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

img = mpimg.imread('\Test_slika.png')
img = rgb2gray(img)
plt.figure()
imgplot = plt.imshow(img, cmap=plt.get_cmap('gray'))
plt.show()


M1=np.array([[1/9, 1/9, 1/9],[1/9, 1/9, 1/9],[1/9, 1/9, 1/9]])
img_m0=signal.convolve2d(img, M1, mode='same').clip(0,1)
plt.figure()
imgplot = plt.imshow(img_m0, cmap=plt.get_cmap('gray'))
plt.show()

M5=np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
img_sharp=signal.convolve2d(img,M5, mode='same').clip(0,1)
plt.figure()
imgplot = plt.imshow(img_sharp, cmap=plt.get_cmap('gray'))
plt.show()
#%%

M6=np.array([[0,0,0,0,0,1/5],[0,0,0,0,1/5,0],[0,0,0,1/5,0,0], [0,0,1/5,0,0,0], [0,1/5,0,0,0,0], [1/5,0,0,0,0,0]])
img_sharp=signal.convolve2d(img, M6, mode='same').clip(0,1)
plt.figure()
imgplot = plt.imshow(img_sharp, cmap=plt.get_cmap('gray'))
plt.show()

#%%

M9=np.array([[-1, -2, -1],[-0, 0, 0],[1, 2, 1]])
img_sharp=signal.convolve2d(img,M9, mode='same').clip(0,1)
plt.figure()
imgplot = plt.imshow(img_sharp, cmap=plt.get_cmap('gray'))
plt.show()

#%%

#Furijeov red

def u(t):
    return 0 if t<0 else 1

def v(L):
    def vk(t):
        s=0
        for k in range(-L,L+1):
            s += 2*( 2*k-t+1) * ( u ( t - 2 * k ) - u ( t - 2 * k -1))
        return s
    return vk

dt = 0.005

data = np . arange ( 0 , 7 + dt , dt )
values = np . vectorize ( v (9) , otypes =[ float ]) ( data )
start_t=-4
d_t=0.01
end_t=4
t=np.arange(start_t, end_t, d_t)
plt.figure()
plt.plot(data,values)
plt.xlabel('t')
plt.ylabel('v(t)')
plt.ylim([0,3])
plt.title('Originalni signal')
plt.show()

        #%% 
#T=pi
#w0 = 2pi/t = 2
N = 50

pi=np.pi
w0 = pi
k=np.arange(-N,N+1)

ak = (np.exp(-1j*k*pi) + 1j*k*pi-1)/(k*k*pi*pi*1j*1j)
ak[N]=0.5

plt.figure()
plt.stem(k, np.absolute(ak))
plt.xlabel('k')
plt.title('Amplitudski linijski spektar')
plt.show()


plt.figure()
plt.stem(k, np.angle(ak))
plt.xlabel('k')
plt.title('Fazni linijski spektar')
plt.show()



#MISLIM DA JE KOD OK SAMO DA SE NAPISE V koje je oblik funkcije
#nisam stigla da ispisem :( )

t=np.arange(-2,2,0.01)

v2=np.zeros(len(t))
for i in range(len(t)):
    v2[i]=np.absolute(ak[N])
    for k in range(1,N+1):
        v2[i] = v2[i] + 2*np.absolute(ak[k+N])*np.cos((k)*w0*t[i]+np.angle(ak[k+N]))   
plt.figure()        
plt.plot(t,V)
plt.ylim([0,2])        
plt.title('K=1');
plt.plot(t,v2)
plt.show()
#%%
N = 2
pi=np.pi
w0 = 2
k=np.arange(-N,N+1)
A=1j*2*k
Ak= -4*k*k
ak= 1/pi * ( ((Ak-A+1)*np.exp(-pi*A/2)-np.exp(-pi*A))/(A*Ak+A) + (1-A*np.exp(-pi/2*A))/(Ak+1)   )


ak[N]=0.5# a0=0.5 (srednja vrednost signala - DC komponenta)
t=np.arange(-2,2,0.01)

v2=np.zeros(len(t))
for i in range(len(t)):
    v2[i]=np.absolute(ak[N])
    for k in range(1,N+1):
        v2[i] = v2[i] + 2*np.absolute(ak[k+N])*np.cos((k)*w0*t[i]+np.angle(ak[k+N]))   
plt.figure()        
plt.plot(t,V)
plt.ylim([0,2])      
plt.title('K=2');  
plt.plot(t,v2)
plt.show()
#%%
N = 5
pi=np.pi
w0 = 2
k=np.arange(-N,N+1)
A=1j*2*k
Ak= -4*k*k
ak= 1/pi * ( ((Ak-A+1)*np.exp(-pi*A/2)-np.exp(-pi*A))/(A*Ak+A) + (1-A*np.exp(-pi/2*A))/(Ak+1)   )

ak[N]=0.5# a0=0.5 (srednja vrednost signala - DC komponenta)
t=np.arange(-2,2,0.01)

v2=np.zeros(len(t))
for i in range(len(t)):
    v2[i]=np.absolute(ak[N])
    for k in range(1,N+1):
        v2[i] = v2[i] + 2*np.absolute(ak[k+N])*np.cos((k)*w0*t[i]+np.angle(ak[k+N]))   
plt.figure()        
plt.plot(t,V)
plt.ylim([0,2])   
plt.title('K=5');     
plt.plot(t,v2)
plt.show()
#%%
N = 10
pi=np.pi
w0 = 2
k=np.arange(-N,N+1)
A=1j*2*k
Ak= -4*k*k
ak= 1/pi * ( ((Ak-A+1)*np.exp(-pi*A/2)-np.exp(-pi*A))/(A*Ak+A) + (1-A*np.exp(-pi/2*A))/(Ak+1)   )


ak[N]=0.5# a0=0.5 (srednja vrednost signala - DC komponenta)
t=np.arange(-2,2,0.01)

v2=np.zeros(len(t))
for i in range(len(t)):
    v2[i]=np.absolute(ak[N])
    for k in range(1,N+1):
        v2[i] = v2[i] + 2*np.absolute(ak[k+N])*np.cos((k)*w0*t[i]+np.angle(ak[k+N]))   
plt.figure()        
plt.plot(t,V)
plt.title('K=10');
plt.ylim([0,2])        
plt.plot(t,v2)
plt.show()
              #%%
N = 50
pi=np.pi
w0 = 2
k=np.arange(-N,N+1)
A=1j*2*k
Ak= -4*k*k
ak= 1/pi * ( ((Ak-A+1)*np.exp(-pi*A/2)-np.exp(-pi*A))/(A*Ak+A) + (1-A*np.exp(-pi/2*A))/(Ak+1)   )

ak[N]=0.5# a0=0.5 (srednja vrednost signala - DC komponenta)
t=np.arange(-2,2,0.01)

v2=np.zeros(len(t))
for i in range(len(t)):
    v2[i]=np.absolute(ak[N])
    for k in range(1,N+1):
        v2[i] = v2[i] + 2*np.absolute(ak[k+N])*np.cos((k)*w0*t[i]+np.angle(ak[k+N]))   
plt.figure()        
plt.plot(t,V)
plt.title('K=50');
plt.ylim([0,2])        
plt.plot(t,v2)
plt.show() 
