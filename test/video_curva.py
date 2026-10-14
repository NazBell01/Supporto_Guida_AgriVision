import numpy as np, subprocess, math
W,H,N,FPS=640,360,60,15
p=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}','-r',str(FPS),'-i','-','-pix_fmt','yuv420p','-c:v','libvpx','-b:v','1M','curva.webm'],stdin=subprocess.PIPE)
yy,xx=np.mgrid[0:H,0:W]
for i in range(N):
    t=np.clip((yy-100)/(H-100),0,1)           # 0 all'orizzonte, 1 in basso
    curv=140*(1-t)**1.8*(1 if i<30 else -1)   # il sentiero piega a destra, poi a sinistra
    cx=W/2+curv+10*math.sin(i/9)
    half=34+170*t
    corr=(yy>=100)&(np.abs(xx-cx)<half)
    f=np.zeros((H,W,3),np.uint8); f[:]=(140,100,70); f[:100]=(235,200,190); f[corr]=(60,170,50)
    p.stdin.write(f.tobytes())
p.stdin.close(); p.wait()
