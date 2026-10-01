import numpy as np, subprocess, math
W,H,N,FPS=640,360,90,15
p=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}','-r',str(FPS),'-i','-','-pix_fmt','yuv420p','-c:v','libx264','prova.mp4'],stdin=subprocess.PIPE)
yy,xx=np.mgrid[0:H,0:W]
for i in range(N):
    off=60*math.sin(i/N*2*math.pi)          # il corridoio si sposta a destra e a sinistra
    t=np.clip((yy-100)/(H-100),0,1)
    cx=W/2+off*(0.3+0.7*t)
    half=40+170*t
    corr=(yy>=100)&(np.abs(xx-cx)<half)
    f=np.zeros((H,W,3),np.uint8); f[:]=(140,100,70); f[:100]=(235,200,190)
    f[corr]=(60,170,50)
    p.stdin.write(f.tobytes())
p.stdin.close(); p.wait()
