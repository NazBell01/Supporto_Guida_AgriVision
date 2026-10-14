"""Video sintetico di stress per le linee guida: la camera oscilla (zig zag), il sentiero cambia curvatura e
sul bordo compaiono ostacoli (fogliame) che tagliano la maschera. Serve a cercare i casi in cui le curve
si incrociano o saltano. Scrive zigzag.webm nella cartella corrente.  Uso: python3 video_zigzag.py"""
import numpy as np, subprocess, math
W, H, N, FPS = 640, 360, 120, 15
rng = np.random.default_rng(11)
p = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS),
                      '-i', '-', '-pix_fmt', 'yuv420p', '-c:v', 'libvpx', '-b:v', '2M', 'zigzag.webm'], stdin=subprocess.PIPE)
yy, xx = np.mgrid[0:H, 0:W]
for i in range(N):
    t = np.clip((yy - 100) / (H - 100), 0, 1)
    yaw = 120 * math.sin(2 * math.pi * i / 45)                   # la camera ruota: tutto il sentiero scivola di lato
    curv = 150 * math.sin(2 * math.pi * i / 70) * (1 - t) ** 1.8  # il sentiero cambia curvatura
    cx = W / 2 + yaw * (0.35 + 0.65 * (1 - t)) + curv
    half = 30 + 175 * t
    corr = (yy >= 100) & (np.abs(xx - cx) < half)
    f = np.zeros((H, W, 3), np.uint8); f[:] = (140, 100, 70); f[:100] = (235, 200, 190); f[corr] = (60, 170, 50)
    # fogliame che invade il sentiero da un lato, a tratti, e a diverse altezze
    for k in range(3):
        if (i // 8 + k) % 3 == 0:
            y0 = 110 + 55 * k; lato = -1 if (i // 8 + k) % 2 else 1
            xb = (cx + lato * half)[y0:y0 + 40, 0].mean() if False else cx[y0, 0] + lato * half[y0, 0]
            xa, xe = sorted((int(xb), int(xb - lato * (40 + 60 * rng.random()))))
            f[y0:y0 + 38, max(0, xa):max(0, xe)] = (140, 100, 70)
    p.stdin.write(f.tobytes())
p.stdin.close(); p.wait()
