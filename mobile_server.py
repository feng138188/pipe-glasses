"""
手机浏览器实时检测服务
========================
无需安装任何 App，手机浏览器打开网页即可实时检测水管坡度/高度

使用方法:
  python mobile_server.py
  然后用手机浏览器打开: https://<PC的IP地址>:8765

依赖:
  pip install websockets opencv-python numpy
"""

import asyncio
import websockets
import json
import math
import time
import socket
import ssl
import http.server
import threading
import os
from pathlib import Path

import cv2
import numpy as np

from pipe_slope_detection import PipeSlopeDetector, AROverlay, IMUData

CERT_FILE = "/tmp/pipe_cert.pem"
KEY_FILE = "/tmp/pipe_key.pem"

# ============================================================
# 手机 IMU
# ============================================================
class MobileIMU:
    def __init__(self):
        self.pitch = 0.0
        self.roll = 0.0

    def update(self, pitch: float, roll: float):
        self.pitch = pitch
        self.roll = roll

    def get_data(self):
        return IMUData(pitch=self.pitch, roll=self.roll, yaw=0.0, timestamp=time.time())


# ============================================================
# HTML 页面
# ============================================================
HTML_PAGE = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1,user-scalable=no">
<meta name="apple-mobile-web-app-capable" content="yes">
<title>AI 管道施工眼镜</title>
<style>
* { margin:0; padding:0; box-sizing:border-box; }
html, body { width:100%; height:100%; overflow:hidden; background:#000; touch-action:none; }
body { display:flex; flex-direction:column; align-items:center; justify-content:center; }
#box { position:relative; width:100vw; }
#c { display:block; width:100%; image-rendering:pixelated; }
#p { position:absolute; top:0; left:0; z-index:10; background:rgba(0,0,0,0.75); color:#fff;
  padding:8px 14px; font:14px monospace; border-radius:0 0 10px 0; line-height:1.6; pointer-events:none; }
#bar { display:flex; gap:10px; padding:8px 12px; background:#222; width:100%; color:#aaa;
  font:13px monospace; justify-content:center; }
#ctl { display:flex; gap:6px; padding:8px; flex-wrap:wrap; justify-content:center;
  background:#111; width:100%; }
button { font-size:16px; padding:10px 16px; border:none; border-radius:8px; background:#333;
  color:#fff; min-width:44px; user-select:none; -webkit-user-select:none; }
button:active, button.on { background:#0a0; }
select { font-size:16px; padding:10px; border-radius:8px; background:#333; color:#fff; border:none; }
.v { color:#0f0; } .w { color:#ff0; } .e { color:#f44; }
</style>
</head>
<body>
<div id="box">
  <video id="vi" autoplay playsinline style="display:none"></video>
  <canvas id="ch" style="display:none"></canvas>
  <canvas id="c"></canvas>
  <div id="p">连接中...</div>
</div>
<div id="bar">
  <span>俯仰:<span class="v" id="sp">0°</span></span>
  <span>横滚:<span class="v" id="sr">0°</span></span>
  <span>FPS:<span class="v" id="sf">0</span></span>
</div>
<div id="ctl">
  <button id="bu">▲</button>
  <button id="bd">▼</button>
  <button id="bl">◀</button>
  <button id="br">▶</button>
  <button id="bz">归零</button>
  <select id="sel">
    <option value="50">DN50</option>
    <option value="100" selected>DN100</option>
    <option value="150">DN150</option>
    <option value="200">DN200</option>
    <option value="300">DN300</option>
  </select>
</div>

<script>
let ws, c=document.getElementById('c'), cx=c.getContext('2d');
let ch=document.getElementById('ch'), chx=ch.getContext('2d');
let vi=document.getElementById('vi'), p=document.getElementById('p');
let imu={pitch:0,roll:0}, streaming=false, rf=0, sf=0, t0=Date.now();

// 最新的检测结果
let last={d:false,a:0,p:0,h:0,c:0,x1:0,y1:0,x2:0,y2:0,ip:0,ir:0,dm:100};
let tapPt=null;

// WebSocket
function con(){
  let pr=location.protocol==='https:'?'wss://':'ws://';
  ws=new WebSocket(pr+location.hostname+':WSPORT/ws');
  ws.onopen=()=>{p.style.display='none'; startCam();};
  // 点击画布选择管道
c.addEventListener('click',(e)=>{
  let r=c.getBoundingClientRect();
  tapPt={x:(e.clientX-r.left)*(c.width/r.width), y:(e.clientY-r.top)*(c.height/r.height)};
  if(ws&&ws.readyState===1) ws.send(JSON.stringify({type:'tap',x:Math.round(tapPt.x),y:Math.round(tapPt.y)}));
  // 闪烁提示
  p.style.display='block'; p.innerHTML='已锁定'; setTimeout(()=>p.style.display='none',800);
});

ws.onmessage=(e)=>{
    try{
      let r=JSON.parse(e.data);
      if(r.t==='r'){
        last=r;
        rf++;
      }
    }catch(ex){}
  };
  ws.onclose=()=>{p.style.display='block'; p.innerHTML='断开，重连中...'; setTimeout(con,3000);};
  ws.onerror=()=>{setTimeout(con,3000);};
}

// 摄像头
async function startCam(){
  try{
    let s=await navigator.mediaDevices.getUserMedia({
      video:{facingMode:'environment',width:{ideal:640},height:{ideal:480}},audio:false});
    vi.srcObject=s;
    vi.onloadedmetadata=()=>{vi.play(); streaming=true; renderLoop();};
  }catch(e){
    p.innerHTML='摄像头错误: '+e.message;
  }
}

// 陀螺仪
if(typeof DeviceOrientationEvent!=='undefined'){
  window.addEventListener('deviceorientation',(e)=>{
    if(e.beta!==null)imu.pitch=e.beta.toFixed(1);
    if(e.gamma!==null)imu.roll=e.gamma.toFixed(1);
    document.getElementById('sp').textContent=imu.pitch+'°';
    document.getElementById('sr').textContent=imu.roll+'°';
    sendIMU();
  });
}

// 渲染循环：显示摄像头画面 + 画 AR 叠加
function renderLoop(){
  if(!streaming)return;
  let vw=vi.videoWidth||640, vh=vi.videoHeight||480;

  // 同尺寸隐藏 canvas 用于发送帧
  ch.width=vw; ch.height=vh;
  chx.drawImage(vi,0,0,vw,vh);

  // 显示 canvas
  c.width=vw; c.height=vh;
  cx.drawImage(vi,0,0,vw,vh);

  // 画 AR 叠加
  if(last.d){
    let sw=c.width/vw, sh=c.height/vh;  // CSS 缩放（实际就是1）
    let x1=last.x1, y1=last.y1, x2=last.x2, y2=last.y2;

    // 水管检测线
    cx.strokeStyle='#0f0';
    cx.lineWidth=3;
    cx.beginPath();
    cx.moveTo(x1,y1);
    cx.lineTo(x2,y2);
    cx.stroke();

    // 端点圆
    cx.fillStyle='#0f0';
    cx.beginPath(); cx.arc(x1,y1,6,0,Math.PI*2); cx.fill();
    cx.beginPath(); cx.arc(x2,y2,6,0,Math.PI*2); cx.fill();

    // 水平参考虚线
    let my=(y1+y2)/2;
    cx.strokeStyle='rgba(150,150,150,0.6)';
    cx.lineWidth=1;
    cx.setLineDash([8,8]);
    cx.beginPath(); cx.moveTo(0,my); cx.lineTo(vw,my); cx.stroke();
    cx.setLineDash([]);
  }

  // 信息面板（仅在检测到水管时显示）
  if(last.d){
    let px=10, py=10, pw=260, ph=160;
    cx.fillStyle='rgba(0,0,0,0.65)';
    cx.fillRect(px,py,pw,ph);

    let sy=py+22, lh=28;
    cx.font='bold 15px monospace';

    let absA=Math.abs(last.a);
    if(absA>=2&&absA<=5) cx.fillStyle='#0f0'; else if(absA<1.6||absA>6) cx.fillStyle='#f44'; else cx.fillStyle='#ff0';
    cx.fillText(`坡度: ${last.a}° (${last.p}%)`, px+10, sy); sy+=lh;

    cx.fillStyle='#0f0';
    cx.fillText(`高度: ${last.h.toFixed(2)}m (DN${last.dm})`, px+10, sy); sy+=lh;

    cx.fillStyle='#aaa'; cx.font='12px monospace';
    cx.fillText(`置信度: ${Math.round(last.c*100)}%  pitch:${last.ip}° roll:${last.ir}°`, px+10, sy); sy+=lh;

    cx.font='bold 16px monospace';
    if(absA>=2&&absA<=5) {cx.fillStyle='#0f0'; cx.fillText('PASS',px+10,sy);}
    else {cx.fillStyle='#f44'; cx.fillText('需调整',px+10,sy);}
  }

  // 底部目标范围
  cx.fillStyle='rgba(150,150,150,0.7)'; cx.font='11px monospace';
  cx.fillText('目标: 坡度2-5° | 高度0.5-1.5m', 10, c.height-10);

  // FPS（用结果帧计数）
  let now=Date.now();
  if(now-t0>=1000){document.getElementById('sf').textContent=Math.round(rf*1000/(now-t0)); rf=0; t0=now;}

  // 每 N 帧把 tap 坐标也发过去（维持锁定）
  if(tapPt && sf%10===0 && ws&&ws.readyState===1){
    ws.send(JSON.stringify({type:'tap',x:Math.round(tapPt.x),y:Math.round(tapPt.y)}));
  }

  // 每次都发送帧给服务器（独立于结果接收）
  sf++;
  if(sf % 3 === 0){  // 每3帧发一次，减少带宽
    ch.toBlob((b)=>{if(b&&ws&&ws.readyState===1)ws.send(b);},'image/jpeg',0.45);
  }

  requestAnimationFrame(renderLoop);
}

function sendIMU(){
  if(ws&&ws.readyState===1)ws.send(JSON.stringify({type:'imu',pitch:Number(imu.pitch),roll:Number(imu.roll)}));
}

// 按钮
let btns={bu:{p:0,r:0},bd:{p:25,r:0},bl:{p:0,r:-15},br:{p:0,r:15}};
Object.keys(btns).forEach(id=>{
  let b=document.getElementById(id);
  b.addEventListener('pointerdown',()=>{b.classList.add('on'); imu.pitch=btns[id].p; imu.roll=btns[id].r; sendIMU();});
  b.addEventListener('pointerup',()=>{b.classList.remove('on'); imu.pitch=0; imu.roll=0; sendIMU();});
  b.addEventListener('pointerleave',()=>{b.classList.remove('on'); imu.pitch=0; imu.roll=0; sendIMU();});
});
document.getElementById('bz').addEventListener('click',()=>{imu.pitch=0; imu.roll=0; sendIMU();});
document.getElementById('sel').addEventListener('change',()=>{
  if(ws&&ws.readyState===1)ws.send(JSON.stringify({type:'config',diameter:parseInt(document.getElementById('sel').value)}));
});

con();
</script>
</body>
</html>"""


# ============================================================
# 网络工具
# ============================================================
def get_all_ips():
    """获取本机所有非回环 IPv4 地址"""
    import fcntl
    import struct
    ips = []
    for iface in socket.if_nameindex():
        _, name = iface
        if name == "lo":
            continue
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            ip = socket.inet_ntoa(
                fcntl.ioctl(s.fileno(), 0x8915,
                    struct.pack("256s", name[:15].encode("utf-8")))[20:24])
            s.close()
            if ip != "127.0.0.1":
                ips.append((name, ip))
        except Exception:
            pass
    return ips


# ============================================================
# HTTPS 服务器
# ============================================================
class HTTPHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/", "/index.html"):
            html = HTML_PAGE.replace("WSPORT", "8766")
            body = html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass


def start_https_server(port=8765):
    """启动 HTTPS 服务器（独立线程）"""
    server = http.server.HTTPServer(("0.0.0.0", port), HTTPHandler)
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(CERT_FILE, KEY_FILE)
    server.socket = ctx.wrap_socket(server.socket, server_side=True)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server


# ============================================================
# WebSocket 处理
# ============================================================
async def handle_connection(websocket):
    detector = PipeSlopeDetector(pipe_diameter_mm=100, use_dino=True)
    detector.imu = MobileIMU()
    overlay = AROverlay()
    overlay.set_targets(2.0, 5.0, 0.5, 1.5)

    peer = websocket.remote_address
    print(f"手机已连接 ✓ ({peer[0]})")

    fc = 0
    t0 = time.time()
    try:
        async for message in websocket:
            if isinstance(message, bytes):
                fc += 1
                if fc == 1:
                    print(f"  收到首帧 ({len(message)} bytes)")

                nparr = np.frombuffer(message, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if frame is None:
                    print(f"  帧解码失败")
                    continue

                result = detector.detect(frame)
                # 只回传检测数据，客户端自己画叠加。所有值转 Python 原生类型
                await websocket.send(json.dumps({
                    "t": "r",
                    "d": bool(result.detected),
                    "a": round(float(result.slope_angle), 1),
                    "p": round(float(result.slope_percent), 1),
                    "h": round(float(result.height), 2),
                    "c": round(float(result.confidence), 2),
                    "x1": int(result.line_start[0]), "y1": int(result.line_start[1]),
                    "x2": int(result.line_end[0]), "y2": int(result.line_end[1]),
                    "ip": round(float(result.imu_pitch), 1),
                    "ir": round(float(result.imu_roll), 1),
                    "dm": int(detector.pipe_diameter_mm),
                }))

                if fc % 30 == 0:
                    elapsed = time.time() - t0
                    print(f"  已处理 {fc} 帧, {fc/elapsed:.0f} fps")

            elif isinstance(message, str):
                data = json.loads(message)
                t = data.get("type", "")
                if t == "imu":
                    detector.imu.update(data.get("pitch", 0), data.get("roll", 0))
                elif t == "config":
                    detector.set_pipe_diameter(data.get("diameter", 100))
                elif t == "tap":
                    detector.tap_point = (data.get("x", 320), data.get("y", 240))
                    detector.track_lost = 0  # 重置跟踪
                    print(f"  点击锁定: ({data.get('x')}, {data.get('y')})")

    except websockets.exceptions.ConnectionClosed:
        print(f"手机断开连接 (共处理 {fc} 帧)")
    except Exception as e:
        print(f"处理错误: {e}")


# ============================================================
# 主入口
# ============================================================
async def main():
    host = "0.0.0.0"
    http_port = 8765
    ws_port = 8766

    # 检查证书
    for f, name in [(CERT_FILE, "证书"), (KEY_FILE, "密钥")]:
        if not os.path.exists(f):
            print(f"错误: {name}文件不存在: {f}")
            print("请先运行: openssl req -x509 -newkey rsa:2048 -keyout /tmp/pipe_key.pem -out /tmp/pipe_cert.pem -days 365 -nodes -subj '/CN=pipe'")
            return

    ips = get_all_ips()

    # 启动 HTTPS 服务器
    httpsd = start_https_server(http_port)

    print("=" * 55)
    print("  AI 管道施工眼镜 - 手机实时检测 (HTTPS)")
    print("=" * 55)
    print(f"\n  用手机浏览器打开:\n")
    if ips:
        for name, ip in ips:
            print(f"  ▶ https://{ip}:{http_port}   ({name})")
    else:
        print(f"  ▶ https://127.0.0.1:{http_port}")
    print()
    print("  注意: 自签名证书会提示'不安全'，点「高级→继续访问」即可")
    print()
    print("  连接方式:")
    print(f"  A. 同一局域网 — 手机和 PC 在同一路由器下")
    print(f"  B. USB 共享  — 手机插 USB，开启 USB 网络共享")
    print()
    print("  Ctrl+C 退出")
    print("-" * 55)

    # WSS 服务器
    ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ssl_ctx.load_cert_chain(CERT_FILE, KEY_FILE)

    server = await websockets.serve(
        handle_connection, host, ws_port,
        ssl=ssl_ctx,
        max_size=2 * 1024 * 1024,
    )

    try:
        await server.wait_closed()
    except asyncio.CancelledError:
        pass


if __name__ == "__main__":
    asyncio.run(main())
