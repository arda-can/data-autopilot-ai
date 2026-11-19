"""README için demo GIF'i ve ekran görüntüleri üretir (geliştirici aracı; uygulamanın parçası değildir).

Kullanım (proje kökünden):  .venv\\Scripts\\python.exe scripts\\demo_capture.py

Yerel bir Streamlit sunucusu başlatır, Chrome'u başsız açar ve DevTools protokolüyle (websocket-client) demo
akışını gezer: `?demo=1` bağlantısı, Kalite, Zaman Serisi ("Neden Değişti?"), Model, Müşteri, Deney.
Görüntüler `docs/ekranlar/`, GIF `docs/demo.gif` olarak yazılır. Bağımlılıkları (websocket-client, Pillow)
`requirements-dev.txt` içinde sabittir. Chrome yolu `CHROME` ortam değişkeniyle değiştirilebilir.
"""

from __future__ import annotations

import base64
import io
import itertools
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import websocket
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "docs"
APP_PORT, DEBUG_PORT = 8599, 9333
WIDTH, HEIGHT = 1440, 960
GIF_WIDTH = 1100
CHROME_CANDIDATES = (
    os.environ.get("CHROME", ""),
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    "/usr/bin/google-chrome", "/usr/bin/chromium", "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
)


def _wait_http(url: str, timeout: float) -> str:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                return response.read().decode("utf-8")
        except OSError:
            time.sleep(0.5)
    raise TimeoutError(f"{url} yanıt vermedi.")


class Browser:
    """Tek sekmeli, en küçük DevTools istemcisi."""

    def __init__(self, socket_url: str) -> None:
        self.socket = websocket.create_connection(socket_url, timeout=120, suppress_origin=True)
        self.ids = itertools.count(1)

    def send(self, method: str, **params) -> dict:
        message_id = next(self.ids)
        self.socket.send(json.dumps({"id": message_id, "method": method, "params": params}))
        while True:
            message = json.loads(self.socket.recv())
            if message.get("id") == message_id:
                if "error" in message:
                    raise RuntimeError(f"{method}: {message['error']}")
                return message.get("result", {})

    def js(self, expression: str):
        result = self.send("Runtime.evaluate", expression=expression, returnByValue=True, awaitPromise=True)
        return result.get("result", {}).get("value")

    def idle(self, text: str, timeout: float = 240) -> None:
        """Metin sayfada görünene ve Streamlit çalışmayı bitirene kadar bekler."""
        deadline, quiet_since = time.time() + timeout, None
        while time.time() < deadline:
            ready = self.js(
                f"document.body.innerText.includes({json.dumps(text)}) && "
                "!document.querySelector('[data-testid=\"stStatusWidget\"]')"
            )
            if ready:
                quiet_since = quiet_since or time.time()
                if time.time() - quiet_since > 2.0:
                    return
            else:
                quiet_since = None
            time.sleep(0.4)
        raise TimeoutError(f"“{text}” görünmedi.")

    def click(self, text: str, selector: str = "button") -> None:
        found = self.js(f"""(() => {{
            const items = [...document.querySelectorAll({json.dumps(selector)})];
            const item = items.find(node => node.innerText.trim() === {json.dumps(text)})
                || items.find(node => node.innerText.includes({json.dumps(text)}));
            if (!item) return false;
            item.scrollIntoView({{block: 'center'}}); item.click(); return true;
        }})()""")
        if not found:
            raise RuntimeError(f"Tıklanacak öğe bulunamadı: {text}")

    def page(self, title: str, wait_for: str) -> None:
        self.click(title, '[data-testid="stSidebarNav"] a, [data-testid="stSidebarNavLink"]')
        self.idle(wait_for)

    def select(self, label: str, option: str) -> None:
        """Etiketiyle bulunan selectbox'ta seçenek seçer (yazıp Enter)."""
        found = self.js(f"""(() => {{
            const box = [...document.querySelectorAll('[data-testid="stSelectbox"]')]
                .find(node => node.innerText.split('\\n')[0].trim() === {json.dumps(label)});
            if (!box) return false;
            const input = box.querySelector('input'); input.scrollIntoView({{block: 'center'}});
            input.focus(); input.click(); return true;
        }})()""")
        if not found:
            raise RuntimeError(f"Seçim kutusu bulunamadı: {label}")
        time.sleep(0.4)
        self.send("Input.insertText", text=option)
        time.sleep(0.6)
        for kind in ("keyDown", "keyUp"):
            self.send("Input.dispatchKeyEvent", type=kind, key="Enter", code="Enter", windowsVirtualKeyCode=13)
        time.sleep(0.5)
        self.js("document.activeElement && document.activeElement.blur()")

    def scroll_to(self, text: str, selector: str = 'h1, h2, h3, [data-testid="stHeading"]') -> None:
        self.js(f"""(() => {{
            const node = [...document.querySelectorAll({json.dumps(selector)})]
                .find(item => item.innerText.includes({json.dumps(text)}));
            if (node) node.scrollIntoView({{block: 'start'}});
        }})()""")
        time.sleep(1.2)

    def shot(self, name: str) -> Image.Image:
        data = self.send("Page.captureScreenshot", format="png")["data"]
        image = Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB")
        image.save(OUTPUT / "ekranlar" / f"{name}.png", optimize=True)
        print(f"  ✓ {name}")
        return image


def _chrome() -> str:
    for candidate in CHROME_CANDIDATES:
        if candidate and Path(candidate).exists():
            return candidate
    raise FileNotFoundError("Chrome bulunamadı; CHROME ortam değişkeniyle yolunu verin.")


def capture() -> list[tuple[Image.Image, int]]:
    browser = Browser(json.loads(_wait_http(f"http://127.0.0.1:{DEBUG_PORT}/json", 30))[0]["webSocketDebuggerUrl"])
    browser.send("Page.enable")
    browser.send("Emulation.setDeviceMetricsOverride", width=WIDTH, height=HEIGHT, deviceScaleFactor=1, mobile=False)
    frames: list[tuple[Image.Image, int]] = []
    browser.send("Page.navigate", url=f"http://localhost:{APP_PORT}/?demo=1")
    browser.idle("Otopilot Yönetici Özeti", timeout=300)
    frames.append((browser.shot("01_genel_bakis"), 3200))
    browser.scroll_to("Veriye Özel Seçilen")
    frames.append((browser.shot("02_grafikler"), 2600))

    browser.page("Kalite", "İş kuralları")
    browser.scroll_to("İş kuralları")
    frames.append((browser.shot("03_is_kurallari"), 3200))

    browser.page("Zaman Serisi", "Zaman Serisi Otopilotu")
    frames.append((browser.shot("04_zaman_serisi"), 2600))
    browser.select("Metrik", "Ortalama")
    browser.idle("Sütun")
    browser.select("Sütun", "değerlendirme_puanı")
    browser.idle("Değişimi ayrıştır")
    browser.click("Değişimi ayrıştır")
    browser.idle("en büyük katkı")
    browser.scroll_to("en büyük katkı", '[data-testid="stAlert"]')
    frames.append((browser.shot("05_neden_degisti"), 3600))

    browser.page("Model", "Leaderboard")
    frames.append((browser.shot("06_model_sizinti_uyarisi"), 3200))

    browser.page("Müşteri", "RFM segmentleri")
    browser.scroll_to("RFM segmentleri")
    frames.append((browser.shot("07_rfm"), 2800))
    browser.scroll_to("Kohort tutundurma")
    frames.append((browser.shot("08_kohort"), 2800))

    browser.page("Deney", "Deneyi analiz et")
    browser.click("Deneyi analiz et")
    browser.idle("Örneklem oranı uyumu")
    frames.append((browser.shot("09_ab_testi"), 3600))

    browser.page("Rapor", "Raporu Hazırla")
    browser.click("Raporu Hazırla")
    browser.idle("PDF raporu indir")
    frames.append((browser.shot("10_rapor"), 2600))
    return frames


def write_gif(frames: list[tuple[Image.Image, int]]) -> Path:
    resized = [
        image.resize((GIF_WIDTH, round(image.height * GIF_WIDTH / image.width)), Image.LANCZOS)
        .quantize(colors=128, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
        for image, _ in frames
    ]
    path = OUTPUT / "demo.gif"
    resized[0].save(path, save_all=True, append_images=resized[1:], duration=[ms for _, ms in frames], loop=0,
                    optimize=True)
    return path


def main() -> None:
    (OUTPUT / "ekranlar").mkdir(parents=True, exist_ok=True)
    environment = os.environ | {"PYTHONIOENCODING": "utf-8"}
    server = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "app.py", "--server.port", str(APP_PORT), "--server.headless",
         "true", "--browser.gatherUsageStats", "false", "--theme.base", "light"],
        cwd=ROOT, env=environment, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    profile = tempfile.TemporaryDirectory()
    chrome = subprocess.Popen(
        [_chrome(), "--headless=new", f"--remote-debugging-port={DEBUG_PORT}", "--remote-allow-origins=*",
         f"--user-data-dir={profile.name}", f"--window-size={WIDTH},{HEIGHT}", "--hide-scrollbars",
         "--no-first-run", "--lang=tr-TR", "about:blank"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        _wait_http(f"http://localhost:{APP_PORT}/_stcore/health", 60)
        print("Ekran görüntüleri alınıyor...")
        frames = capture()
        path = write_gif(frames)
        print(f"GIF: {path} ({path.stat().st_size / 1_048_576:.1f} MB, {len(frames)} kare)")
    finally:
        for process in (chrome, server):
            _stop_tree(process)
        profile.cleanup()


def _stop_tree(process: subprocess.Popen) -> None:
    """Süreci alt süreçleriyle kapatır: Windows'ta venv başlatıcısı ve scikit-learn işçileri ayrı süreçtir ve
    sunucunun soketini devralıp portu açık tutabilir."""
    if os.name == "nt":
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(process.pid)], capture_output=True, check=False)
    else:
        process.terminate()
    try:
        process.wait(10)
    except subprocess.TimeoutExpired:
        process.kill()


if __name__ == "__main__":
    main()
