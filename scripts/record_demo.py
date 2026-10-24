"""Graba docs/demo.gif recorriendo la página de demo con un navegador headless.

Requiere: pip install playwright pillow && playwright install chromium
Uso: python scripts/record_demo.py [http://localhost:8000]
"""

import io
import sys
import time
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
OUT = Path(__file__).resolve().parent.parent / "docs"
STEPS = [
    "¿Cuál es la capital de Francia?",
    "dime la capital de francia",
    "¿Qué ciudad es la capital de Francia?",
    "¿Cuál es la capital de Suecia?",
    "Dime la capital de Suecia",
]
FRAME_S = 0.12


def main() -> None:
    OUT.mkdir(exist_ok=True)
    frames: list[tuple[Image.Image, int]] = []

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1100, "height": 720}, device_scale_factor=1)
        page.goto(BASE)
        page.click("#reset")
        page.wait_for_timeout(500)

        def snap(hold_ms: int = int(FRAME_S * 1000)) -> None:
            img = Image.open(io.BytesIO(page.screenshot())).convert("RGB")
            frames.append((img, hold_ms))

        snap(1200)
        for query in STEPS:
            page.fill("#q", "")
            for i in range(0, len(query), 3):  # tecleo visible
                page.fill("#q", query[: i + 3])
                snap(60)
            page.click("#b")
            # Las capturas consumen CPU y competirían con el embedding: un HIT termina antes.
            page.wait_for_timeout(150)
            # Captura mientras espera: en un MISS se ve el botón ocupado durante la llamada al LLM.
            start = time.monotonic()
            while page.is_disabled("#b") and time.monotonic() - start < 30:
                snap()
                page.wait_for_timeout(int(FRAME_S * 1000))
            page.wait_for_timeout(250)
            snap(2200)
        page.screenshot(path=OUT / "screenshot.png")
        browser.close()

    images = [f.quantize(colors=128, method=Image.Quantize.MEDIANCUT) for f, _ in frames]
    images[0].save(
        OUT / "demo.gif",
        save_all=True,
        append_images=images[1:],
        duration=[d for _, d in frames],
        loop=0,
        optimize=True,
    )
    print(f"{len(frames)} frames -> {OUT / 'demo.gif'}")


if __name__ == "__main__":
    main()
