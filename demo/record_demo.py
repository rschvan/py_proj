# demo/record_demo.py: records the PyPathfinder demo video and writes its narration script.
#
#   venv/Scripts/python.exe demo/record_demo.py           # full-length recording (~10 min)
#   venv/Scripts/python.exe demo/record_demo.py --fast    # quick test run (no narration pauses)
#
# It starts its own copy of the app on port 8599, drives a fresh browser session through
# demo/scenes.py, and writes to demo/output/:
#   pypf_demo.webm   the silent video (YouTube accepts .webm)
#   demo_script.md   the narration with each scene's start time, plus YouTube chapters
# Needs Playwright:  venv/Scripts/python.exe -m pip install playwright
#                    venv/Scripts/python.exe -m playwright install chromium
import argparse
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path(__file__).parent))
from scenes import SCENES  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "output"
PORT = 8599
SIZE = {"width": 1600, "height": 900}  # 16:9; the whole network display fits
WORDS_PER_SECOND = 2.4  # relaxed narration, about 145 words a minute
ROW_H = 35              # Streamlit dataframe header and row height (px)

# Added to every page: a visible pointer (headless video shows none), a scene-title
# banner, and no Streamlit toolbar (Deploy button / menu) in the recording.
PAGE_EXTRAS_JS = r"""
(() => {
  const add = () => {
    if (document.getElementById("demo-cursor")) return;
    const css = document.createElement("style");
    css.textContent = `
      #demo-cursor { position: fixed; left: -50px; top: -50px; width: 22px; height: 22px;
        margin: -11px 0 0 -11px; border-radius: 50%; pointer-events: none; z-index: 2147483647;
        background: rgba(255, 75, 75, 0.35); border: 2px solid rgba(220, 30, 30, 0.85);
        transition: transform 0.12s, background 0.12s; }
      #demo-cursor.down { transform: scale(0.7); background: rgba(255, 40, 40, 0.75); }
      #demo-caption { position: fixed; left: 24px; bottom: 24px; z-index: 2147483646;
        padding: 10px 18px; border-radius: 8px; font: 600 22px "Source Sans Pro", sans-serif;
        color: #fff; background: rgba(30, 40, 60, 0.85); opacity: 0; transition: opacity 0.5s;
        pointer-events: none; }
      #demo-caption.show { opacity: 1; }
      [data-testid="stToolbar"], [data-testid="stAppDeployButton"] { display: none !important; }
    `;
    document.head.appendChild(css);
    const c = document.createElement("div"); c.id = "demo-cursor"; document.body.appendChild(c);
    const cap = document.createElement("div"); cap.id = "demo-caption"; document.body.appendChild(cap);
    document.addEventListener("mousemove", e => { c.style.left = e.clientX + "px"; c.style.top = e.clientY + "px"; }, true);
    document.addEventListener("mousedown", () => c.classList.add("down"), true);
    document.addEventListener("mouseup", () => c.classList.remove("down"), true);
  };
  if (document.body) add(); else document.addEventListener("DOMContentLoaded", add);
})();
"""


class Demo:
    """Helpers the scenes use to drive the app at a watchable pace."""

    def __init__(self, page, fast=False):
        self.page = page
        self.fast = fast
        self.xy = (SIZE["width"] / 2, SIZE["height"] / 2)

    # ------------------------------------------------------------ timing
    def pause(self, seconds):
        self.page.wait_for_timeout(int(seconds * (0.25 if self.fast else 1) * 1000))

    def settle(self):
        """Waits for Streamlit to finish rerunning after an interaction: the running
        indicator gone and no element still marked stale from the previous run."""
        self.page.wait_for_timeout(500)
        end = time.monotonic() + 20
        while time.monotonic() < end:
            busy = self.page.evaluate("""() => {
                const s = document.querySelector('[data-testid="stStatusWidget"]');
                return !!(s && s.offsetParent) || !!document.querySelector('[data-stale="true"]');
            }""")
            if not busy:
                break
            self.page.wait_for_timeout(150)
        self.page.wait_for_timeout(300)

    # ------------------------------------------------------------ finding things
    def text(self, s, sidebar=False):
        scope = self.page.locator('[data-testid="stSidebar"]' if sidebar else '[data-testid="stMain"]')
        return scope.get_by_text(s, exact=True).first

    def button(self, name):
        return self.page.get_by_role("button", name=name).first

    def dataframe(self, i):
        return self.page.locator('[data-testid="stMain"] [data-testid="stDataFrame"]').nth(i)

    def toolbar(self, act, **_):
        return self.page.locator(f'[data-act="{act}"]').first

    # ------------------------------------------------------------ pointer
    def _move(self, x, y, steps=22):
        self.page.mouse.move(x, y, steps=1 if self.fast else steps)
        self.xy = (x, y)

    def _in_view(self, box):
        return box and 40 <= box["y"] and box["y"] + box["height"] <= SIZE["height"] - 10

    def _still_box(self, locator, timeout=4.0):
        """Bounding box once the element has stopped moving (smooth scrolls, expanders opening)."""
        last, end = None, time.monotonic() + timeout
        while time.monotonic() < end:
            box = locator.bounding_box()
            if box and last and all(abs(box[k] - last[k]) < 0.5 for k in box):
                return box
            last = box
            self.page.wait_for_timeout(120)
        return locator.bounding_box()

    def _box(self, locator):
        locator.wait_for(state="visible", timeout=15000)
        box = self._still_box(locator)
        if not self._in_view(box):
            locator.evaluate("el => el.scrollIntoView({behavior: 'smooth', block: 'center'})")
            box = self._still_box(locator)
        return box

    def point_at(self, locator):
        box = self._box(locator)
        self._move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)

    def click(self, locator):
        self.point_at(locator)
        self.pause(0.3)
        self.page.mouse.click(*self.xy)
        self.settle()

    # ------------------------------------------------------------ tables
    _ROWS_JS = """el => [...el.querySelectorAll('[role=row]')]
        .map(r => ({i: +r.getAttribute('aria-rowindex'),
                    selected: r.getAttribute('aria-selected') === 'true',
                    first: (r.querySelector('td, [role=gridcell]') || {}).textContent}))
        .filter(r => r.i > 1)"""

    def _rows(self, table):
        """The table's rows once they have stopped changing (a rerun may be redrawing them)."""
        last, end = None, time.monotonic() + 10
        while time.monotonic() < end:
            rows = self.dataframe(table).evaluate(self._ROWS_JS)
            names = [r["first"] for r in rows]
            if rows and names == last:
                return rows
            last = names
            self.page.wait_for_timeout(250)
        return self.dataframe(table).evaluate(self._ROWS_JS)

    def _find_row(self, table, row=None, row_index=None, row_startswith=None):
        rows = self._rows(table)
        if row_index is not None:
            return rows[row_index]
        for r in rows:
            if (row is not None and r["first"] == row) or                (row_startswith is not None and (r["first"] or "").startswith(row_startswith)):
                return r
        raise LookupError(f"row {row or row_startswith!r} not found in table {table}: {rows}")

    def _row_y(self, box, found):
        return box["y"] + (found["i"] - 1) * ROW_H + ROW_H / 2

    def point_at_table_cell(self, table, frac=0.12, **row):
        box = self._box(self.dataframe(table))
        self._move(box["x"] + box["width"] * frac, self._row_y(box, self._find_row(table, **row)))

    def _click_row_box(self, table, name):
        box = self._box(self.dataframe(table))
        self._move(box["x"] + 17, self._row_y(box, self._find_row(table, row=name)))
        self.pause(0.4)
        self.page.mouse.click(*self.xy)
        self.settle()

    def select_rows(self, table, names):
        """Checks the named rows, making sure each intended row is the one selected."""
        for name in names:
            for _attempt in range(3):
                self._click_row_box(table, name)
                rows = self._rows(table)
                wrong = [r["first"] for r in rows if r["selected"] and r["first"] not in names]
                if any(r["first"] == name and r["selected"] for r in rows) and not wrong:
                    break
                for other in wrong:  # undo a stray selection, then try again
                    self._click_row_box(table, other)
            else:
                raise RuntimeError(f"could not select {name!r} in table {table}")
            self.pause(0.6)

    # ------------------------------------------------------------ widgets
    def type_into(self, label_prefix, text):
        label = self.page.locator('[data-testid="stWidgetLabel"]', has_text=label_prefix)
        box_input = self.page.locator('[data-testid="stTextInput"]', has=label).locator("input").first
        self.click(box_input)
        self.page.keyboard.press("Control+A")
        self.page.keyboard.type(text, delay=0 if self.fast else 140)
        self.page.keyboard.press("Enter")
        self.settle()

    def drag_slider(self, selector, fraction):
        slider = self.page.locator(selector).first
        box = self._box(slider)
        now = slider.evaluate("el => (el.value - el.min) / ((el.max - el.min) || 1)")
        y = box["y"] + box["height"] / 2
        self._move(box["x"] + box["width"] * now, y)
        self.page.mouse.down()
        self._move(box["x"] + box["width"] * fraction, y, steps=40)
        self.page.mouse.up()
        self.settle()

    def _choose(self, label, option):
        select = self.page.locator('[data-testid="stSelectbox"]').filter(has_text=label).first
        self.click(select)
        self.page.keyboard.type(option, delay=0 if self.fast else 90)
        self.pause(0.6)
        self.page.get_by_role("option", name=option, exact=True).first.click()
        self.settle()

    def choose_layout(self, name):
        self._choose("Select Layout Algorithm", name)

    def choose_network(self, name):
        self._choose("Select Network", name)

    # ------------------------------------------------------------ navigation
    def go_page(self, name):
        self.click(self.page.locator('[data-testid="stSidebarNavLink"]').filter(has_text=name).first)
        self.pause(1.5)

    def page_top(self):
        self.page.evaluate("document.querySelector('[data-testid=stMain]').scrollTo({top: 0, behavior: 'smooth'})")
        self.pause(1)

    def scroll_to_heading(self, text):
        heading = self.page.locator("h1, h2, h3").filter(has_text=text).first
        heading.wait_for(state="attached", timeout=15000)
        heading.evaluate("el => el.scrollIntoView({behavior: 'smooth', block: 'start'})")
        self._still_box(heading)
        self.pause(0.6)

    def scroll_to_text(self, text):
        el = self.page.locator('[data-testid="stMain"]').get_by_text(text).first
        el.evaluate("el => el.scrollIntoView({behavior: 'smooth', block: 'start'})")
        self._still_box(el)
        self.pause(0.6)

    def help_jump(self, anchor):
        self.click(self.page.locator(f'[data-testid="stSidebar"] a[href="#{anchor}"]').first)

    def caption(self, text, seconds=3.5):
        self.page.evaluate("""([t, ms]) => { const c = document.getElementById('demo-caption');
            if (!c) return; c.textContent = t; c.classList.add('show');
            clearTimeout(window.__capT); window.__capT = setTimeout(() => c.classList.remove('show'), ms); }""",
                           [text, int(seconds * 1000)])


# ---------------------------------------------------------------------- app server
def start_app():
    proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "stapp.py", "--server.port", str(PORT),
         "--server.headless", "true", "--server.runOnSave", "false",
         "--browser.gatherUsageStats", "false"],
        cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(120):
        try:
            urllib.request.urlopen(f"http://localhost:{PORT}/_stcore/health", timeout=1)
            return proc
        except Exception:
            time.sleep(0.5)
    proc.terminate()
    raise RuntimeError("the app did not start")


def wait_for_app(page):
    page.goto(f"http://localhost:{PORT}/")
    page.locator('[data-testid="stDataFrame"]').first.wait_for(state="visible", timeout=60000)
    page.wait_for_timeout(1500)


# ---------------------------------------------------------------------- script output
def mmss(seconds):
    s = int(round(seconds))
    return f"{s // 60}:{s % 60:02d}"


def write_script(timings, total, path):
    lines = ["# PyPathfinder demo: narration script", "",
             f"Video: `pypf_demo.webm`, {mmss(total)} long. Each section starts at the time shown;",
             "read its narration while that part of the video plays.", "",
             "## YouTube chapters (paste into the video description)", ""]
    # YouTube needs the first chapter at 0:00 (the first seconds are the app loading)
    lines += [f"{mmss(0 if i == 0 else t)} {scene.title}" for i, (scene, t, _) in enumerate(timings)]
    lines += [""]
    for scene, start, length in timings:
        text = " ".join(scene.narration.split())
        lines += [f"## {mmss(start)} {scene.title}",
                  f"*{len(text.split())} words, about {round(length)} seconds of video*", "", text, ""]
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast", action="store_true", help="quick test run without narration pauses")
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    video_name = "pypf_demo_fast.webm" if args.fast else "pypf_demo.webm"

    app = start_app()
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            # warm-up visit (not recorded) so the recording doesn't start on a loading screen
            warm = browser.new_page(viewport=SIZE)
            wait_for_app(warm)
            warm.close()

            raw = OUT / "_raw"  # Playwright writes the video here; it is moved when done
            shutil.rmtree(raw, ignore_errors=True)
            if True:
                context = browser.new_context(viewport=SIZE, record_video_dir=raw, record_video_size=SIZE)
                context.add_init_script(PAGE_EXTRAS_JS)
                page = context.new_page()
                t0 = time.monotonic()
                wait_for_app(page)
                d = Demo(page, fast=args.fast)

                timings = []
                for n, scene in enumerate(SCENES, 1):
                    start = time.monotonic() - t0
                    print(f"[{mmss(start)}] {n}/{len(SCENES)} {scene.title}", flush=True)
                    if n > 1:
                        d.caption(scene.title)
                    try:
                        scene.actions(d)
                    except Exception:
                        page.screenshot(path=str(OUT / f"failed_scene_{n}.png"))
                        context.close()
                        raise
                    if not args.fast:
                        target = len(scene.narration.split()) / WORDS_PER_SECOND + 1.5
                        spent = time.monotonic() - t0 - start
                        if spent < target:
                            d.pause(target - spent)
                    timings.append((scene, start, time.monotonic() - t0 - start))
                total = time.monotonic() - t0
                d.pause(2)

                video_path = page.video.path()
                context.close()  # finishes writing the video
                shutil.move(video_path, OUT / video_name)
            browser.close()
    finally:
        app.terminate()

    if not args.fast:
        write_script(timings, total, OUT / "demo_script.md")
    print(f"done: {OUT / video_name} ({mmss(total)})")


if __name__ == "__main__":
    main()
