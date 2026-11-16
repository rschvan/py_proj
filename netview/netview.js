// netview.js: Streamlit CCv2 front end for the PyPathfinder network view.
// vis-network's UMD bundle is prepended by netview/__init__.py and defines globalThis.vis.
//
// Python -> JS (data):  token, nodes [{id,label,title,x,y}], edges [{from,to,rank,weight}],
//                       base [[x,y],...], directed, maxRank, canRedraw, view {fontSize, weights, maxRank},
//                       legend [[code, name],...] (merged networks only)
// JS -> Python:         state  "coords" {token, xy:[[x,y],...]}   after any change to positions
//                       state  "view"   {fontSize, weights, maxRank}  text/label settings
//                       trigger "redraw" (timestamp)               recompute the layout in Python

const instances = new WeakMap();

const FONT_MIN = 6;
const FONT_MAX = 48;

function edgeFontSize(fontSize) {
  return Math.max(8, fontSize - 8);
}

function createInstance(parentElement) {
  const root = parentElement.querySelector(".nv-root");
  const container = root.querySelector(".nv-canvas");
  const q = (sel) => root.querySelector(sel);

  const inst = {
    root,
    container,
    network: null,
    nodes: null,
    edges: null,
    token: null,
    ids: [],
    base: [],
    maxRank: 1,
    directed: false,
    view: { fontSize: 16, weights: false, maxRank: null },
    physics: false,
    rotateStart: null, // positions captured when a rotate drag begins
    api: {}, // latest setStateValue / setTriggerValue
  };

  // ---------------------------------------------------------------- helpers
  inst.positions = () => {
    const pos = inst.network.getPositions(inst.ids);
    return inst.ids.map((id) => [pos[id].x, pos[id].y]);
  };

  inst.centroid = (xy) => {
    const n = xy.length || 1;
    let cx = 0;
    let cy = 0;
    xy.forEach(([x, y]) => {
      cx += x;
      cy += y;
    });
    return [cx / n, cy / n];
  };

  inst.place = (xy) => {
    xy.forEach(([x, y], i) => inst.network.moveNode(inst.ids[i], x, y));
  };

  inst.commitCoords = () => {
    if (!inst.network) return;
    inst.api.setStateValue("coords", { token: inst.token, xy: inst.positions() });
  };

  inst.commitView = () => {
    inst.api.setStateValue("view", { ...inst.view });
  };

  inst.fit = (animate = true) => {
    inst.network.fit(
      animate ? { animation: { duration: 350, easingFunction: "easeInOutQuad" } } : {}
    );
  };

  inst.transform = (fn) => {
    // fn maps (dx, dy) relative to the centroid to new (dx, dy)
    const xy = inst.positions();
    const [cx, cy] = inst.centroid(xy);
    inst.place(xy.map(([x, y]) => {
      const [nx, ny] = fn(x - cx, y - cy);
      return [cx + nx, cy + ny];
    }));
  };

  inst.rotate = (xy, degrees) => {
    // screen y points down, so this turns the picture clockwise for positive degrees
    const t = (degrees * Math.PI) / 180;
    const c = Math.cos(t);
    const s = Math.sin(t);
    const [cx, cy] = inst.centroid(xy);
    return xy.map(([x, y]) => {
      const dx = x - cx;
      const dy = y - cy;
      return [cx + dx * c - dy * s, cy + dx * s + dy * c];
    });
  };

  // ------------------------------------------------------------ text/labels
  inst.applyFont = () => {
    const fs = inst.view.fontSize;
    const scale = fs / 28.0;
    inst.nodes.update(
      inst.ids.map((id) => ({ id, font: { size: fs }, margin: 5 * scale }))
    );
    inst.edges.update(
      inst.edges.getIds().map((id) => ({
        id,
        width: 1.5 * scale,
        font: { size: edgeFontSize(fs) },
        arrows: { to: { enabled: inst.directed, scaleFactor: Math.max(0.5, scale) } },
      }))
    );
  };

  inst.applyEdgeLabels = () => {
    const maxAllowed = inst.view.maxRank ?? inst.maxRank;
    const all = inst.edges.get();
    inst.edges.update(
      all.map((e) => ({
        id: e.id,
        label: inst.view.weights ? e.weight : " ",
        hidden: e.rank > maxAllowed,
      }))
    );
    const shown = all.filter((e) => e.rank <= maxAllowed).length;
    q(".nv-linkcount").textContent = `${shown} of ${all.length} links`;
  };

  inst.syncControls = () => {
    q('[data-act="weights"]').setAttribute("aria-pressed", String(inst.view.weights));
    q('[data-act="physics"]').setAttribute("aria-pressed", String(inst.physics));
    const slider = q(".nv-maxrank");
    slider.max = String(inst.maxRank);
    slider.value = String(inst.view.maxRank ?? inst.maxRank);
    q(".nv-maxrank-val").textContent = slider.value;
    slider.disabled = inst.maxRank <= 1;
    q('[data-act="rankdown"]').disabled = Number(slider.value) <= 1;
    q('[data-act="rankup"]').disabled = Number(slider.value) >= inst.maxRank;
  };

  // --------------------------------------------------------------- physics
  inst.setPhysics = (on) => {
    inst.physics = on;
    inst.network.setOptions({ physics: { enabled: on } });
    inst.syncControls();
    if (!on) inst.commitCoords();
  };

  // --------------------------------------------------------------- toolbar
  const actions = {
    flipx: () => {
      inst.transform((dx, dy) => [-dx, dy]);
      inst.commitCoords();
    },
    flipy: () => {
      inst.transform((dx, dy) => [dx, -dy]);
      inst.commitCoords();
    },
    rot90: () => {
      inst.place(inst.rotate(inst.positions(), 90));
      inst.commitCoords();
    },
    redraw: () => {
      inst.api.setTriggerValue("redraw", Date.now());
    },
    physics: () => inst.setPhysics(!inst.physics),
    fit: () => inst.fit(),
    reset: () => {
      if (inst.physics) inst.setPhysics(false);
      inst.place(inst.base);
      inst.fit();
      inst.commitCoords();
    },
    fontdec: () => {
      inst.view.fontSize = Math.max(FONT_MIN, inst.view.fontSize - 2);
      inst.applyFont();
      inst.commitView();
    },
    fontinc: () => {
      inst.view.fontSize = Math.min(FONT_MAX, inst.view.fontSize + 2);
      inst.applyFont();
      inst.commitView();
    },
    weights: () => {
      inst.view.weights = !inst.view.weights;
      inst.applyEdgeLabels();
      inst.syncControls();
      inst.commitView();
    },
    rankdown: () => inst.stepRank(-1),
    rankup: () => inst.stepRank(1),
    png: () => inst.savePng(),
  };

  root.querySelectorAll("[data-act]").forEach((btn) => {
    btn.onclick = (e) => {
      e.preventDefault();
      if (inst.network) actions[btn.dataset.act]();
    };
  });

  const rotSlider = q(".nv-rotate");
  const rotVal = q(".nv-rotate-val");
  rotSlider.oninput = () => {
    if (!inst.network) return;
    if (!inst.rotateStart) inst.rotateStart = inst.positions();
    const deg = Number(rotSlider.value);
    rotVal.textContent = `${deg}°`;
    inst.place(inst.rotate(inst.rotateStart, deg));
  };
  rotSlider.onchange = () => {
    // released: keep the rotation and re-centre the slider for the next one
    inst.rotateStart = null;
    rotSlider.value = "0";
    rotVal.textContent = "0°";
    inst.commitCoords();
  };

  const rankSlider = q(".nv-maxrank");
  rankSlider.oninput = () => {
    inst.view.maxRank = Number(rankSlider.value);
    q(".nv-maxrank-val").textContent = rankSlider.value;
    inst.applyEdgeLabels();
    inst.syncControls();  // enables or disables the step buttons
  };
  rankSlider.onchange = () => inst.commitView();

  inst.stepRank = (delta) => {
    const now = inst.view.maxRank ?? inst.maxRank;
    const next = Math.min(inst.maxRank, Math.max(1, now + delta));
    if (next === now) return;
    inst.view.maxRank = next;
    inst.applyEdgeLabels();
    inst.syncControls();
    inst.commitView();
  };

  // ------------------------------------------------------- merged-network legend
  const legendBox = q(".nv-legend");
  inst.legend = [];
  inst.showLegend = (items) => {
    inst.legend = items;
    legendBox.replaceChildren();
    legendBox.hidden = items.length === 0;
    if (!items.length) return;
    const title = document.createElement("div");
    title.className = "nv-legend-title";
    title.textContent = "Merged from";
    legendBox.append(title);
    for (const [code, name] of items) {
      const row = document.createElement("div");
      const c = document.createElement("code");
      c.textContent = code;
      row.append(c, name);
      legendBox.append(row);
    }
  };

  // Draws the legend into the PNG where it appears on screen (ratio: canvas pixels per CSS pixel).
  inst.drawLegend = (ctx, ratio) => {
    if (!inst.legend.length) return;
    const font = 13 * ratio, line = 19.5 * ratio, pad = 8 * ratio, gap = 8 * ratio;
    const x0 = 8 * ratio, y0 = 8 * ratio;
    const mono = `${font}px ui-monospace, Consolas, monospace`;
    const sans = `${font}px arial, sans-serif`;
    ctx.font = mono;
    const codeW = Math.max(...inst.legend.map(([c]) => ctx.measureText(c).width));
    ctx.font = sans;
    const nameW = Math.max(...inst.legend.map(([, n]) => ctx.measureText(n).width));
    ctx.font = `bold ${sans}`;
    const titleW = ctx.measureText("Merged from").width;
    const w = pad * 2 + Math.max(titleW, codeW + gap + nameW);
    const h = pad * 2 + line * (inst.legend.length + 1);
    ctx.fillStyle = "rgba(255,255,255,0.85)";
    ctx.strokeStyle = "#d6d6d9";
    ctx.lineWidth = ratio;
    ctx.fillRect(x0, y0, w, h);
    ctx.strokeRect(x0, y0, w, h);
    ctx.textBaseline = "middle";
    ctx.fillStyle = "#262730";
    ctx.fillText("Merged from", x0 + pad, y0 + pad + line / 2);
    inst.legend.forEach(([code, name], i) => {
      const y = y0 + pad + line * (i + 1.5);
      ctx.font = mono;
      ctx.fillStyle = "blue";
      ctx.fillText(code, x0 + pad, y);
      ctx.font = sans;
      ctx.fillStyle = "#262730";
      ctx.fillText(name, x0 + pad + codeW + gap, y);
    });
  };

  // ------------------------------------------------------------ PNG export
  inst.savePng = () => {
    const src = container.querySelector("canvas");
    if (!src) return;
    const out = document.createElement("canvas");
    out.width = src.width;
    out.height = src.height;
    const ctx = out.getContext("2d");
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, out.width, out.height);
    ctx.drawImage(src, 0, 0);
    inst.drawLegend(ctx, src.width / (src.clientWidth || src.width));
    const a = document.createElement("a");
    a.download = `${inst.name || "network"}.png`;
    a.href = out.toDataURL("image/png");
    a.click();
  };

  // ---------------------------------------------------------------- build
  inst.build = (data) => {
    inst.token = data.token;
    inst.name = data.name;
    inst.directed = !!data.directed;
    inst.maxRank = Math.max(1, data.maxRank || 1);
    inst.base = data.base || data.nodes.map((n) => [n.x, n.y]);
    inst.ids = data.nodes.map((n) => n.id);

    if (data.view) {
      inst.view.fontSize = data.view.fontSize ?? inst.view.fontSize;
      inst.view.weights = !!data.view.weights;
      inst.view.maxRank = data.view.maxRank ?? null;
    } else {
      inst.view.fontSize = data.fontSize || inst.view.fontSize;
    }
    // the max-rank filter belongs to one network: start a different net with all links shown
    const sameNet = (inst.view.net ?? data.view?.net) === data.netId;
    if (!sameNet || inst.view.maxRank == null || inst.view.maxRank > inst.maxRank) {
      inst.view.maxRank = inst.maxRank;
    }
    inst.view.net = data.netId;
    inst.showLegend(data.legend || []);

    const fs = inst.view.fontSize;
    const scale = fs / 28.0;
    inst.nodes = new vis.DataSet(
      data.nodes.map((n) => ({
        id: n.id,
        label: n.label,
        title: n.title,
        x: n.x,
        y: n.y,
        shape: "box",
        color: { background: "#ffffff", border: "#ffffff" },
        font: { size: fs, face: "arial", align: "center" },
        margin: 5 * scale,
        borderWidth: 0,
      }))
    );
    inst.edges = new vis.DataSet(
      data.edges.map((e, i) => ({
        id: i,
        from: e.from,
        to: e.to,
        rank: e.rank,
        weight: e.weight,
        label: " ",
        width: 1.5 * scale,
        color: "blue",
        font: { size: edgeFontSize(fs), color: "blue", align: "top" },
        arrows: { to: { enabled: inst.directed, scaleFactor: Math.max(0.5, scale) } },
        arrowStrikethrough: false,
        smooth: inst.directed ? { enabled: true, type: "curvedCW", roundness: 0.15 } : false,
      }))
    );

    const options = {
      autoResize: true,
      physics: {
        enabled: false,
        solver: "forceAtlas2Based",
        forceAtlas2Based: {
          gravitationalConstant: -120,
          centralGravity: 0.015,
          springLength: 75,
          springConstant: 0.1,
          damping: 0.6,
          avoidOverlap: 0.8,
        },
        minVelocity: 0.5,
        stabilization: { enabled: false },
      },
      interaction: { hover: true, tooltipDelay: 200, zoomView: true },
    };

    if (inst.network) inst.network.destroy();
    inst.physics = false;
    inst.network = new vis.Network(container, { nodes: inst.nodes, edges: inst.edges }, options);
    inst.applyEdgeLabels();
    inst.syncControls();

    inst.network.on("dragEnd", (params) => {
      if (params.nodes && params.nodes.length && !inst.physics) inst.commitCoords();
    });
    inst.network.on("stabilized", () => {
      if (inst.physics) inst.commitCoords();
    });

    inst.network.once("afterDrawing", () => inst.fit(false));
  };

  inst.update = (data) => {
    q('[data-act="redraw"]').hidden = !data.canRedraw;
    if (!data.nodes || !data.nodes.length) return;
    if (data.token !== inst.token || !inst.network) inst.build(data);
  };

  inst.destroy = () => {
    if (inst.network) inst.network.destroy();
    inst.network = null;
  };

  return inst;
}

function scrollPageToTop(parentElement) {
  // Streamlit keeps the scroll position when switching pages (e.g. Display Network at the
  // bottom of Build Nets), which can hide the header and toolbar. Scroll the nearest
  // scrolling ancestor of the component back to the top.
  let el = parentElement.host || parentElement;
  while (el && el !== document.body) {
    const style = getComputedStyle(el);
    if (/(auto|scroll)/.test(style.overflowY) && el.scrollHeight > el.clientHeight) {
      el.scrollTo({ top: 0, behavior: "instant" });
      return;
    }
    el = el.parentElement || (el.getRootNode && el.getRootNode().host);
  }
  window.scrollTo({ top: 0, behavior: "instant" });
}

export default function (component) {
  const { data, parentElement, setStateValue, setTriggerValue } = component;
  let inst = instances.get(parentElement);
  if (!inst) {
    inst = createInstance(parentElement);
    instances.set(parentElement, inst);
    scrollPageToTop(parentElement); // only when the view first appears, not on reruns
  }
  inst.api = { setStateValue, setTriggerValue };
  inst.update(data || {});

  return () => {
    inst.destroy();
    instances.delete(parentElement);
  };
}
