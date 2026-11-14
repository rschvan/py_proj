// netview.js: Streamlit CCv2 front end for the PyPathfinder network view.
// vis-network's UMD bundle is prepended by netview/__init__.py and defines globalThis.vis.
//
// Python -> JS (data):  token, nodes [{id,label,title,x,y}], edges [{from,to,rank}],
//                       base [[x,y],...], directed, maxRank, canRedraw, view {fontSize, ranks, maxRank}
// JS -> Python:         state  "coords" {token, xy:[[x,y],...]}   after any change to positions
//                       state  "view"   {fontSize, ranks, maxRank}  text/label settings
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
    view: { fontSize: 16, ranks: false, maxRank: null },
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
    inst.edges.update(
      inst.edges.get().map((e) => ({
        id: e.id,
        label: inst.view.ranks ? String(e.rank) : " ",
        hidden: e.rank > maxAllowed,
      }))
    );
  };

  inst.syncControls = () => {
    q('[data-act="ranks"]').setAttribute("aria-pressed", String(inst.view.ranks));
    q('[data-act="physics"]').setAttribute("aria-pressed", String(inst.physics));
    const slider = q(".nv-maxrank");
    slider.max = String(inst.maxRank);
    slider.value = String(inst.view.maxRank ?? inst.maxRank);
    q(".nv-maxrank-val").textContent = slider.value;
    slider.disabled = inst.maxRank <= 1;
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
    ranks: () => {
      inst.view.ranks = !inst.view.ranks;
      inst.applyEdgeLabels();
      inst.syncControls();
      inst.commitView();
    },
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
  };
  rankSlider.onchange = () => inst.commitView();

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
      inst.view.ranks = !!data.view.ranks;
      inst.view.maxRank = data.view.maxRank ?? null;
    } else {
      inst.view.fontSize = data.fontSize || inst.view.fontSize;
    }
    if (inst.view.maxRank == null || inst.view.maxRank > inst.maxRank) {
      inst.view.maxRank = inst.maxRank;
    }

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

export default function (component) {
  const { data, parentElement, setStateValue, setTriggerValue } = component;
  let inst = instances.get(parentElement);
  if (!inst) {
    inst = createInstance(parentElement);
    instances.set(parentElement, inst);
  }
  inst.api = { setStateValue, setTriggerValue };
  inst.update(data || {});

  return () => {
    inst.destroy();
    instances.delete(parentElement);
  };
}
