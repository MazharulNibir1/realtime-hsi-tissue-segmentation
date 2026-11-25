/**
 * Guided tour: a sequence of cards, each spotlighting one part of the page.
 * Shown once per browser (localStorage, wrapped in try/catch), replayable from the "Guide" button.
 */

export interface TourStep {
  title: string;
  body: string;
  /** CSS selector of the element to spotlight; omitted = centred card with no pointer. */
  target?: string;
}

export const STEPS: TourStep[] = [
  {
    title: "What is this?",
    body:
      "A hyperspectral camera records 100 colours per pixel instead of 3. Different organs reflect those " +
      "colours differently, so a small AI model can tell which organ each pixel shows. It runs right here " +
      "in your browser, on pig surgery images it never saw during training.",
  },
  {
    title: "1. Pick a frame",
    body: "Each tab is a real surgical image from a test pig. Start with Liver, then try Kidney: that one is the model's known weak spot.",
    target: "#samples",
  },
  {
    title: "2. Choose the camera",
    body:
      "Full spectrum uses all 100 colours. 16-band simulates a compact snapshot camera that can film hyperspectral " +
      "video. Switch between them and watch how little changes.",
    target: "#models",
  },
  {
    title: "3. Read the prediction",
    body:
      "Each colour is the organ the model predicts for that pixel (the class table lists them). The white outline " +
      "marks the area a surgeon labelled: only pixels inside it are scored.",
    target: "#over",
  },
  {
    title: "4. Click any pixel",
    body:
      "Click inside the image. The chart then shows that pixel's spectrum (black) against the average spectra " +
      "of the two most likely organs. Similar curves mean a confident prediction.",
    target: ".spectrum",
  },
  {
    title: "5. Check the result",
    body: "This line says how many labelled pixels were classified correctly, which model ran, and how long one frame took.",
    target: "#finding",
  },
  {
    title: "6. Measure the speed",
    body:
      "Press Run 50 frames to benchmark your own device. The runtime panel shows the backend (WebGPU or " +
      "WebAssembly), the time per frame and the frames per second.",
    target: "#bench",
  },
];

const SEEN_KEY = "hsi-tour-seen-v1";

function storage(get: boolean): boolean {
  try {
    if (get) return localStorage.getItem(SEEN_KEY) === "1";
    localStorage.setItem(SEEN_KEY, "1");
  } catch {
    /* private mode or blocked storage: the tour simply shows again next time */
  }
  return false;
}

export function tourSeen(): boolean {
  return storage(true);
}

let active: (() => void) | null = null;

export function startTour(steps: TourStep[] = STEPS): void {
  active?.();
  let i = 0;
  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const opener = document.activeElement as HTMLElement | null;

  const layer = document.createElement("div");
  layer.className = "tour-layer";
  const spot = document.createElement("div");
  spot.className = "tour-spot";
  const card = document.createElement("div");
  card.className = "tour-card";
  card.setAttribute("role", "dialog");
  card.setAttribute("aria-modal", "false");
  card.setAttribute("aria-labelledby", "tour-title");
  card.innerHTML = `
    <div class="tour-arrow" aria-hidden="true"></div>
    <p class="tour-count"></p>
    <h2 id="tour-title" class="tour-title"></h2>
    <p class="tour-body" aria-live="polite"></p>
    <div class="tour-actions">
      <button type="button" class="tour-skip">Skip tour</button>
      <span class="tour-nav">
        <button type="button" class="btn secondary tour-back">Back</button>
        <button type="button" class="btn primary tour-next">Next</button>
      </span>
    </div>`;
  layer.append(spot, card);
  document.body.append(layer);

  const $ = <T extends HTMLElement>(s: string) => card.querySelector(s) as T;
  const back = $<HTMLButtonElement>(".tour-back");
  const next = $<HTMLButtonElement>(".tour-next");

  const place = () => {
    const step = steps[i];
    const el = step.target ? (document.querySelector(step.target) as HTMLElement | null) : null;
    const phone = window.innerWidth < 640;
    card.classList.toggle("centred", !el);
    card.classList.toggle("docked", Boolean(el) && phone);
    if (!el) {
      spot.style.display = "none";
      card.style.left = card.style.top = "";
      return;
    }
    const r = el.getBoundingClientRect();
    const pad = 6;
    spot.style.display = "block";
    Object.assign(spot.style, {
      left: `${r.left - pad}px`,
      top: `${r.top - pad}px`,
      width: `${r.width + 2 * pad}px`,
      height: `${r.height + 2 * pad}px`,
    });
    if (phone) {
      card.style.left = card.style.top = "";
      return;
    }
    const cw = card.offsetWidth;
    const ch = card.offsetHeight;
    const gap = 16;
    const below = r.bottom + gap + ch < window.innerHeight;
    const top = below ? r.bottom + gap : Math.max(12, r.top - gap - ch);
    const left = Math.min(Math.max(12, r.left + r.width / 2 - cw / 2), window.innerWidth - cw - 12);
    card.style.left = `${left}px`;
    card.style.top = `${top}px`;
    card.dataset.side = below ? "below" : "above";
    const arrow = $(".tour-arrow");
    arrow.style.left = `${Math.min(Math.max(20, r.left + r.width / 2 - left), cw - 20)}px`;
  };

  const show = () => {
    const step = steps[i];
    $(".tour-count").textContent = `${i + 1} of ${steps.length}`;
    $(".tour-title").textContent = step.title;
    $(".tour-body").textContent = step.body;
    back.disabled = i === 0;
    next.textContent = i === steps.length - 1 ? "Start exploring" : "Next";
    const el = step.target ? document.querySelector(step.target) : null;
    if (el) {
      const behavior: ScrollBehavior = reduced ? "auto" : "smooth";
      if (window.innerWidth < 640) {
        // phone: the card docks at the bottom, so bring the target to the top of the screen
        const y = el.getBoundingClientRect().top + window.scrollY - 72;
        window.scrollTo({ top: Math.max(0, y), behavior });
      } else el.scrollIntoView({ block: "center", behavior });
      // place after the scroll has settled; reposition continuously during it
      let frames = 0;
      const follow = () => {
        place();
        if (frames++ < 40) requestAnimationFrame(follow);
      };
      requestAnimationFrame(follow);
    } else place();
    next.focus({ preventScroll: true });
  };

  const close = () => {
    storage(false);
    layer.remove();
    window.removeEventListener("resize", place);
    window.removeEventListener("scroll", place, true);
    document.removeEventListener("keydown", onKey);
    active = null;
    opener?.focus?.({ preventScroll: true });
  };
  const go = (d: number) => {
    const j = i + d;
    if (j < 0) return;
    if (j >= steps.length) return close();
    i = j;
    show();
  };
  const onKey = (e: KeyboardEvent) => {
    if (e.key === "Escape") close();
    else if (e.key === "ArrowRight") go(1);
    else if (e.key === "ArrowLeft") go(-1);
  };

  next.addEventListener("click", () => go(1));
  back.addEventListener("click", () => go(-1));
  $(".tour-skip").addEventListener("click", close);
  layer.addEventListener("click", (e) => {
    if (e.target === layer) close();
  });
  window.addEventListener("resize", place);
  window.addEventListener("scroll", place, true);
  document.addEventListener("keydown", onKey);
  active = close;
  show();
}
