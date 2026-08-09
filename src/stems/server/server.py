"""FastAPI-based LAN web server for stems.

Run headlessly via ``stems serve``, or start/stop from the GUI's server toggle.
A single :class:`WebServer` wraps uvicorn in a daemon thread so the calling
process (GUI or CLI) keeps control of the main thread.
"""

import asyncio
import io
import json
import queue
import socket
import threading
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from stems.registry import JobRegistry

# ─────────────────────────────────────────────── HTML served at GET /

_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Stems — Audio Separator</title>
<style>
:root{
  --bg:#111317;
  --card:#1c1f24;
  --card2:#242830;
  --border:#383d47;
  --border-hi:#50566a;
  --accent:#5aabf0;
  --accent-dim:rgba(90,171,240,.12);
  --text:#eef0f4;
  --sub:#b0b8c8;
  --muted:#7a8494;
  --ok:#3ec87a;
  --err:#e05555;
  --warn:#e09a30;
}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:system-ui,-apple-system,sans-serif;background:var(--bg);color:var(--text);min-height:100vh;font-size:15px;line-height:1.5}
header{padding:14px 28px;border-bottom:1px solid var(--border);display:flex;align-items:baseline;gap:14px;background:var(--card)}
header h1{font-size:1.15rem;color:var(--accent);font-weight:700;letter-spacing:-.01em}
header .ver{font-size:.78rem;color:var(--muted)}
main{max-width:860px;margin:0 auto;padding:24px 18px;display:flex;flex-direction:column;gap:20px}
.card{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:20px 24px}
.card h2{font-size:.72rem;font-weight:700;color:var(--muted);text-transform:uppercase;letter-spacing:.1em;margin-bottom:16px}
/* drop zone */
#drop-zone{border:2px dashed var(--border-hi);border-radius:8px;padding:32px 16px;text-align:center;color:var(--sub);transition:border-color .15s,background .15s;cursor:default;font-size:.95em}
#drop-zone.drag-over{border-color:var(--accent);background:var(--accent-dim)}
#browse-lnk{color:var(--accent);cursor:pointer;text-decoration:underline}
#file-name{margin-top:8px;font-size:.85em;color:var(--sub)}
/* form controls */
.opts{display:flex;flex-wrap:wrap;gap:14px 24px;margin:16px 0}
.opts label{display:flex;flex-direction:column;gap:5px;font-size:.8rem;font-weight:600;color:var(--sub);text-transform:uppercase;letter-spacing:.04em}
.opts select{
  background:var(--card2);border:1px solid var(--border-hi);color:var(--text);
  padding:7px 10px;border-radius:6px;font-size:.9rem;min-width:130px;
  appearance:none;-webkit-appearance:none;
  background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='6'%3E%3Cpath d='M0 0l5 6 5-6z' fill='%237a8494'/%3E%3C/svg%3E");
  background-repeat:no-repeat;background-position:right 10px center;padding-right:28px;
}
.opts select:focus{outline:2px solid var(--accent);outline-offset:1px;border-color:var(--accent)}
.opts .chk{flex-direction:row;align-items:center;gap:8px;font-size:.88rem;color:var(--text);text-transform:none;letter-spacing:0;font-weight:400;margin-top:2px}
.opts .chk input[type=checkbox]{width:16px;height:16px;accent-color:var(--accent);cursor:pointer}
/* preset desc */
#preset-desc{font-size:.8rem;color:var(--muted);margin-top:2px;line-height:1.4;font-weight:400;text-transform:none;letter-spacing:0}
/* submit row */
.row-btn{display:flex;align-items:center;gap:14px;margin-top:6px}
#submit-btn{background:var(--accent);color:#0a0e14;border:none;padding:10px 26px;border-radius:6px;font-size:.95rem;font-weight:700;cursor:pointer;letter-spacing:.01em}
#submit-btn:hover{filter:brightness(1.1)}
#submit-btn:disabled{opacity:.4;cursor:not-allowed}
#msg{font-size:.85rem}
#msg.ok{color:var(--ok)}
#msg.err{color:var(--err)}
/* job rows */
.job{background:var(--card2);border:1px solid var(--border);border-radius:8px;padding:12px 16px;margin-bottom:8px}
.job-hdr{display:flex;align-items:center;gap:10px}
.badge{font-size:.65rem;padding:2px 8px;border-radius:10px;font-weight:800;text-transform:uppercase;letter-spacing:.05em;flex-shrink:0}
.badge.web{background:var(--accent-dim);color:var(--accent)}
.badge.gui{background:#2a2f38;color:var(--muted)}
.jlabel{flex:1;font-size:.9rem;color:var(--text);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.jstatus{font-size:.75rem;padding:2px 10px;border-radius:4px;background:#2a2f38;flex-shrink:0;font-weight:600}
.s-queued{color:var(--muted)}
.s-running{color:var(--accent)}
.s-done{color:var(--ok)}
.s-failed{color:var(--err)}
.s-cancelled{color:var(--warn)}
.prog-wrap{margin-top:10px}
.prog-bar{height:5px;background:#2a2f38;border-radius:3px;overflow:hidden;margin-bottom:4px}
.prog-fill{height:100%;background:var(--accent);border-radius:3px;transition:width .25s}
.step-lbl{font-size:.75rem;color:var(--muted)}
.jact{margin-top:10px}
a.dl-btn{display:inline-block;background:var(--ok);color:#0a0e14;padding:6px 18px;border-radius:5px;font-size:.85rem;font-weight:700;text-decoration:none}
a.dl-btn:hover{filter:brightness(1.08)}
.empty{color:var(--muted);font-size:.9rem}
</style>
</head>
<body>
<header>
  <div>
    <h1>Stems Audio Separator</h1>
    <div class="ver">LAN server — submit audio, get stems</div>
  </div>
</header>
<main>
  <section class="card">
    <h2>Submit Job</h2>
    <div id="drop-zone">Drop an audio file here or <span id="browse-lnk">browse</span>
      <input type="file" id="fpick" accept=".mp3,.wav,.flac,.aiff,.aif,.ogg,.m4a,.wma" hidden>
    </div>
    <div id="file-name">No file selected</div>
    <form id="jform">
      <div class="opts">
        <label style="flex-basis:100%">Preset
          <select id="o-preset" style="width:100%;max-width:400px"></select>
          <span id="preset-desc" style="display:block;margin-top:4px;font-size:.78em;color:var(--muted)"></span>
        </label>
        <label>Format<select id="o-fmt">
          <option value="both">WAV + MP3</option>
          <option value="wav">WAV only</option>
          <option value="mp3">MP3 only</option>
        </select></label>
        <label>Device<select id="o-dev">
          <option value="auto">Auto</option>
          <option value="cuda">CUDA (GPU)</option>
          <option value="cpu">CPU</option>
        </select></label>
        <label>WAV bits<select id="o-bits">
          <option value="24">24-bit</option>
          <option value="16">16-bit</option>
          <option value="32">32-bit</option>
        </select></label>
        <label class="chk"><input type="checkbox" id="o-tail" checked> Tail rescue</label>
      </div>
      <div class="row-btn">
        <button type="submit" id="submit-btn">Separate</button>
        <span id="msg"></span>
      </div>
    </form>
  </section>
  <section class="card">
    <h2>Job Queue</h2>
    <div id="jlist"><p class="empty">No jobs yet. Submit an audio file above.</p></div>
  </section>
</main>
<script>
const jobMap={};
let curJobId=null;

// Load presets — store metadata for the description hint
const presetMeta={};
fetch('/api/presets').then(r=>r.json()).then(presets=>{
  const sel=document.getElementById('o-preset');
  sel.innerHTML='';
  presets.forEach(p=>{
    presetMeta[p.name]=p;
    const o=document.createElement('option');
    o.value=p.name;
    o.textContent=p.name;
    sel.appendChild(o);
  });
  updatePresetDesc();
});
function updatePresetDesc(){
  const p=presetMeta[document.getElementById('o-preset').value];
  const el=document.getElementById('preset-desc');
  if(p)el.textContent=`${p.description}  →  ${p.output_stems.join(', ')}`;
}
document.getElementById('o-preset').addEventListener('change',updatePresetDesc);

// File picker
const dz=document.getElementById('drop-zone');
const fp=document.getElementById('fpick');
const fn=document.getElementById('file-name');
let chosenFile=null;
document.getElementById('browse-lnk').onclick=()=>fp.click();
fp.onchange=e=>{setFile(e.target.files[0]);};
dz.addEventListener('dragover',e=>{e.preventDefault();dz.classList.add('drag-over');});
dz.addEventListener('dragleave',()=>dz.classList.remove('drag-over'));
dz.addEventListener('drop',e=>{e.preventDefault();dz.classList.remove('drag-over');setFile(e.dataTransfer.files[0]);});
function setFile(f){if(!f)return;chosenFile=f;fn.textContent=f.name;}

// Form submit
document.getElementById('jform').addEventListener('submit',async e=>{
  e.preventDefault();
  if(!chosenFile){setMsg('Please choose a file first.','err');return;}
  const fd=new FormData();
  fd.append('file',chosenFile);
  fd.append('preset',document.getElementById('o-preset').value);
  fd.append('format',document.getElementById('o-fmt').value);
  fd.append('bitdepth',document.getElementById('o-bits').value);
  fd.append('device',document.getElementById('o-dev').value);
  fd.append('tail_rescue',document.getElementById('o-tail').checked?'true':'false');
  const btn=document.getElementById('submit-btn');
  btn.disabled=true; setMsg('Uploading…','');
  try{
    const r=await fetch('/api/jobs',{method:'POST',body:fd});
    if(r.ok){
      const j=await r.json();
      setMsg(`Job #${j.id} queued.`,'ok');
      chosenFile=null;fn.textContent='No file selected';fp.value='';
    }else{
      let detail=r.statusText;
      try{const j=await r.json();if(j.detail)detail=j.detail;}catch{}
      setMsg(`Error ${r.status}: ${detail}`,'err');
    }
  }catch(err){setMsg(`Upload failed: ${err.message}`,'err');}
  finally{btn.disabled=false;}
});
function setMsg(t,cls){const el=document.getElementById('msg');el.textContent=t;el.className=cls;}

// SSE
const es=new EventSource('/api/events');
es.onmessage=e=>{const{kind,data}=JSON.parse(e.data);handleEv(kind,data);};

// Load existing jobs on page load
fetch('/api/jobs').then(r=>r.json()).then(jobs=>{
  jobs.slice().reverse().forEach(j=>{
    if(!jobMap[j.id])addRow(j.id,j.label,j.source);
    if(['done','failed','cancelled'].includes(j.status))finalize(j.id,j.status,j.output_dir);
    else if(j.status==='running'){setStatus(j.id,'running','running');curJobId=j.id;}
  });
});

function handleEv(kind,data){
  if(kind==='job_queued'){addRow(data.id,data.label,data.source);}
  else if(kind==='job_start'){curJobId=data.id;setStatus(data.id,'running','starting…');}
  else if(kind==='job_done'){curJobId=null;finalize(data.id,'done',data.out_dir);}
  else if(kind==='job_error'){curJobId=null;finalize(data.id,'failed',null);}
  else if(kind==='job_cancelled'){if(curJobId===data.id)curJobId=null;finalize(data.id,'cancelled',null);}
  else if(kind==='step'&&curJobId!=null){setProg(curJobId,data.index/(data.total||1),`${data.model} · ${data.action}`);}
  else if(kind==='file_start'&&curJobId!=null){setStatus(curJobId,'running',`[${data.index}/${data.total}] ${data.name}`);setProg(curJobId,0,'');}
  else if(kind==='prefetch'&&curJobId!=null){setStatus(curJobId,'running',`↓ Downloading ${data.model}…`);}
  else if(kind==='file_done'&&curJobId!=null){setProg(curJobId,1,'done');}
}

function addRow(id,label,source){
  const list=document.getElementById('jlist');
  const empty=list.querySelector('.empty');if(empty)empty.remove();
  const d=document.createElement('div');
  d.className='job';d.id=`j${id}`;
  d.innerHTML=`<div class="job-hdr"><span class="badge ${esc(source)}">${esc(source)}</span><span class="jlabel">${esc(label)}</span><span class="jstatus s-queued">queued</span></div><div class="prog-wrap" style="display:none"><div class="prog-bar"><div class="prog-fill" style="width:0"></div></div><div class="step-lbl"></div></div><div class="jact"></div>`;
  list.prepend(d);
  jobMap[id]={el:d};
}
function setStatus(id,cls,text){
  const e=jobMap[id];if(!e)return;
  const s=e.el.querySelector('.jstatus');if(s){s.textContent=text;s.className=`jstatus s-${cls}`;}
  const pw=e.el.querySelector('.prog-wrap');if(pw&&cls==='running')pw.style.display='';
}
function setProg(id,frac,label){
  const e=jobMap[id];if(!e)return;
  const f=e.el.querySelector('.prog-fill');if(f)f.style.width=`${Math.round(frac*100)}%`;
  const l=e.el.querySelector('.step-lbl');if(l&&label)l.textContent=label;
}
function finalize(id,status,outDir){
  const e=jobMap[id];if(!e)return;
  const s=e.el.querySelector('.jstatus');if(s){s.textContent=status;s.className=`jstatus s-${status}`;}
  const pw=e.el.querySelector('.prog-wrap');
  if(status==='done'){if(pw){pw.style.display='';const f=pw.querySelector('.prog-fill');if(f)f.style.width='100%';}}
  else if(pw){pw.style.display='none';}
  const acts=e.el.querySelector('.jact');
  if(status==='done'&&acts){acts.innerHTML=`<a class="dl-btn" href="/api/jobs/${id}/download">Download ZIP</a>`;}
}
function esc(t){return String(t).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');}
</script>
</body>
</html>
"""


# ─────────────────────────────────────────────── FastAPI application factory

def _make_app(registry: "JobRegistry"):
    import re
    import sys
    import traceback as _tb

    try:
        from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
        from fastapi.responses import HTMLResponse, Response, StreamingResponse
    except ImportError as exc:
        raise RuntimeError(
            "Web server dependencies not installed. "
            "Run: pip install -e \".[server]\""
        ) from exc

    from stems.gui.worker import JobParams  # imported once at app-creation time
    from stems.presets import PRESETS

    app = FastAPI(title="Stems", docs_url=None, redoc_url=None)

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return HTMLResponse(_HTML)

    @app.get("/api/presets")
    async def list_presets():
        return [
            {"name": p.name, "kind": p.kind, "description": p.description,
             "output_stems": p.output_stems}
            for p in PRESETS.values()
        ]

    @app.get("/api/jobs")
    async def list_jobs():
        return [
            {
                "id": j.id, "source": j.source, "label": j.label,
                "status": j.status, "created": j.created,
                "started": j.started, "elapsed": j.elapsed,
                "output_dir": str(j.output_dir) if j.output_dir else None,
                "error": j.error,
            }
            for j in registry.get_jobs()
        ]

    @app.get("/api/jobs/{job_id}")
    async def get_job(job_id: int):
        info = registry.get_job(job_id)
        if info is None:
            raise HTTPException(404, "Job not found")
        return {
            "id": info.id, "source": info.source, "label": info.label,
            "status": info.status, "created": info.created,
            "started": info.started, "elapsed": info.elapsed,
            "output_dir": str(info.output_dir) if info.output_dir else None,
            "error": info.error,
        }

    @app.post("/api/jobs")
    async def submit_job(
        file: UploadFile = File(...),
        preset: str = Form(None),
        model: str = Form(None),
        format: str = Form("both"),
        bitdepth: str = Form("24"),
        device: str = Form("auto"),
        tail_rescue: str = Form("true"),
        vocal_method: str = Form(None),
        guitar_source: str = Form(None),
    ):
        try:
            job_id = registry.allocate_id()
            upload_dir = registry.output_root / "uploads"
            upload_dir.mkdir(parents=True, exist_ok=True)

            # Sanitise filename: keep letters, digits, spaces, dots, dashes.
            raw_name = Path(file.filename or "audio").name
            stem_part = re.sub(r'[^\w\s\-.]', '_', Path(raw_name).stem)
            safe_name = stem_part + Path(raw_name).suffix.lower()
            upload_path = upload_dir / f"{job_id}_{safe_name}"

            content = await file.read()
            upload_path.write_bytes(content)

            output_root = registry.output_root / str(job_id)
            output_root.mkdir(parents=True, exist_ok=True)

            params = JobParams(
                input_path=upload_path,
                output_root=output_root,
                preset=preset or None,
                model=model or None,
                engine=None,
                stems=None,
                fmt=format,
                bitdepth=int(bitdepth),
                device=device,
                recursive=False,
                skip_existing=False,
                guitar_source=guitar_source or None,
                vocal_method=vocal_method or None,
                combine=None,
                tail_rescue=(tail_rescue.lower() == "true"),
            )

            actual_id = registry.submit_job(
                params, raw_name, source="web", job_id=job_id
            )
            return {"id": actual_id}

        except Exception as exc:
            _tb.print_exc(file=sys.stderr)
            raise HTTPException(500, f"{type(exc).__name__}: {exc}")

    @app.delete("/api/jobs/{job_id}")
    async def cancel_job(job_id: int):
        info = registry.get_job(job_id)
        if info is None:
            raise HTTPException(404, "Job not found")
        registry.cancel_job(job_id)
        return {"ok": True}

    @app.get("/api/events")
    async def stream_all_events(request: Request):
        sub_q: queue.Queue = queue.Queue()
        registry.subscribe(sub_q)

        async def generate():
            try:
                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        ev = sub_q.get_nowait()
                        payload = json.dumps({"kind": ev.kind, "data": ev.data})
                        yield f"data: {payload}\n\n"
                    except queue.Empty:
                        yield ": keep-alive\n\n"
                        await asyncio.sleep(0.1)
            except (asyncio.CancelledError, GeneratorExit):
                pass
            finally:
                registry.unsubscribe(sub_q)

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    @app.get("/api/jobs/{job_id}/events")
    async def stream_job_events(job_id: int, request: Request):
        info = registry.get_job(job_id)
        if info is None:
            raise HTTPException(404, "Job not found")

        sub_q: queue.Queue = queue.Queue()
        registry.subscribe(sub_q)

        async def generate():
            try:
                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        ev = sub_q.get_nowait()
                        # Forward job-level events only for this job, plus
                        # file/step events (always for the currently-running job).
                        relevant = (
                            ev.data.get("id") == job_id
                            or ev.kind in (
                                "file_start", "step", "file_done",
                                "file_error", "prefetch", "log",
                            )
                        )
                        if relevant:
                            payload = json.dumps({"kind": ev.kind, "data": ev.data})
                            yield f"data: {payload}\n\n"
                    except queue.Empty:
                        yield ": keep-alive\n\n"
                        await asyncio.sleep(0.1)
            except (asyncio.CancelledError, GeneratorExit):
                pass
            finally:
                registry.unsubscribe(sub_q)

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/api/jobs/{job_id}/download")
    async def download_job(job_id: int):
        info = registry.get_job(job_id)
        if info is None or info.status != "done":
            raise HTTPException(404, "Job not found or not complete yet")
        out_dir = info.output_dir
        if out_dir is None or not out_dir.exists():
            raise HTTPException(404, "Output directory not found on server")

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in sorted(out_dir.rglob("*")):
                if f.is_file():
                    zf.write(f, f.relative_to(out_dir))
        buf.seek(0)

        stem_name = info.label.replace(" ", "_").replace("/", "_")
        return Response(
            content=buf.read(),
            media_type="application/zip",
            headers={
                "Content-Disposition": f'attachment; filename="{stem_name}_stems.zip"'
            },
        )

    return app


# ─────────────────────────────────────────────── WebServer wrapper

def _get_lan_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


class WebServer:
    """Runs a FastAPI/uvicorn server in a daemon thread."""

    def __init__(
        self,
        registry: "JobRegistry",
        host: str = "0.0.0.0",
        port: int = 8765,
    ) -> None:
        self._registry = registry
        self._host = host
        self._port = port
        self._server = None
        self._thread: threading.Thread | None = None
        self.lan_ip = _get_lan_ip()

    @property
    def url(self) -> str:
        return f"http://{self.lan_ip}:{self._port}"

    def start(self) -> None:
        try:
            import uvicorn
        except ImportError as exc:
            raise RuntimeError(
                "uvicorn not installed. Run: pip install -e \".[server]\""
            ) from exc

        fastapi_app = _make_app(self._registry)
        config = uvicorn.Config(
            fastapi_app,
            host=self._host,
            port=self._port,
            log_level="error",
            log_config={                    # minimal config that never conflicts with an existing GUI logging setup
                "version": 1,
                "disable_existing_loggers": False,
                "handlers": {
                    "default": {
                        "class": "logging.StreamHandler",
                        "stream": "ext://sys.stderr",
                    }
                },
                "loggers": {
                    "uvicorn.error": {"handlers": ["default"], "level": "ERROR"},
                    "uvicorn.access": {"handlers": [], "level": "ERROR"},
                },
            },
        )
        server = uvicorn.Server(config)
        server.install_signal_handlers = False
        self._server = server
        self._thread = threading.Thread(
            target=server.run, daemon=True, name="stems-web-server"
        )
        self._thread.start()

    def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._server = None
        self._thread = None
