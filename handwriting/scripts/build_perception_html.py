"""Self-contained 2AFC page for the perception pilot (images embedded as WebP).

Each participant sees every line at most once. Per session: N_SHUFFLE orig-vs-shuffle,
N_DETREND orig-vs-detrend, N_FLAT orig-vs-flat and N_CATCH orig-vs-catch trials, order and
top/bottom placement randomised in the browser (seeded by participant id, so a session can
be re-created). Two feedback practice trials (catch lines not used later) come first.
Responses go to the artifact's db (responses/<viewer id>, every 5 trials and at the end);
without it (plain file, signed out) the page offers a download link and a Copy button.
"""
import argparse
import base64
import io
import json
from pathlib import Path

import pandas as pd
from PIL import Image

PAGE = r"""<!DOCTYPE html>
<html lang="vi"><head><meta charset="utf-8"><title>Handwriting line study</title>
<style>
body{font-family:system-ui,sans-serif;max-width:1100px;margin:24px auto;padding:0 16px;color:#222}
.opt{border:2px solid #ccc;border-radius:8px;padding:10px;margin:12px 0;cursor:pointer;background:#fff;text-align:center}
.opt:hover{border-color:#36c}.opt img{max-width:100%;height:auto}
.k{display:inline-block;border:1px solid #999;border-radius:4px;padding:0 6px;font-family:monospace}
#bar{height:6px;background:#eee;border-radius:3px}#bar>div{height:6px;background:#36c;border-radius:3px}
button{font-size:16px;padding:8px 18px}.fb{font-weight:600;margin:8px 0}
</style></head><body>
<div id="app"></div>
<script>
const D = __DATA__;
const app = document.getElementById('app');
function rng(seed){let s=0;for(const c of seed)s=(s*31+c.charCodeAt(0))>>>0;return()=>{s=(s+0x6D2B79F5)>>>0;let t=s;t=Math.imul(t^t>>>15,t|1);t^=t+Math.imul(t^t>>>7,t|61);return((t^t>>>14)>>>0)/4294967296}}
function shuffle(a,r){for(let i=a.length-1;i>0;i--){const j=Math.floor(r()*(i+1));[a[i],a[j]]=[a[j],a[i]]}return a}
let pid='', trials=[], res=[], i=0, t0=0, busy=false;
function intro(){
  app.innerHTML=`<h2>Dòng chữ viết tay nào là bản gốc? / Which handwritten line is the original?</h2>
  <p>Mỗi lượt có <b>hai dòng cùng nội dung, cùng một người viết, cùng hình dạng từng chữ</b>. Một dòng là bản scan gốc. Ở dòng kia, <b>vị trí các từ</b> (độ cao, khoảng cách giữa các từ) đã bị chỉnh lại.
  Hãy chọn dòng mà bạn nghĩ là <b>bản gốc</b>. Nếu không chắc, cứ chọn theo cảm giác.</p>
  <p><i>Each trial shows two lines with the same words, writer and letter shapes. One is the original scan; in the other the word positions (height, spacing) were altered. Pick the original; guess if unsure.</i></p>
  <p>Bấm vào dòng, hoặc dùng phím <span class="k">↑</span> (trên) / <span class="k">↓</span> (dưới). Khoảng ${D.n_total} lượt, mất 10–15 phút.</p>
  <p>Kết quả được lưu tự động vào trang (cần đăng nhập claude.ai). <i>Results are saved automatically.</i></p>
  <p>Mã người tham gia / participant ID: <input id="pid" size="16"> <button onclick="start()">Bắt đầu / Start</button>
  <span id="msg" style="color:#b00"></span></p>`;
}
function start(){
  pid=document.getElementById('pid').value.trim();
  if(!pid){document.getElementById('msg').textContent='Nhập mã / enter an ID';return}
  initStore();
  const r=rng(pid), lines=shuffle(D.lines.map(l=>l.id),r);
  const plan=[]; let k=0;
  for(const [cond,n] of D.plan) for(let j=0;j<n;j++) plan.push({line:lines[k++],cond});
  trials=[...D.practice.map(l=>({line:l,cond:'catch',practice:true})), ...shuffle(plan,r)];
  trials.forEach(t=>t.origTop=r()<0.5);
  i=0; res=[]; show();
}
function show(){
  if(i>=trials.length) return done();
  const t=trials[i], L=D.img[t.line];
  const a=`<div class="opt" onclick="pick(0)"><img src="${t.origTop?L.orig:L[t.cond]}"></div>`;
  const b=`<div class="opt" onclick="pick(1)"><img src="${t.origTop?L[t.cond]:L.orig}"></div>`;
  const n=trials.filter(x=>!x.practice).length, done_=Math.max(0,i-D.practice.length);
  app.innerHTML=`<div id="bar"><div style="width:${100*done_/n}%"></div></div>
  <p>${t.practice?'<b>Lượt thử / practice</b>':`Lượt ${done_+1} / ${n}`}: dòng nào là bản gốc?</p>${a}${b}<div class="fb" id="fb"></div>`;
  busy=false; t0=performance.now();
}
function pick(top){
  if(busy) return; busy=true;
  const t=trials[i], correct=(top===0)===t.origTop;
  res.push({pid, trial:i, line:t.line, cond:t.cond, practice:!!t.practice, orig_top:t.origTop,
            chose_top:top===0, correct, rt_ms:Math.round(performance.now()-t0)});
  if(t.practice){document.getElementById('fb').textContent=correct?'Đúng / Correct':'Sai: bản gốc là dòng '+(t.origTop?'trên / top':'dưới / bottom');
    setTimeout(()=>{i++;show()},1500)} else {i++; if(res.length%5===0) save(false); show()}
}
document.addEventListener('keydown',e=>{if(!trials.length||i>=trials.length)return;
  if(e.key==='ArrowUp')pick(0); if(e.key==='ArrowDown')pick(1)});
// Storage: the artifact's db (one doc per signed-in participant, readable by the owner only).
// Outside an artifact, or if saving fails, fall back to a download link and a Copy button.
let store=null, chain=Promise.resolve(), lastOk=false;
function withTimeout(p,ms){return Promise.race([p,new Promise(r=>setTimeout(()=>r(null),ms))]).catch(()=>null)}
function initStore(){
  if(store||!(window.claude&&window.claude.use)) return;
  store=(async()=>{const db=await withTimeout(window.claude.use('db'),10000);
    const u=await withTimeout(window.claude.use('user'),10000);
    const id=u?await withTimeout(u.id(),10000):null;
    return db&&id?{db,id}:null})();
}
function payload(final){return {pid, version:D.version, ua:navigator.userAgent, final, n:res.length,
  updated:new Date().toISOString(), responses:res}}
function save(final){
  if(!store) return Promise.resolve(false);
  const body=payload(final);
  chain=chain.then(async()=>{const s=await store; if(!s) return lastOk=false;
    try{await s.db.collection('responses').doc(s.id).set(body); lastOk=true}catch(e){console.warn(e); lastOk=false}
    return lastOk});
  return chain;
}
async function done(){
  app.innerHTML='<h2>Đang lưu… / Saving…</h2>';
  const ok=await save(true);
  const json=JSON.stringify(payload(true));
  if(ok){app.innerHTML='<h2>Xong, cảm ơn bạn! / Done, thank you!</h2><p>Kết quả đã được lưu. Bạn có thể đóng trang. <i>Your results are saved; you can close this page.</i></p>';return}
  const url=URL.createObjectURL(new Blob([json],{type:'application/json'}));
  app.innerHTML=`<h2>Xong, cảm ơn bạn! / Done, thank you!</h2>
  <p>Chưa lưu tự động được. Hãy <a href="${url}" download="perception_${pid}.json">tải file kết quả</a>, hoặc bấm Copy rồi dán gửi cho người tổ chức.
  <i>Automatic saving failed: download the file or copy the text below and send it to the organiser.</i></p>
  <button id="cp">Copy</button> <span id="cpm"></span><br><textarea id="js" rows="6" style="width:100%">${json.replace(/</g,'&lt;')}</textarea>`;
  document.getElementById('cp').onclick=()=>{navigator.clipboard.writeText(json).then(()=>cpm.textContent='Đã copy / copied',
    ()=>{const t=document.getElementById('js');t.select();cpm.textContent='Hãy bấm Ctrl+C / press Ctrl+C'})};
}
intro();
</script></body></html>
"""


def webp_uri(path, q):
    buf = io.BytesIO()
    Image.open(path).convert("L").save(buf, "WEBP", quality=q, method=6)
    return "data:image/webp;base64," + base64.b64encode(buf.getvalue()).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stim", type=Path, default=Path("data/perception"))
    ap.add_argument("--out", type=Path, default=Path("reports/perception/study.html"))
    ap.add_argument("--n-shuffle", type=int, default=30)
    ap.add_argument("--n-detrend", type=int, default=14)
    ap.add_argument("--n-flat", type=int, default=14)
    ap.add_argument("--n-catch", type=int, default=6)
    ap.add_argument("--quality", type=int, default=80)
    a = ap.parse_args()
    st = pd.read_csv(a.stim / "stimuli.csv")
    practice = st.line_id.iloc[:2].tolist()
    pool = st.line_id.iloc[2:].tolist()
    plan = [["shuffle", a.n_shuffle], ["detrend", a.n_detrend], ["flat", a.n_flat], ["catch", a.n_catch]]
    assert sum(n for _, n in plan) <= len(pool)
    img = {}
    for lid in practice + pool:
        conds = ["orig", "catch"] if lid in practice else ["orig", "shuffle", "detrend", "flat", "catch"]
        img[lid] = {c: webp_uri(a.stim / f"{lid}__{c}.png", a.quality) for c in conds}
    data = {"version": "pilot-v2", "plan": plan, "practice": practice,
            "lines": [{"id": l} for l in pool], "img": img,
            "n_total": sum(n for _, n in plan) + len(practice)}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(PAGE.replace("__DATA__", json.dumps(data)))
    print(json.dumps({"out": str(a.out), "mb": round(a.out.stat().st_size / 1e6, 1),
                      "pool": len(pool), "per_session": data["n_total"]}))


if __name__ == "__main__":
    main()
