#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""edulab_studio.py — edulab 一键课件工作台 · 后端

纯 Python 标准库实现（无第三方依赖）：
  * 静态托管 GUI（static/index.html）
  * 环境自检（sympy / node / playwright / edge-tts / LLM Key）
  * 图片 OCR（通义千问 qwen-vl-max，走百炼 OpenAI 兼容接口）
  * 题目理解与编排（qwen-plus）
  * 网页技能流水线：立体几何（参数化）/ 解析几何（全量数据）/ 化学（高层 spec）→ HTML
  * 视频技能流水线：scaffold → problem.png → LLM 写 anim/script → 检查 → 配音 → 截图 → 渲染 MP4

用法：  python edulab_studio.py [--port 8620]
配置：  config.json（与本文件同目录；LLM Key 只存在本机，分享前删除）
"""
import argparse
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
import urllib.request
import urllib.error
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs, unquote

APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"
OUT_ROOT = APP_DIR / "输出"
CONFIG_PATH = APP_DIR / "config.json"


def skills_root():
    """技能根目录：优先 WorkBuddy 用户技能目录，其次本应用自带 skills/（随包分发模式）。"""
    for c in (Path.home() / ".workbuddy" / "skills", APP_DIR / "skills"):
        if (c / "edu-solid-geometry" / "scripts" / "generate.py").exists():
            return c
    return None


SKILLS_ROOT = skills_root()

# ---------------------------------------------------------------- 配置
DEFAULT_CONFIG = {
    "dashscope_key": "",      # 阿里云百炼 Key（OCR + LLM），https://bailian.console.aliyun.com/
    "glm_key": "",            # 智谱 GLM_API_KEY（可选，GLM-TTS 音质更好；留空用 edge-tts 兜底）
    "llm_model": "qwen-plus",
    "ocr_model": "qwen-vl-max",
    "port": 8620,
    "tts_engine": "auto",     # auto|glm|edge|say
}


def load_config():
    cfg = dict(DEFAULT_CONFIG)
    if CONFIG_PATH.exists():
        try:
            cfg.update(json.loads(CONFIG_PATH.read_text(encoding="utf-8")))
        except Exception:
            pass
    return cfg


def save_config(cfg):
    CONFIG_PATH.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")


CFG = load_config()

# ---------------------------------------------------------------- 解释器定位
MANAGED_PY = Path.home() / ".workbuddy/binaries/python/envs/default/Scripts/python.exe"
MANAGED_NODE_DIR = Path.home() / ".workbuddy/binaries/node/versions"


def find_python():
    if MANAGED_PY.exists():
        return str(MANAGED_PY)
    return sys.executable


def find_node():
    if MANAGED_NODE_DIR.exists():
        for d in sorted(MANAGED_NODE_DIR.glob("*"), reverse=True):
            exe = d / "node.exe"
            if exe.exists():
                return str(exe)
    for c in ("node", "node.exe"):
        p = shutil.which(c)
        if p:
            return p
    return None


def node_modules_root():
    """从本应用目录向上找含 playwright+ffmpeg-static 的 node_modules。"""
    d = APP_DIR
    while True:
        nm = d / "node_modules"
        if (nm / "playwright").exists() and (nm / "ffmpeg-static").exists():
            return nm
        if d.parent == d:
            return None
        d = d.parent


def read_doc(path, limit=9000):
    try:
        t = Path(path).read_text(encoding="utf-8")
        return t if len(t) <= limit else t[:limit] + "\n……（文档过长，已截断）"
    except Exception:
        return ""


# ---------------------------------------------------------------- 环境自检
def env_status():
    py = find_python()
    st = {"python": py, "node": find_node(), "skills": {}, "deps": {}, "keys": {}}
    for name in ("edu-solid-geometry", "edu-analytic-geometry", "edu-chem-reaction",
                 "edu-math-video", "edu-physics-video", "edu-chem-video"):
        st["skills"][name] = (SKILLS_ROOT / name / "SKILL.md").exists()
    if py:
        probe = subprocess.run(
            [py, "-c", "import sympy,numpy,requests,pypinyin,PIL,edge_tts;print('ok')"],
            capture_output=True, text=True, timeout=60)
        st["deps"]["python_pkgs"] = probe.returncode == 0 and "ok" in probe.stdout
        if probe.returncode != 0:
            st["deps"]["python_pkgs_err"] = probe.stderr.strip()[-300:]
    nm = node_modules_root()
    st["deps"]["node_modules"] = nm is not None
    st["deps"]["node_modules_path"] = str(nm) if nm else None
    st["keys"]["dashscope"] = bool(CFG.get("dashscope_key"))
    st["keys"]["glm"] = bool(CFG.get("glm_key"))
    # pron 词表就位检查（视频技能需要）
    st["deps"]["pron"] = (APP_DIR / "pron.py").exists() or node_modules_root() is not None
    return st


def ensure_pron():
    """把共享读音词表复制到应用根目录（Windows 下 ln -s 会生成 0 字节空文件，必须用 cp）。"""
    src_root = SKILLS_ROOT / "edu-math-video" / "shared"
    for f in ("pron.py", "pron.json"):
        dst = APP_DIR / f
        if not dst.exists() and (src_root / f).exists():
            shutil.copy2(src_root / f, dst)


# ---------------------------------------------------------------- LLM（百炼 OpenAI 兼容接口）
DASHSCOPE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"


def llm_chat(messages, model=None, temperature=0.2, max_tokens=8192, json_mode=False):
    key = CFG.get("dashscope_key", "")
    if not key:
        raise RuntimeError("未配置百炼 Key：请在 config.json 填 dashscope_key（用于题目理解与内容生成）")
    max_tokens = max(1, min(int(max_tokens), 32768))
    body = {"model": model or CFG.get("llm_model", "qwen-plus"),
            "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    req = urllib.request.Request(
        DASHSCOPE_URL,
        data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST")
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            data = json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:500]
        raise RuntimeError(f"LLM 接口 HTTP {e.code}：{detail}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"LLM 接口连接失败：{e}")
    choice = data.get("choices", [{}])[0]
    if "message" not in choice:
        raise RuntimeError(f"LLM 返回异常：{json.dumps(data)[:300]}")
    if choice.get("finish_reason") == "length":
        raise RuntimeError("LLM 输出被截断（finish_reason=length）：请缩小题目或分幕")
    return choice["message"]["content"]


def llm_ocr(image_b64, hint=""):
    """图片 → 题目文字（qwen-vl）。"""
    key = CFG.get("dashscope_key", "")
    if not key:
        raise RuntimeError("未配置百炼 Key：图片识别需要 dashscope_key")
    prompt = ("请把这张图片里的题目完整、逐字转录为文本，保留所有数学符号、数字、单位和条件。"
              "只输出题目本身，不要解释、不要额外文字。")
    if hint:
        prompt += f"\n注意：{hint}"
    body = {"model": CFG.get("ocr_model", "qwen-vl-max"),
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                {"type": "text", "text": prompt}]}],
            "temperature": 0.1, "max_tokens": 4096}
    req = urllib.request.Request(
        DASHSCOPE_URL, data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            data = json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"OCR 接口 HTTP {e.code}：{e.read().decode('utf-8', 'replace')[:300]}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"OCR 接口连接失败：{e}")
    choice = data.get("choices", [{}])[0]
    if "message" not in choice:
        raise RuntimeError(f"OCR 返回异常：{json.dumps(data)[:300]}")
    return choice["message"]["content"].strip()


def extract_json_block(text):
    """从 LLM 回复中稳健提取 JSON 对象。"""
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.S).strip()
    try:
        return json.loads(t)
    except Exception:
        pass
    m = re.search(r"\{.*\}", t, re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    # 逐字符找括号配平
    start = t.find("{")
    if start < 0:
        raise ValueError("回复中找不到 JSON 对象")
    depth, end = 0, None
    in_str, esc = False, False
    for i, ch in enumerate(t[start:], start):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end:
        return json.loads(t[start:end])
    raise ValueError("JSON 括号不配平")


# ---------------------------------------------------------------- 任务管理
class Job:
    def __init__(self, kind, title):
        self.id = uuid.uuid4().hex[:8]
        self.kind = kind
        self.title = title
        self.status = "queued"          # queued|running|done|error
        self.stage = ""
        self.logs = []
        self.artifacts = []
        self.error = ""
        self.created = time.time()

    def log(self, msg, level="info"):
        line = {"t": round(time.time() - self.created, 1), "level": level, "msg": str(msg)}
        self.logs.append(line)
        if len(self.logs) > 500:
            self.logs = self.logs[-400:]

    def to_dict(self):
        return {"id": self.id, "kind": self.kind, "title": self.title, "status": self.status,
                "stage": self.stage, "logs": self.logs, "artifacts": self.artifacts, "error": self.error}


JOBS = {}
JOBS_LOCK = threading.Lock()
RUN_LOCK = threading.Lock()   # 同一时间只跑一个任务（渲染重，避免互相踩）


def new_job(kind, title):
    with JOBS_LOCK:
        j = Job(kind, title)
        JOBS[j.id] = j
    return j


def run_job(job, fn):
    def _worker():
        with RUN_LOCK:
            job.status = "running"
            try:
                fn(job)
                if job.status != "error":
                    job.status = "done"
                    job.stage = "完成"
            except Exception as e:
                job.status = "error"
                job.error = str(e)
                job.log(f"❌ {e}", "error")
                job.log(traceback.format_exc()[-1500:], "error")
    threading.Thread(target=_worker, daemon=True).start()
    return job.id


# ---------------------------------------------------------------- 子进程工具
def run_cmd(job, cmd, cwd=None, env_extra=None, timeout=1800, tail=True):
    job.log("$ " + " ".join(str(c) for c in cmd))
    env = dict(os.environ)
    if env_extra:
        env.update(env_extra)
    proc = subprocess.Popen([str(c) for c in cmd], cwd=str(cwd) if cwd else None,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace", env=env)
    out_lines = []
    for line in proc.stdout:
        line = line.rstrip()
        if line:
            out_lines.append(line)
            job.log(line)
    code = proc.wait(timeout=timeout)
    return code, out_lines


def safe_name(text, maxlen=24):
    s = re.sub(r"[\\/:*?\"<>|\s]+", "_", text.strip())
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:maxlen] or "题目"


def now_str():
    return dt.datetime.now().strftime("%m%d_%H%M")


# ---------------------------------------------------------------- 流水线：网页技能
def pipeline_web(job, skill, mode, payload, text):
    py = find_python()
    if not py:
        raise RuntimeError("找不到 Python 解释器")
    skill_dir = SKILLS_ROOT / skill
    if not skill_dir.exists():
        raise RuntimeError(f"技能目录不存在：{skill_dir}")

    job.stage = "理解题目 / 选题型"
    out_dir = OUT_ROOT / f"{now_str()}_{skill}_{safe_name(text or 'preset')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    html_out = out_dir / "solution.html"

    if mode == "preset":
        key = payload.get("key", "")
        code, _ = run_cmd(job, [py, skill_dir / "scripts" / "generate.py", key, str(html_out)])
        if code != 0:
            raise RuntimeError(f"generate.py 退出码 {code}")
    else:
        if not CFG.get("dashscope_key"):
            raise RuntimeError("自定义题目需要百炼 Key（题目理解）。也可直接用下方【预设题库】一键体验。")
        jt = text.strip()
        if not jt:
            raise RuntimeError("请先输入题目文字（或上传图片识别）")
        if skill == "edu-solid-geometry":
            plan = llm_plan_solid(jt)
            job.log(f"题型判定：{plan.get('key')}  参数：{plan.get('params')}  理由：{plan.get('reason', '')}")
            cmd = [py, APP_DIR / "bridge_web.py", f"solid-{plan['key'].replace('solid-', '')}",
                   "--params", json.dumps(plan.get("params", {}), ensure_ascii=False),
                   "--out", str(html_out)]
            code, _ = run_cmd(job, cmd)
            if code != 0:
                raise RuntimeError("立体几何生成失败（参数不被内核支持？）")
        elif skill == "edu-analytic-geometry":
            data = llm_plan_analytic(jt)
            code, _ = run_cmd(job, [py, APP_DIR / "bridge_web.py", "analytic",
                                    "--data", json.dumps(data, ensure_ascii=False),
                                    "--out", str(html_out)])
            if code != 0:
                raise RuntimeError("解析几何数据装配失败")
        elif skill == "edu-chem-reaction":
            spec, err = llm_plan_chem(jt), ""
            for attempt in range(2):
                code, out = run_cmd(job, [py, APP_DIR / "bridge_web.py", "chem",
                                          "--spec", json.dumps(spec, ensure_ascii=False),
                                          "--out", str(html_out)])
                if code == 0:
                    break
                err = "\n".join(out)[-800:]
                job.log(f"化学 spec 校验失败，重试第 {attempt + 1} 次：{err[:300]}", "warn")
                spec = llm_plan_chem(jt, error_feedback=err)
            else:
                raise RuntimeError(f"化学反应装配失败：{err[:300]}")

    # 自检摘要
    size = html_out.stat().st_size if html_out.exists() else 0
    job.log(f"网页生成完毕：{html_out}（{size // 1024} KB）")
    job.artifacts = [
        {"type": "html", "name": "交互课件网页", "path": str(html_out)},
        {"type": "dir", "name": "输出目录", "path": str(out_dir)},
    ]


def llm_plan_solid(text):
    sys_prompt = (
        "你是立体几何题型判定器。把用户的题目映射到可用内核题型，输出 JSON："
        '{"key":"pyramid|cube|box|random","params":{...},"reason":"..."}\n'
        "规则：\n"
        "- pyramid：正四棱锥 P-ABCD，E 为 PC 中点，求（判断）线面角。params: base_edge(1-12 整数), height(0.5-12)\n"
        "- cube：正方体棱长 edge(1-12 整数)，求 A1C 与底面 ABCD 所成角的正弦值\n"
        "- box：长方体 lx/ly/lz(1-12 整数)，求体积\n"
        "- 只有当题目确实是上述题型时才给对应 key；否则给出最接近的 key，并在 reason 说明差异\n"
        "- 从题干提取数字作为参数；没有数字时用常见值（正方体棱长 2）\n"
        "- 只输出 JSON，不要解释")
    content = llm_chat([{"role": "system", "content": sys_prompt},
                        {"role": "user", "content": text}], json_mode=True, temperature=0.1)
    plan = extract_json_block(content)
    if plan.get("key") not in ("pyramid", "cube", "box", "random"):
        plan["key"] = "pyramid"
    return plan


def llm_plan_analytic(text):
    schema = read_doc(SKILLS_ROOT / "edu-analytic-geometry" / "references" / "problem-schema.md", 12000)
    sys_prompt = (
        "你是解析几何课件数据生成器。根据用户题目，产出注入 board.html 的完整 JSON 数据"
        "（对象含 lesson / steps / board 三段）。数值必须自己用准确的数学计算（可用 sympy 心算或手算），"
        "答案要正确；步骤 content 用 HTML（公式 $…$ / $$…$$）。board 的交互引擎 schema 如下，"
        "按题型选用 rangeBar / constant / answerBand / trace 之一：\n\n" + schema +
        "\n\n只输出 JSON 对象本身，不要 markdown 代码块、不要解释。")
    content = llm_chat([{"role": "system", "content": sys_prompt},
                        {"role": "user", "content": "题目：" + text}],
                       json_mode=True, temperature=0.3, max_tokens=16384)
    return extract_json_block(content)


def llm_plan_chem(text, error_feedback=""):
    species = None
    try:
        code, out = run_cmd_local([find_python(), "-c",
                                   "import sys;sys.path.insert(0,r'" + str(SKILLS_ROOT / 'edu-chem-reaction' / 'lib') +
                                   "');import molecules as M;print(','.join(M.known_species()))"])
        species = out[-1] if out else ""
    except Exception:
        pass
    sys_prompt = (
        "你是化学反应演示 spec 生成器。把用户的反应写成高层 spec（JSON 对象）：\n"
        '{"meta":{"title":"...","subtitle":"...","category":"junior|senior","accent":"amber|sky|rose|violet|emerald","engine":"morph|mechanism"},'
        '"conditions":{"text":"点燃","exothermic":true,"flame":true},'
        '"reactants":[{"species":"CH4","count":1,"pos":[-4,0.2,0]},{"species":"O2","count":2}],'
        '"products":[{"species":"CO2","count":1,"pos":[-2.7,1.1,0]},{"species":"H2O","count":2}],'
        '"atom_map":[["CH4#1.C","CO2#1.Y"],["O2#1.Oa","CO2#1.Xa"],["CH4#1.H1","H2O#1.Ha"]],'
        '"energy":{"activation":0.5,"deltaH":-0.75,"reactantLabel":"CH4+O2","productLabel":"CO2+H2O"},'
        '"steps":[{"title":"...","html":"<span>...</span> 讲解（可含元素高亮 span）"}]}\n'
        "规则：\n"
        "- species 只能取自 3D 模型库：" + (species or "C,CH4,CO,CO2,Cl2,Fe,Glucose,H2,H2O,Mg,N2,NH3,Na,NaCl,O2") + "\n"
        "- atom_map 把每个反应物原子编号映射到产物原子编号（格式 物种#序号.槽位，如 CH4#1.C、O2#1.Oa、H2O#1.Ha）；"
        "必须一一对应且守恒（前端会做 sympy 校验，失败会重试）\n"
        "- counts/实例数要与 atom_map 一致；pos 只给少数实例排版用，可省略\n"
        "- steps 给 3 步讲解：反应物 → 断键/反应条件 → 产物与守恒\n"
        "- 有机机理（如酯化反应）用 engine=mechanism，并补充 fragments/catalyst（没有把握就用 morph）\n"
        "- 只输出 JSON，不要解释")
    if error_feedback:
        sys_prompt += "\n上一次生成的 spec 校验失败，错误信息：\n" + error_feedback + "\n请修正后重新输出。"
    content = llm_chat([{"role": "system", "content": sys_prompt},
                        {"role": "user", "content": "反应：" + text}], json_mode=True, temperature=0.3)
    return extract_json_block(content)


def run_cmd_local(cmd, timeout=60):
    proc = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, timeout=timeout)
    return proc.returncode, proc.stdout.strip().splitlines()


# ---------------------------------------------------------------- 流水线：视频技能
SKILL_VIDEO_DOCS = {
    "edu-math-video": ["reference/animation.md", "reference/script-writing.md", "reference/visual-design.md"],
    "edu-physics-video": ["reference/animation.md", "reference/script-writing.md", "reference/visual-design.md"],
    "edu-chem-video": ["reference/animation.md", "reference/script-writing.md", "reference/visual-design.md",
                       "reference/chemistry-solving.md"],
}


def pipeline_video(job, skill, text, image_b64=None):
    py, node = find_python(), find_node()
    if not node:
        raise RuntimeError("找不到 Node.js（视频渲染需要）")
    if node_modules_root() is None:
        raise RuntimeError("未找到 node_modules（playwright + ffmpeg-static）。请先在本应用目录执行："
                           "npm install playwright@1.49 ffmpeg-static && npx playwright install chromium")
    ensure_pron()
    skill_dir = SKILLS_ROOT / skill
    tpl = skill_dir / "template"
    if not (tpl / "engine.js").exists():
        raise RuntimeError(f"技能模板不完整：{tpl}")

    problem = text.strip()
    if image_b64:
        job.stage = "识别图片中的题目"
        problem = llm_ocr(image_b64)
        job.log(f"OCR 结果：{problem[:200]}")
    if not problem:
        raise RuntimeError("请先输入题目文字（或上传图片识别）")

    proj_name = f"video_{now_str()}_{safe_name(problem)}"
    proj = OUT_ROOT / proj_name
    proj.mkdir(parents=True, exist_ok=True)
    job.stage = "搭建项目"

    # 1) scaffold：复制模板（不带 build/ 与 node_modules）
    for f in tpl.iterdir():
        if f.name in ("build", "node_modules"):
            continue
        dst = proj / f.name
        if f.is_dir():
            shutil.copytree(f, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(f, dst)
    for f in ("pron.py", "pron.json"):
        src = APP_DIR / f
        if src.exists() and not (proj / f).exists():
            shutil.copy2(src, proj / f)
    job.log(f"项目目录：{proj}")

    # 2) LLM 先规划：幕/旁白/分镜（同时给出要在题图上高亮的条件短语）
    job.stage = "规划解题与分镜"
    plan = llm_video_plan(job, skill, problem)
    marks = plan.pop("mark_phrases", []) or []
    for name in ("episode.json", "script.json", "storyboard.md"):
        (proj / name).write_text(plan[name], encoding="utf-8")
        job.log(f"已写出 {name}（{len(plan[name])} 字符）")
    fix_storyboard(proj / "storyboard.md")

    # 3) problem.png：文字题渲染成图（带高亮框）；图片题直接用原图
    if image_b64:
        img = proj / "problem.png"
        img.write_bytes(__import__("base64").b64decode(image_b64))
        from PIL import Image  # noqa
        with Image.open(img) as im:
            src_w, src_h = im.size
        boxes = {}
        job.log(f"使用上传的题目图片：{src_w}×{src_h}")
    else:
        job.stage = "生成题目图片"
        cmd = [py, skill_dir / "scripts" / "make_problem_png.py",
               "--text", problem, "--out", str(proj / "problem.png")]
        for m in marks[:6]:
            if m and m in problem:
                cmd += ["--mark", m]
        code, out = run_cmd(job, cmd)
        if code != 0 or not (proj / "problem.png").exists():
            raise RuntimeError("题目图片生成失败")
        try:
            meta = extract_json_block(out[-1])
            src_w, src_h = meta.get("srcW", 1500), meta.get("srcH", 800)
            boxes = meta.get("boxes", {})
        except Exception:
            src_w, src_h, boxes = 1500, 800, {}
        job.log(f"题目图片就绪：{src_w}×{src_h}，高亮框 {len(boxes)} 个")

    # 4) LLM 写 anim.js
    job.stage = "编写动画脚本"
    anim = llm_video_anim(job, skill, problem, src_w, src_h, boxes, plan)
    (proj / "anim.js").write_text(anim, encoding="utf-8")
    job.log(f"已写出 anim.js（{len(anim)} 字符）")

    # 5) 检查（可带错误反馈整体重来一次）
    env_extra = {"TTS_ENGINE": CFG.get("tts_engine", "auto"), "PYTHONIOENCODING": "utf-8"}
    if CFG.get("glm_key"):
        env_extra["GLM_API_KEY"] = CFG["glm_key"]
    job.stage = "脚本与读音检查"
    for attempt in range(2):
        code, out = run_cmd(job, [py, "build_audio.py", "--check"], cwd=proj, env_extra=env_extra)
        if code == 0:
            break
        err = "\n".join(out)[-1200:]
        job.log("检查未通过，让模型修订…", "warn")
        plan = llm_video_plan(job, skill, problem, error_feedback=err)
        marks = plan.pop("mark_phrases", []) or []
        for name in ("episode.json", "script.json", "storyboard.md"):
            (proj / name).write_text(plan[name], encoding="utf-8")
        fix_storyboard(proj / "storyboard.md")
        anim = llm_video_anim(job, skill, problem, src_w, src_h, boxes, plan, error_feedback=err)
        (proj / "anim.js").write_text(anim, encoding="utf-8")
    else:
        raise RuntimeError("脚本检查未通过，请把上方日志发给开发者或手动修改 script.json / anim.js")

    # 6) 免费预览：构图截图（不花钱）
    job.stage = "免费预览截图"
    code, _ = run_cmd(job, [py, "build_audio.py", "--preview"], cwd=proj, env_extra=env_extra)
    if code != 0:
        raise RuntimeError("预览时间线生成失败")
    code, _ = run_cmd(job, [node, "render.mjs", "stills", "auto"], cwd=proj)
    if code != 0:
        raise RuntimeError("渲染截图失败（anim.js 可能有 JS 错误，见日志）")
    run_cmd(job, [py, skill_dir / "scripts" / "contact_sheet.py", str(proj)], cwd=proj)
    sheets = sorted((proj / "build" / "stills").glob("*.png"))
    job.log(f"预览截图 {len(sheets)} 张：" + ", ".join(s.name for s in sheets[:8]))

    # 7) 配音
    job.stage = "生成配音（TTS）"
    code, _ = run_cmd(job, [py, "build_audio.py"], cwd=proj, env_extra=env_extra, timeout=3600)
    if code != 0:
        raise RuntimeError("配音失败（网络或 TTS 引擎问题，见日志）")

    # 8) 渲染 MP4
    job.stage = "渲染视频（逐帧，请耐心）"
    workers = 4
    code, _ = run_cmd(job, [node, "render.mjs", "video", str(workers)], cwd=proj, timeout=5400)
    if code != 0:
        raise RuntimeError("视频渲染失败，见日志")

    ep = json.loads((proj / "episode.json").read_text(encoding="utf-8"))
    out_name = ep.get("output_name", "video")
    mp4 = proj.parent / f"{out_name}.mp4"
    srt = proj.parent / f"{out_name}.srt"
    arts = []
    if mp4.exists():
        arts.append({"type": "mp4", "name": f"{out_name}.mp4", "path": str(mp4)})
    if srt.exists():
        arts.append({"type": "srt", "name": f"{out_name}.srt", "path": str(srt)})
    for s in sheets[:4]:
        arts.append({"type": "img", "name": f"预览·{s.name}", "path": str(s)})
    arts.append({"type": "dir", "name": "项目目录", "path": str(proj)})
    arts.append({"type": "dir", "name": "视频所在目录", "path": str(proj.parent)})
    job.artifacts = arts
    job.log(f"🎬 完成：{mp4}")


def _video_docs(skill):
    docs = ""
    for d in SKILL_VIDEO_DOCS.get(skill, []):
        docs += f"\n===== {d} =====\n" + read_doc(SKILLS_ROOT / skill / d, 6000)
    return docs


STORY_HEADER = "| 幕id 句号 | 旁白要点 | 指 | 动 | 留 | 板书 |"


def fix_storyboard(path):
    """确保 storyboard.md 的分镜表带标准表头（LLM 常漏掉表头行，检查器强制要求）。"""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except Exception:
        return
    lines = text.splitlines()
    table_idx = [i for i, ln in enumerate(lines) if ln.lstrip().startswith("|")]
    if not table_idx:
        return
    first = table_idx[0]
    # 已在表头区找到「动」列 → 无需修复
    header_zone = lines[first:first + 3]
    if any(any(c.strip().startswith("动") for c in ln.strip().strip("|").split("|")) for ln in header_zone):
        return
    lines.insert(first, STORY_HEADER)
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def llm_video_plan(job, skill, problem, error_feedback=""):
    """第一次 LLM 调用：episode.json + script.json + storyboard.md + 高亮短语。"""
    sys_prompt = (
        f"你是 edulab {skill} 技能的讲题视频策划。根据题目先完整解对题，再产出 JSON 对象：\n"
        '{"episode.json":"...","script.json":"...","storyboard.md":"...","mark_phrases":["条件1","所求"]}\n'
        "要求：\n"
        "- episode.json：{\"title\":\"视频标题\",\"output_name\":\"中文名_English_Name\",\"pop_scenes\":[\"可空\"]}\n"
        "- script.json：幕数组 [{\"scene\":\"英文id\",\"lines\":[{\"zh\":\"中文字幕\",\"tts\":\"口语稿\",\"en\":\"English\"}]}]；"
        "第一幕 intro 逐条读题（每条读一个条件），最后一幕 outro 回顾并报答案；每句只讲一件事，≤36 字宽\n"
        "- tts 只能念中文：数字汉字化（AC等于三、十九点二克）、化学式念名称（HNO3→硝酸）、单位念出来\n"
        "- storyboard.md：顶部写【解题卡】（已知/所求/关键步骤/答案）与【图形清单】，然后每句旁白一行表格；"
        "表格第一行表头必须严格为：| 幕id 句号 | 旁白要点 | 指 | 动 | 留 | 板书 |"
        "除首尾幕外每句的『动』必须是真实运动（滑/移/转/飞/计数…），不能只写出现/显示/高亮\n"
        "- mark_phrases：从原题中挑 3~5 个要在题图上框出的关键短语（必须是题干原文子串，如『19.2 g 铜』『足量稀硝酸』『标准状况』）\n"
        "- 解题必须正确（化学题先配平再计算；数值带单位）\n"
        "- 只输出 JSON 对象，4 个键的值都是字符串（注意 JSON 转义）\n" + _video_docs(skill))
    if error_feedback:
        sys_prompt += "\n\n上一次生成未通过检查，错误信息：\n" + error_feedback + "\n请修正后重新输出。"
    content = llm_chat([{"role": "system", "content": sys_prompt},
                        {"role": "user", "content": "题目：\n" + problem}],
                       temperature=0.3, max_tokens=24576)
    files = extract_json_block(content)
    need = {"episode.json", "script.json", "storyboard.md"}
    missing = need - set(files)
    if missing:
        raise RuntimeError(f"LLM 未产出全部策划文件，缺少：{', '.join(missing)}")
    return files


def llm_video_anim(job, skill, problem, src_w, src_h, boxes, plan, error_feedback=""):
    """第二次 LLM 调用：按策划与题目图数据写 anim.js。"""
    tpl = SKILLS_ROOT / skill / "template"
    anim_example = read_doc(tpl / "anim.js", 11000)
    script = plan.get("script.json", "")
    sys_prompt = (
        f"你是 edulab {skill} 技能的 canvas 动画工程师。为下面这道题写 anim.js（只输出 JS 代码本身，"
        "不要 markdown 代码块标记、不要解释）。\n"
        "硬性要求：\n"
        "- 结构：PROBLEM（题目图数据）、TAGS（每幕标签）、若干图形函数、SC.<id> = (lt, S) => {...}（每幕一个，id 与 script.json 一致）\n"
        "- 所有动画用 S.at(k, f)（第 k 句旁白开始后 f 比例处）定时，禁止写死秒数\n"
        f"- 每句旁白：点亮提到元素 → 一个体现推理的动作 → 留下标记；除首尾幕外不能只有出现/显示\n"
        "- 内容全部在 y ≤ 860（y≈914~1044 是字幕框）；`engine.js` 已有全部绘图 API，直接用\n"
        f"- 题目图 PROBLEM.srcW={src_w}, srcH={src_h}, src=\"problem.png\", boxes={json.dumps(boxes, ensure_ascii=False)[:800]}\n"
        "- 多音字在字后标 [拼音] 只出现在 tts/字幕文本处；不要在 JS 注释外写中文以外的多余内容\n"
        "- 修改重试时注意保持与 storyboard.md 的分镜一致\n"
        + _video_docs(skill) +
        f"\n===== template/anim.js 结构示例（仿照其结构）=====\n{anim_example}")
    if error_feedback:
        sys_prompt += "\n\n上一次 anim.js 未通过检查，错误信息：\n" + error_feedback + "\n请修正后重新输出完整 anim.js。"
    user = "题目：\n" + problem + "\n\nscript.json：\n" + script[:12000]
    content = llm_chat([{"role": "system", "content": sys_prompt}, {"role": "user", "content": user}],
                       temperature=0.3, max_tokens=32768)
    js = content.strip()
    js = re.sub(r"^```(?:javascript|js)?\s*|\s*```$", "", js, flags=re.S).strip()
    if "SC." not in js or "PROBLEM" not in js:
        raise RuntimeError("anim.js 缺少 SC.<id> 场景或 PROBLEM 数据")
    return js


# ---------------------------------------------------------------- HTTP 服务
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _send_json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path, ctype=None):
        path = Path(path)
        if not path.exists():
            self._send_json({"error": "not found"}, 404)
            return
        data = path.read_bytes()
        if ctype is None:
            suffix = path.suffix.lower()
            ctype = {".html": "text/html; charset=utf-8", ".js": "text/javascript",
                     ".css": "text/css", ".png": "image/png", ".jpg": "image/jpeg",
                     ".mp4": "video/mp4", ".srt": "text/plain; charset=utf-8",
                     ".json": "application/json", ".md": "text/plain; charset=utf-8"}.get(suffix,
                                                                                          "application/octet-stream")
        total = len(data)
        rng = self.headers.get("Range")
        if rng and total > 0:
            m = re.match(r"bytes=(\d*)-(\d*)", rng.strip())
            if m:
                start = int(m.group(1)) if m.group(1) else 0
                end = int(m.group(2)) if m.group(2) else total - 1
                end = min(end, total - 1)
                chunk = data[start:end + 1]
                self.send_response(206)
                self.send_header("Content-Type", ctype)
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Range", f"bytes {start}-{end}/{total}")
                self.send_header("Content-Length", str(len(chunk)))
                self.end_headers()
                self.wfile.write(chunk)
                return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(total))
        self.end_headers()
        self.wfile.write(data)

    def _body(self):
        n = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(n) if n else b"{}"
        return json.loads(raw.decode("utf-8"))

    def do_GET(self):
        url = urlparse(self.path)
        path = unquote(url.path)
        if path in ("/", "/index.html"):
            self._send_file(STATIC_DIR / "index.html", "text/html; charset=utf-8")
        elif path == "/api/status":
            self._send_json(env_status())
        elif path.startswith("/api/job/"):
            jid = path.rsplit("/", 1)[-1]
            with JOBS_LOCK:
                job = JOBS.get(jid)
            self._send_json(job.to_dict() if job else {"error": "job not found"}, 200 if job else 404)
        elif path == "/api/jobs":
            with JOBS_LOCK:
                self._send_json([j.to_dict() for j in sorted(JOBS.values(), key=lambda x: -x.created)[:20]])
        elif path == "/api/out":
            q = parse_qs(url.query)
            p = q.get("path", [""])[0]
            target = Path(p).resolve()
            roots = [OUT_ROOT.resolve(), APP_DIR.resolve()]
            if not any(str(target).startswith(str(r)) for r in roots):
                self._send_json({"error": "path outside allowed roots"}, 403)
                return
            self._send_file(target)
        else:
            self._send_json({"error": "not found"}, 404)

    def do_POST(self):
        url = urlparse(self.path)
        path = url.path
        try:
            body = self._body()
        except Exception:
            self._send_json({"error": "bad json"}, 400)
            return
        if path == "/api/ocr":
            try:
                text = llm_ocr(body.get("image", ""))
                self._send_json({"text": text})
            except Exception as e:
                self._send_json({"error": str(e)}, 500)
        elif path == "/api/generate":
            skill = body.get("skill", "")
            mode = body.get("mode", "custom")
            text = body.get("text", "")
            image = body.get("image", "")
            payload = body.get("payload", {})
            if skill not in env_status()["skills"]:
                self._send_json({"error": f"未知技能 {skill}"}, 400)
                return
            kind = "web" if skill in ("edu-solid-geometry", "edu-analytic-geometry", "edu-chem-reaction") else "video"
            job = new_job(kind, f"{skill} · {(text or '预设题库')[:30]}")
            if kind == "web":
                run_job(job, lambda j: pipeline_web(j, skill, mode, payload, text))
            else:
                run_job(job, lambda j: pipeline_video(j, skill, text, image or None))
            self._send_json({"job_id": job.id})
        elif path == "/api/config":
            try:
                for k in ("dashscope_key", "glm_key", "llm_model", "ocr_model", "tts_engine"):
                    if k in body:
                        CFG[k] = body[k]
                save_config(CFG)
                self._send_json({"ok": True, "keys": {"dashscope": bool(CFG.get("dashscope_key")),
                                                     "glm": bool(CFG.get("glm_key"))}})
            except Exception as e:
                self._send_json({"error": str(e)}, 500)
        else:
            self._send_json({"error": "not found"}, 404)


def pick_port(preferred):
    import socket
    for p in range(preferred, preferred + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", p))
                return p
            except OSError:
                continue
    return preferred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=CFG.get("port", 8620))
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    ensure_pron()
    port = pick_port(args.port)
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}"
    print("=" * 56)
    print("  edulab 一键课件工作台")
    print(f"  地址: {url}")
    print(f"  技能目录: {SKILLS_ROOT}")
    print(f"  输出目录: {OUT_ROOT}")
    print("  按 Ctrl+C 停止")
    print("=" * 56)
    if not args.no_browser:
        threading.Timer(0.8, lambda: __import__("webbrowser").open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")


if __name__ == "__main__":
    main()
