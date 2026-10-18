#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""bridge_web.py — edulab 工作台 · 网页技能桥接层

把「参数 / 数据规格」变成交互网页 HTML，全程 sympy 精确计算（答案与画面同源）。

用法（全部从 stdin 或 --json 读 JSON，结果写到 --out，日志走 stdout/stderr）：

  # 立体几何 · 自定义参数（正四棱锥底面边长 a、高 h，求线面角）
  python bridge_web.py solid-pyramid --params '{"base_edge":4,"height":2}' --out sol.html

  # 立体几何 · 正方体（棱长 edge）线面角 / 长方体（lx,ly,lz）体积
  python bridge_web.py solid-cube  --params '{"edge":2}' --out sol.html
  python bridge_web.py solid-box   --params '{"lx":3,"ly":4,"lz":5}' --out sol.html

  # 立体几何 · 随机出题（seed 可固定）
  python bridge_web.py solid-random --params '{"seed":7}' --out sol.html

  # 解析几何 · LLM 产出的完整 board 数据（schema 见技能 references/problem-schema.md）
  python bridge_web.py analytic --data '{"lesson":{...},"steps":[...],"board":{...}}' --out sol.html

  # 化学 · LLM 产出的高层反应 spec（species + atom_map + steps，配平由 sympy 校验）
  python bridge_web.py chem --spec '{"reactants":[...],"products":[...],"atom_map":[...],"steps":[...]}' --out sol.html

退出码：0 成功；1 参数/数据错误（stderr 有可读原因）；2 环境错误（缺 sympy 等）。
"""
import argparse
import json
import sys
import traceback
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent


def _skills_root():
    """定位技能根目录：优先 ~/.workbuddy/skills，其次本应用自带 skills/ 目录。"""
    candidates = [
        Path.home() / ".workbuddy" / "skills",
        APP_DIR / "skills",
        APP_DIR.parent / "skills",
    ]
    for c in candidates:
        if (c / "edu-solid-geometry" / "scripts" / "generate.py").exists():
            return c
    return None


SKILLS = _skills_root()


def _fail(msg, code=1):
    print(f"BRIDGE-ERROR: {msg}", file=sys.stderr)
    sys.exit(code)


def _load_json_arg(text, name):
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        _fail(f"{name} 不是合法 JSON：{e}（常见原因：中文引号“”、尾随逗号、注释）")


def _num(v, name, lo=None, hi=None, integer=False):
    try:
        x = int(v) if integer else float(v)
    except (TypeError, ValueError):
        _fail(f"参数 {name} 需要数字，收到 {v!r}")
    if lo is not None and x < lo:
        _fail(f"参数 {name} 不能小于 {lo}（收到 {x}）")
    if hi is not None and x > hi:
        _fail(f"参数 {name} 不能大于 {hi}（收到 {x}）")
    return x


# ---------------------------------------------------------------- 立体几何
def cmd_solid(args):
    if SKILLS is None:
        _fail("找不到 edu-solid-geometry 技能目录（~/.workbuddy/skills/）", 2)
    skill = SKILLS / "edu-solid-geometry"
    sys.path.insert(0, str(skill / "lib"))
    sys.path.insert(0, str(skill / "scripts"))
    try:
        import generate  # noqa: E402
    except Exception as e:
        _fail(f"导入技能失败（缺 sympy？）：{e}", 2)

    p = _load_json_arg(args.params, "params") or {}
    kind = args.kind
    if kind == "solid-pyramid":
        data = generate.build_data(
            base_edge=_num(p.get("base_edge", 2), "base_edge", 1, 12, integer=True),
            height=_num(p.get("height", 1), "height", 0.5, 12),
        )
    elif kind == "solid-cube":
        edge = _num(p.get("edge", 1), "edge", 1, 12, integer=True)
        import geometry_kernel as gk  # noqa: E402
        import bodies  # noqa: E402
        sol = gk.solve_cube_line_plane_angle(edge=edge, scale=2)
        data = _build_cube_p(gk, bodies, sol, edge)
    elif kind == "solid-box":
        data = generate.build_box_volume_data(
            lx=_num(p.get("lx", 3), "lx", 1, 12, integer=True),
            ly=_num(p.get("ly", 4), "ly", 1, 12, integer=True),
            lz=_num(p.get("lz", 5), "lz", 1, 12, integer=True),
        )
    elif kind == "solid-random":
        data = generate.build_random_data(seed=_num(p.get("seed", 0), "seed", 0, 10 ** 9, integer=True))
    else:
        _fail(f"未知立体几何题型 {kind}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    generate.render_html(data, out)
    print(f"OK solid {out}")


def _build_cube_p(gk, bodies, sol, edge):
    """参数化正方体线面角 builder（与技能 build_cube_data 同构，棱长可调）。"""
    mp, v, ans, tp = sol["math_points"], sol["vals"], sol["answer_latex"], sol["three_points"]
    topo = bodies.cuboid()
    names = list(tp)
    center = [sum(tp[k][i] for k in names) / len(names) for i in range(3)]
    e = str(edge)

    model = {
        "target": center,
        "initialCamera": [6, 5, 7],
        "points": tp,
        "spheres": topo["spheres"],
        "edges": topo["edges"],
        "elements": {
            "Line_A1C": {"type": "line", "a": "A1", "b": "C", "color": "emphasis", "depthTest": False},
            "Plane_ABCD": {"type": "plane", "pts": ["A", "B", "C", "D"]},
            "Normal_Vector": {"type": "arrow", "origin": "A", "dir": [0, 1, 0], "length": 1.6, "color": "normal"},
            "Axis": {"type": "axes", "size": 2.6},
        },
    }
    lesson = {
        "language": "zh-CN",
        "meta": "交互解题 · 线面角",
        "title": f"正方体ABCD-A₁B₁C₁D₁中，棱长为{e}，求直线A₁C与底面ABCD所成角的正弦值",
        "answerLabel": "直线 A₁C 与底面 ABCD 所成角的正弦值",
        "answerValue": f"${ans}$",
    }
    steps = [
        {"title": "建立空间直角坐标系",
         "content": (r"<p>以顶点 $A$ 为原点，$AB$、$AD$、$AA_1$ 分别为 $x$、$y$、$z$ 轴建立坐标系。</p>"
                     r"<p>因为棱长为 $" + e + r"$，关键点坐标为：</p>"
                     r"$$A" + mp["A"] + r", C" + mp["C"] + r", A_1" + mp["A1"] + r"$$"),
         "highlight": ["Axis"], "cameraPos": {"x": 6, "y": 5, "z": 7}},
        {"title": "求直线 A₁C 的方向向量",
         "content": (r"<p>直线 $A_1C$ 的方向向量为：</p>"
                     r"$$\vec{A_1C} = C - A_1 = " + v["A1C"] + r"$$"),
         "highlight": ["Line_A1C"], "cameraPos": {"x": 5, "y": 4, "z": 8}},
        {"title": "确定底面 ABCD 的法向量",
         "content": (r"<p>底面 $ABCD$ 在平面 $z = 0$ 上，其法向量竖直向上：</p>"
                     r"$$\vec{n} = " + v["n_simpl"] + r"$$"),
         "highlight": ["Line_A1C", "Plane_ABCD", "Normal_Vector"], "cameraPos": {"x": 4, "y": 6, "z": 5}},
        {"title": "利用向量公式求解",
         "content": (r"<p>设直线 $A_1C$ 与底面所成角为 $\theta$。</p>"
                     r"<p>线面角公式：$\sin\theta = \dfrac{|\vec{A_1C} \cdot \vec{n}|}{|\vec{A_1C}|\,|\vec{n}|}$</p>"
                     r"<p>计算数量积与模长：</p>"
                     r"$$\vec{A_1C} \cdot \vec{n} = " + v["dot"] + r", \quad |\vec{A_1C}| = " + v["norm_line"] + r"$$"
                     r"<p>代入公式：</p>"
                     r"$$\sin\theta = " + v["sin"] + r"$$"
                     r"<p>所以，直线 $A_1C$ 与底面 $ABCD$ 所成角的正弦值为 $" + ans + r"$。</p>"),
         "highlight": ["Line_A1C", "Plane_ABCD", "Normal_Vector"], "cameraPos": {"x": 5, "y": 5, "z": 6}},
    ]
    return {"lesson": lesson, "steps": steps, "model": model, "_answer": ans}


# ---------------------------------------------------------------- 解析几何
def cmd_analytic(args):
    if SKILLS is None:
        _fail("找不到 edu-analytic-geometry 技能目录", 2)
    skill = SKILLS / "edu-analytic-geometry"
    sys.path.insert(0, str(skill / "lib"))
    sys.path.insert(0, str(skill / "scripts"))
    data = _load_json_arg(args.data, "data")
    if not isinstance(data, dict) or "lesson" not in data:
        _fail("解析几何数据必须是包含 lesson/steps/board 的对象（schema 见技能 references/problem-schema.md）")
    try:
        import generate  # noqa: E402
    except Exception as e:
        _fail(f"导入技能失败（缺 sympy？）：{e}", 2)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    generate.render_html(data, out)
    print(f"OK analytic {out}")


# ---------------------------------------------------------------- 化学
def cmd_chem(args):
    if SKILLS is None:
        _fail("找不到 edu-chem-reaction 技能目录", 2)
    skill = SKILLS / "edu-chem-reaction"
    sys.path.insert(0, str(skill / "lib"))
    sys.path.insert(0, str(skill / "scripts"))
    spec = _load_json_arg(args.spec, "spec")
    if not isinstance(spec, dict) or "reactants" not in spec or "products" not in spec:
        _fail("化学 spec 必须包含 reactants / products（高层：species + atom_map）")
    try:
        import generate  # noqa: E402
        import reaction_kernel as K  # noqa: E402
        import molecules as M  # noqa: E402
    except Exception as e:
        _fail(f"导入技能失败（缺 sympy？）：{e}", 2)

    # 物种必须在 VSEPR 库内（否则无法构建 3D 模型）
    missing = []
    for side in ("reactants", "products"):
        for item in spec.get(side, []):
            sid = item.get("species") if isinstance(item, dict) else item
            if not M.has_species(str(sid)):
                missing.append(str(sid))
    if missing:
        _fail(f"物种不在 3D 模型库中：{', '.join(sorted(set(missing)))}。"
              f"可用物种：{', '.join(M.known_species())}")

    try:
        data = K.assemble_data(spec)
    except SystemExit as e:
        _fail(f"反应校验失败：{e}")
    except Exception as e:
        _fail(f"反应装配失败：{e}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    generate.render_html(data, out)
    print(f"OK chem {out}")


KINDS = {
    "solid-pyramid": cmd_solid, "solid-cube": cmd_solid, "solid-box": cmd_solid,
    "solid-random": cmd_solid,
    "analytic": cmd_analytic,
    "chem": cmd_chem,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=sorted(KINDS))
    ap.add_argument("--params", default=None)
    ap.add_argument("--data", default=None)
    ap.add_argument("--spec", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    try:
        KINDS[args.kind](args)
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        _fail("未预期的错误（详见上方堆栈）")


if __name__ == "__main__":
    main()
