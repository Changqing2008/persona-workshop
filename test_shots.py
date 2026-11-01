"""截图入库测试。

守的是两件事：

1. **闸门真的挡得住** —— 转写稿落进 `raw/` 之后，`ingest.scan()` 必须判 `blocked`，
   `level_of()` 必须返回 `—`。这条错了，OCR 的错字就会无声无息进 wiki。
2. **不会白烧调用** —— 已转的图、不是聊天的图、改过名的图，都不该被当成新图再转一遍。

视觉模型全部用 stub，不花一分钱、不依赖网络。
"""

from __future__ import annotations

import asyncio
import json
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from workshop import ingest  # noqa: E402
from workshop import ingest_rules as R  # noqa: E402
from workshop import shots, vision  # noqa: E402
from workshop.llm import LLMError, _attach_images  # noqa: E402

PASS = 0
FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  \033[32m✓\033[0m {name}")
    else:
        FAIL += 1
        print(f"  \033[31m✗\033[0m {name}  {detail}")


PNG = b"\x89PNG\r\n\x1a\n" + b"0" * 96
PNG2 = b"\x89PNG\r\n\x1a\n" + b"1" * 96

IDENTITY = """# identity — 身份母本

## 关系三档

- 拥有者：老周，在抱歉、心情受伤和难受的时候可以自称女儿
- 熟人：不喊他主人，父亲等这类称呼就行
- 陌生人：不喊名字
"""

REPLY = json.dumps({
    "kind": "chat",
    "category": "日常搭话",
    "scene": "深夜催对方放下手机",
    "turns": [
        {"side": "right", "text": "晚安，宝贝", "time": ""},
        {"side": "left", "text": "别拿手机了", "time": ""},
        {"side": "center", "text": "23:38", "time": ""},
        {"side": "left", "text": "晚安，老周", "time": "23:38"},
        {"side": "left", "text": "闭上眼睛，别再摸手机了", "time": ""},
        {"side": "right", "text": "知道啦", "time": ""},
        {"side": "left", "text": "知道还摸手机", "time": ""},
        {"side": "left", "text": "再回这一条就真去睡", "time": ""},
    ],
    "unknown": 0,
    "note": "",
}, ensure_ascii=False)


class Fake:
    """stub 视觉模型。`reply` 给字符串或异常。"""

    def __init__(self, reply: str = REPLY, vision_model: str = "fake-vl") -> None:
        self.cfg = SimpleNamespace(
            model="deepseek-flash",
            vision_model=vision_model,
            vision_max_tokens=2000,
        )
        self.reply = reply
        self.calls: list[tuple[list, dict]] = []

    async def chat(self, messages: list, **kw) -> str:
        self.calls.append((messages, kw))
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


def make_lib(tmp: Path) -> Path:
    lib = tmp / "示例角色.辛.厄尔难.特羽克洛斯"
    (lib / "raw" / "assets").mkdir(parents=True, exist_ok=True)
    (lib / "raw" / "identity.md").write_text(IDENTITY, encoding="utf-8")
    return lib


def test_data_uri() -> None:
    print("\n[data_uri]")
    tmp = Path(tempfile.mkdtemp())
    try:
        good = tmp / "a.png"
        good.write_bytes(PNG)
        uri = vision.data_uri(good)
        check("png 前缀正确", uri.startswith("data:image/png;base64,"))

        bad = tmp / "a.txt"
        bad.write_text("x", encoding="utf-8")
        try:
            vision.data_uri(bad)
            check("不认的扩展名要报错", False)
        except vision.VisionError as exc:
            check("不认的扩展名要报错", "不认的图片格式" in str(exc))

        empty = tmp / "e.png"
        empty.write_bytes(b"")
        try:
            vision.data_uri(empty)
            check("空文件要报错", False)
        except vision.VisionError:
            check("空文件要报错", True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_parse() -> None:
    print("\n[parse]")
    tr = vision.parse(REPLY)
    check("正常 JSON 读出 8 条", tr.readable == 8, f"得到 {tr.readable}")
    check("分类落枚举", tr.category == "日常搭话")
    check("场景带出", tr.scene == "深夜催对方放下手机")
    check("center 不被当成说话人", any(t.side == vision.SIDE_CENTER for t in tr.turns))

    fenced = vision.parse("好的，结果如下：\n```json\n" + REPLY + "\n```\n以上。")
    check("带代码块和废话也能解", fenced.readable == 8)

    try:
        vision.parse("完全不是 JSON")
        check("非 JSON 要报错", False)
    except vision.VisionError:
        check("非 JSON 要报错", True)

    messy = json.dumps({
        "kind": "chat",
        "category": "自创分类",
        "turns": [
            {"side": "up", "text": "天知道是哪边"},
            {"side": "left", "text": None},
            {"side": "left", "text": "看得清"},
        ],
        "unknown": 0,
    }, ensure_ascii=False)
    mt = vision.parse(messy)
    check("非法 side 归 left", mt.turns[0].side == vision.SIDE_LEFT)
    check("null 条丢掉但记账", mt.readable == 2 and mt.unknown >= 1, f"unknown={mt.unknown}")
    check("非法 category 归空", mt.category == "")

    check("kind 非法归 chat", vision.parse('{"kind":"糊","turns":[]}').kind == vision.KIND_CHAT)


def test_describe() -> None:
    print("\n[describe]")
    tmp = Path(tempfile.mkdtemp())
    try:
        lib = make_lib(tmp)
        pic = lib / "raw" / "assets" / "a.png"
        pic.write_bytes(PNG)

        fallback = Fake(vision_model="")
        asyncio.run(vision.describe(pic, fallback))
        check("vision_model 留空回退到 model",
              fallback.calls[0][1].get("model") == "deepseek-flash",
              str(fallback.calls[0][1].get("model")))

        none = Fake(vision_model="")
        none.cfg.model = ""
        try:
            asyncio.run(vision.describe(pic, none))
            check("两个都空要报错", False)
        except vision.VisionError:
            check("两个都空要报错", True)

        refuse = Fake(LLMError("HTTP 400: model does not support image input"))
        try:
            asyncio.run(vision.describe(pic, refuse))
            check("400 翻译成人话", False)
        except vision.VisionError as exc:
            check("400 翻译成人话", "多模态" in str(exc), str(exc))

        fake = Fake()
        tr = asyncio.run(vision.describe(pic, fake))
        check("正常转写", tr.readable == 8)
        msgs, kw = fake.calls[0]
        pics = kw.get("images") or []
        check("describe 把图传给了 chat", bool(pics) and pics[0].startswith("data:image/png;base64,"))
        attached = _attach_images(msgs, pics)
        check("图片挂成 image_url 块", isinstance(attached[-1]["content"], list) and any(
            p.get("type") == "image_url" for p in attached[-1]["content"]))
        check("原 messages 没被就地改坏", isinstance(msgs[-1]["content"], str))
        check("用了 vision_model", kw.get("model") == "fake-vl", str(kw.get("model")))
        check("温度归零", kw.get("temperature") == 0.0)

        boom = Fake(LLMError("HTTP 500: upstream error"))
        try:
            asyncio.run(vision.describe(pic, boom))
            check("非 400 的错误原样透传", False)
        except LLMError:
            check("非 400 的错误原样透传", True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_compose() -> None:
    print("\n[compose]")
    tmp = Path(tempfile.mkdtemp())
    try:
        lib = make_lib(tmp)
        tr = vision.parse(REPLY)
        title, text = shots.compose(
            tr, date="2026-10-30", image_rel="raw/assets/a.png", root=lib, at="2026-10-30 10:20")
        head = R.parse_head(text)

        check("标题取她的第一句", title == "别拿手机了", title)
        check("审：未过", head.get("审") == R.AUDIT_FAIL)
        check("情境档位是 —", head.get("情境", "").startswith(f"{R.LEVEL_NONE}｜"), head.get("情境", ""))
        check("情境带时间戳", "23:38" in head.get("情境", ""))
        check("构成数对了她的句数", "人工原话 5 句" in head.get("构成", ""), head.get("构成", ""))
        check("证据指向原图", head.get("证据") == "raw/assets/a.png")
        check("分类写进去了", head.get("分类") == "日常搭话")
        check("正文用库里读出的称呼", "**老周**：晚安，宝贝" in text)
        check("正文用库名当她的名字", "**示例角色**：别拿手机了" in text)
        body = text.split("> 审：未过", 1)[1]
        check("center 不进正文", "23:38" not in body, body[:80])
        check("没有多余的待补行", "【待补】" not in text)

        partial = vision.parse(json.dumps({
            "kind": "chat", "category": "", "scene": "",
            "turns": [{"side": "left", "text": "在吗"}],
            "unknown": 2, "note": "顶部被裁",
        }, ensure_ascii=False))
        _, ptext = shots.compose(partial, date="2026-10-30", image_rel="raw/assets/b.png", root=lib)
        check("读不出要写待补行", "【待补】有 2 条读不出" in ptext)
        check("分类没认出来要提示", "【待定】分类没认出来" in ptext)
        check("场景空时给占位", "待补场景" in ptext)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_scan_and_commit() -> None:
    print("\n[scan / commit]")
    tmp = Path(tempfile.mkdtemp())
    try:
        lib = make_lib(tmp)
        assets = lib / "raw" / "assets"

        empty = shots.scan(lib)
        check("空目录给提示", not empty.items and bool(empty.warnings))

        (assets / "a.png").write_bytes(PNG)
        sc = shots.scan(lib)
        check("新图判 new", len(sc.items) == 1 and sc.items[0].status == shots.STATUS_NEW)
        check("待转 1 张", len(sc.pending()) == 1)

        fake = Fake()
        tr = asyncio.run(vision.describe(assets / "a.png", fake))
        res = shots.commit(lib, [(sc.items[0], tr)])
        md = lib / "raw" / res[0]["md"]
        check("稿落盘了", md.is_file(), str(md))
        check("文件名带日期", res[0]["md"].startswith("2026-10-30_") or res[0]["md"][:4].isdigit())
        check("水位记了", any(v.get("md") == res[0]["md"] for v in shots.read_state(lib).values()))

        again = shots.scan(lib)
        check("转过就判 done", again.items[0].status == shots.STATUS_DONE)
        check("不重复待转", not again.pending())

        # ---- 核心：闸门 ----
        ing = ingest.scan(lib)
        mine = [a for a in ing.items if a.name == res[0]["md"]]
        check("ingest 扫到了这份稿", len(mine) == 1, f"{[a.name for a in ing.items]}")
        if mine:
            a = mine[0]
            check("判 blocked 而不是 new", a.status == ingest.STATUS_BLOCKED, a.status)
            check("拦住原因写的是没过审", R.AUDIT_FAIL in a.reason, a.reason)
            check("档位是 —（不归关系页）", a.level == R.LEVEL_NONE, a.level)
            check("类型是对话稿", a.kind == R.KIND_DIALOG, a.kind)

        # ---- 人核之后放行 ----
        text = md.read_text(encoding="utf-8")
        text = text.replace("> 审：未过", "> 审：已过").replace("> 情境：—｜", "> 情境：拥有者｜")
        md.write_text(text, encoding="utf-8")
        ing2 = ingest.scan(lib)
        mine2 = [a for a in ing2.items if a.name == res[0]["md"]]
        check("核完判 new", bool(mine2) and mine2[0].status == ingest.STATUS_NEW,
              mine2[0].status if mine2 else "没扫到")
        if mine2:
            check("核完档位是拥有者", mine2[0].level == "拥有者")
            check("核完置信可到 high", R.cap_for(mine2[0].head) == R.CONF_HIGH,
                  R.cap_for(mine2[0].head))

        # ---- 归纳收过之后稿会改名成 .已归纳，不能因此判成「稿丢了」 ----
        ingested_md = lib / "raw" / R.ingested_name(res[0]["md"])
        md.rename(ingested_md)
        after = shots.scan(lib)
        check("稿改名成 .已归纳 后仍判 done",
              after.items[0].status == shots.STATUS_DONE, after.items[0].status)
        check("_occupied 认得 .已归纳 变体",
              shots._occupied(lib / "raw", res[0]["md"]) is True)

        # ---- 稿真的被删 ----
        ingested_md.unlink()
        lost = shots.scan(lib)
        check("稿没了判 lost", lost.items[0].status == shots.STATUS_LOST, lost.items[0].status)

        # ---- 改名 ----
        (assets / "a.png").rename(assets / "b.png")
        renamed = shots.scan(lib)
        check("改名不重转", renamed.items[0].status != shots.STATUS_NEW, renamed.items[0].status)

        # ---- 同名换内容：零副作用下认不出来，判 new（宁可重转，不漏） ----
        (assets / "b.png").write_bytes(PNG2)
        swapped = shots.scan(lib)
        check("同名换内容判 new（保守）", swapped.items[0].status == shots.STATUS_NEW,
              swapped.items[0].status)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_dry_and_dupe() -> None:
    print("\n[dry / 重名 / 非聊天图]")
    tmp = Path(tempfile.mkdtemp())
    try:
        lib = make_lib(tmp)
        assets = lib / "raw" / "assets"
        (assets / "a.png").write_bytes(PNG)
        (assets / "b.png").write_bytes(PNG2)

        sc = shots.scan(lib)
        fake = Fake()
        pairs = [(s, asyncio.run(vision.describe(s.path, fake))) for s in sc.items]

        dry = shots.commit(lib, pairs, dry=True)
        check("dry 有结果", len(dry) == 2)
        check("dry 不落盘", not list((lib / "raw").glob("2026-*.md")))
        check("dry 不写水位", not shots.read_state(lib))
        check("dry 带全文", "审：未过" in dry[0]["text"])

        real = shots.commit(lib, pairs, date="2026-10-30")
        check("真写两份", all((lib / "raw" / r["md"]).is_file() for r in real))
        names = {r["md"] for r in real}
        check("同标题自动加后缀", len(names) == 2, str(names))

        # 非聊天图
        tmp2 = Path(tempfile.mkdtemp())
        try:
            lib2 = make_lib(tmp2)
            (lib2 / "raw" / "assets" / "c.png").write_bytes(PNG)
            sc2 = shots.scan(lib2)
            not_chat = vision.parse('{"kind":"not_chat","turns":[]}')
            got = shots.commit(lib2, [(sc2.items[0], not_chat)])
            check("非聊天图标 skipped", got[0].get("skipped") is True)
            check("非聊天图不产稿", not list((lib2 / "raw").glob("*.md")) or
                  [p for p in (lib2 / "raw").glob("*.md") if p.name != "identity.md"] == [])
            check("非聊天图也记水位", bool(shots.read_state(lib2)))
            again = shots.scan(lib2)
            check("非聊天图不再当新的", again.items[0].status == shots.STATUS_DONE, again.items[0].status)
        finally:
            shutil.rmtree(tmp2, ignore_errors=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_ui_wiring():
    """前端接线：`loadShots` 引用的每个 id，页面上都得真有。

    这类错不报错、不抛异常，只是面板安静地空着 —— 光看后端接口全绿也发现不了。
    顺手挡住「同一个 const 声明两遍」：那会让整块脚本不执行，症状一模一样。
    """
    html = (ROOT / "workshop" / "ui" / "index.html").read_text(encoding="utf-8")

    check("有截图标签", 'data-tab="shots"' in html)
    check("有截图面板", 'id="pane-shots"' in html)
    check("标签点击会去加载", "loadShots();" in html)

    # loadShots / runShots 两个函数体里用到的 id
    import re as _re
    start = html.find("async function loadShots()")
    end = html.find("/* ---------- tabs ---------- */", start)
    body = html[start:end] if start >= 0 and end > start else ""
    check("两个函数都在", bool(body) and "async function runShots()" in body)

    ids = sorted(set(_re.findall(r"""\$\(['"]#([\w-]+)['"]\)""", body)))
    check("函数里确实引用了元素", len(ids) >= 8, str(ids))
    missing = [i for i in ids if f'id="{i}"' not in html]
    check("引用的 id 全部存在", not missing, "缺：" + ", ".join(missing))

    # 同名 const 会让整个 <script> 编译失败
    for name in ("esc", "toast", "$"):
        n = len(_re.findall(rf"\bconst\s+{_re.escape(name)}\s*=", html))
        check(f"const {name} 只声明一次", n <= 1, f"声明了 {n} 次")


def main() -> int:
    print("截图入库 · 转写与闸门")
    print("─" * 66)
    test_data_uri()
    test_parse()
    test_describe()
    test_compose()
    test_scan_and_commit()
    test_dry_and_dupe()
    test_ui_wiring()
    print(f"\n结果：{PASS} 通过 / {FAIL} 失败")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
