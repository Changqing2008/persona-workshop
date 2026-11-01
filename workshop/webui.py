"""内置 WebUI。

零依赖：标准库 ThreadingHTTPServer + 一张单页 HTML。
只绑 127.0.0.1，且校验来源地址，避免把配置面板暴露到局域网。

它解决的问题很具体：
1. 改参数不用手改 TOML 再重启
2. 不用翻控制台就知道"为什么这条没回"
3. 调人格时能直接看到本轮注入了什么提示词
"""

from __future__ import annotations

import asyncio
import base64
import copy
import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from . import ingest, library, log, shots, tomlwrite, vision
from .agent import Agent
from .config import Config
from .distill import Distiller
from .llm import LLMClient, LLMError
from .memory import Memory
from .persona import Persona
from .seed import TEMPLATE, Seed
from .sim import Studio, dims
from .sim.cards import Card, generate_cards

log_ = log.get("webui")

UI_FILE = Path(__file__).parent / "ui" / "index.html"

_SPEAKER = re.compile(r"^([^\s:：]{1,32})[:：]\s*")


def _speaker_name(content: str) -> str:
    """从落库内容里取出说话人。

    群消息落库时带 "名字: " 前缀（agent._annotate），读回来就能按人分组。
    老消息没有 user_id 列，靠这个前缀一样分得开。
    """
    m = _SPEAKER.match(str(content or ""))
    return m.group(1).strip() if m else ""

# 表单描述：前端照着这个渲染，后端照着这个校验
# 工坊不接通道：adapter.* / QQ 凭据全部不出现，通道配置归通道器程序管。
FIELDS: list[dict[str, Any]] = [
    {"path": "llm.base_url", "label": "接口地址", "type": "text", "group": "模型",
     "hint": "任何 OpenAI 兼容端点，填到 /v1 为止"},
    {"path": "llm.model", "label": "模型名", "type": "text", "group": "模型"},
    {"path": "llm.api_key", "label": "API Key", "type": "password", "group": "模型",
     "hint": "留空则读环境变量（见下一项）"},
    {"path": "llm.api_key_env", "label": "Key 环境变量名", "type": "text", "group": "模型"},
    {"path": "llm.temperature", "label": "温度", "type": "float", "group": "模型", "step": 0.05,
     "hint": "0.7~0.95 之间比较像人说话；太高会开始胡说"},
    {"path": "llm.max_tokens", "label": "最大回复长度", "type": "int", "group": "模型"},

    {"path": "bot.persona_dir", "label": "启动时加载的人格库", "type": "text", "group": "人格",
     "readonly": True, "tag": "跟随切换",
     "hint": "切到哪个库，开机就读哪个库；这里只读。换库用总端，或点上面那个库的「切到此库运行」"},
    {"path": "bot.nickname", "label": "机器昵称", "type": "text", "group": "人格",
     "hint": "协议端群里喊这个名字也触发；官方通道靠 @，此项无效"},

    {"path": "owner.ids", "label": "拥有者 id", "type": "list", "group": "拥有者",
     "hint": "官方通道填 openid：私聊一个、群聊一个（同一人在两处是两个 id）。协议端填 QQ 号。逗号或换行分隔"},
    {"path": "owner.call_me", "label": "她怎么叫你", "type": "text", "group": "拥有者",
     "hint": "留空则不指定称呼"},
    {"path": "owner.familiar_turns", "label": "熟人门槛（轮）", "type": "int", "group": "拥有者",
     "hint": "非主人聊够这么多轮算熟人。0 = 只有主客两档"},
    {"path": "owner.carry_summary", "label": "群里带私聊摘要", "type": "bool", "group": "拥有者",
     "hint": "只在主人说话时注入，且禁止复述"},




    # 漂移：上面四个旋钮的"活"版本。开关和参数都在这里，状态本身存在
    # 各库的 data/drift.json 里，配置页只管"允不允许漂、漂多快"。



    {"path": "memory.recent_turns", "label": "注入历史条数", "type": "int", "group": "记忆"},
    {"path": "memory.summarize_after", "label": "摘要触发条数", "type": "int", "group": "记忆"},
    {"path": "memory.wiki_enabled", "label": "记忆层（persona/memory）", "type": "bool", "group": "记忆",
     "hint": "开 = 命中旧事时去记忆库检索；关 = 退回旧行为，零残留。改完即时生效"},
    {"path": "memory.wiki_budget", "label": "记忆注入字数上限", "type": "int", "group": "记忆",
     "hint": "记忆段不许挤占人格的预算，超了就截断"},
    {"path": "memory.ingest_interval_hours", "label": "自动归档间隔（小时）", "type": "float", "step": 0.5,
     "group": "记忆",
     "hint": "0 = 不自动跑。>0 时每 N 小时把新聊天沉淀进记忆库。会花模型额度，改完一分钟内生效"},
    {"path": "memory.ingest_allow_new", "label": "自动归档允许新建页", "type": "bool", "group": "记忆",
     "hint": "关着时只往已有页面里补事实，不擅自开新页——没人看着的时候稳妥些"},

    # 一键归纳的三个人格库内开关。前端不传参数，所以这里设的是什么，点按钮就是什么。
    {"path": "ingest.max_per_run", "label": "一轮最多收几份", "type": "int", "group": "归纳",
     "hint": "0 = 不限。素材多时限一下，免得一轮跑十几分钟"},
    {"path": "ingest.auto_index", "label": "收完重建总括段", "type": "bool", "group": "归纳",
     "hint": "开 = 每轮收完顺手重排 index.md 的四小节（纯确定性抽取，不花额度）；关 = 只写 wiki"},
    {"path": "ingest.preview_first", "label": "先空跑一遍", "type": "bool", "group": "归纳",
     "hint": "开 = 每轮只算不落盘，用来看会改哪些地方。平时关着"},

    {"path": "sim.dir", "label": "素材台目录", "type": "text", "group": "素材台",
     "hint": "下分 cards/（收藏的卡）与 drafts/（草稿与成品）。改完需重启"},
    {"path": "sim.candidates", "label": "候选条数", "type": "int", "group": "素材台",
     "hint": "每轮出几条供挑选，3~4 条比较好挑"},
    {"path": "sim.card_count", "label": "一次出几张卡", "type": "int", "group": "素材台"},
    {"path": "sim.card_max_tokens", "label": "卡片输出上限", "type": "int", "group": "素材台",
     "hint": "三张卡约 1200~1500 token。报「输出被截断」时往上调，改完即时生效"},
    {"path": "sim.temperature", "label": "候选温度", "type": "float", "step": 0.05,
     "group": "素材台", "hint": "越高候选差异越大；太低会出三条一样的"},
    {"path": "sim.self_label", "label": "角色显示名", "type": "text", "group": "素材台",
     "hint": "收稿时写进对话的称呼。留空取人格库目录名的第一段"},

    {"path": "distill.dir", "label": "蒸馏目录", "type": "text", "group": "蒸馏",
     "hint": "下分 books/（原文与提取结果）与 out/（成品 md）。改完需重启"},
    {"path": "distill.min_lines", "label": "成段最少句数", "type": "int", "group": "蒸馏",
     "hint": "少于这么多句不成段。一段至少要有来有回才值得当素材"},
    {"path": "distill.max_lines", "label": "一段最多句数", "type": "int", "group": "蒸馏",
     "hint": "默认 3（一个回合：对方一句、她一句、对方再说一句）。满了立刻断，"
             "段内必须双方都在，只有一方说话的段不成段"},
    {"path": "distill.gap_paragraphs", "label": "断段间隔（段）", "type": "int", "group": "蒸馏",
     "hint": "两句之间隔超过这么多段就断开，不硬连成一段"},
    {"path": "distill.judge", "label": "台词归属裁判", "type": "select",
     "options": ["rules", "auto", "off"], "group": "蒸馏",
     "hint": "rules 只按规则、不调模型；auto 让模型按场景块判每句谁说的，"
             "准得多，但每块一次调用、大书要等一两分钟，也会花 token"},
    {"path": "distill.guess", "label": "轮次推测", "type": "select",
     "options": ["chain", "anchor", "off"], "group": "蒸馏",
     "hint": "chain 按 A→B→A→B 一路推，召回主力（本书 1465 句里留 1321 句，"
             "金标准判对 56%）；anchor 只在上一句真有名字时才推一句，只留 94 句，"
             "适合只想看有把握的那几句；off 完全不推。推出来的句子在清单里标「推定」"},
]

# 配置页上的每一项都随库走：一个库就是一套完整的机器人——通道凭证、模型、
# 触发、限速、合流都写进库内 preferences.toml，换库等于换人 + 换账号。
# 唯一的例外是 bot.persona_dir：它是"程序启动时加载哪个库"的自指项，
# 让它跟库走等于"用 A 库的偏好把自己指成 B"，永远留在全局 config.toml。
# 它也不接受配置页提交（readonly）——只由「切库」写回，做到切到哪读哪。
PERSONA_SCOPED: set[str] = {f["path"] for f in FIELDS} - set(library.GLOBAL_PATHS)

# 所有表单路径。新建库时用来导出整份配置打底。
ALL_PATHS: tuple[str, ...] = tuple(f["path"] for f in FIELDS)

for _f in FIELDS:
    _f["scope"] = "persona" if _f["path"] in PERSONA_SCOPED else "global"
del _f


class _Handler(BaseHTTPRequestHandler):
    server_version = "persona-workshop-webui"
    ctx: dict[str, Any] = {}

    # ---------- 基础工具 ----------

    def log_message(self, fmt: str, *args: Any) -> None:  # 静音访问日志
        return

    def _is_local(self) -> bool:
        host = self.client_address[0]
        return host in {"127.0.0.1", "::1"}

    def _json(self, obj: Any, code: int = 200) -> None:
        data = json.dumps(obj, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _body(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        if not length:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return {}

    def _run(self, coro, timeout: float = 180.0):
        """把协程丢回主事件循环执行，并等结果。"""
        loop: asyncio.AbstractEventLoop | None = self.ctx.get("loop")
        if loop is None or loop.is_closed():
            raise RuntimeError("主循环不可用")
        return asyncio.run_coroutine_threadsafe(coro, loop).result(timeout=timeout)

    def _persona_id(self) -> str:
        """本次请求看哪个人格库：X-Persona 头 > ?persona= > 启动库。"""
        pid = (self.headers.get("X-Persona") or "").strip()
        if not pid:
            pid = (self._query_param("persona") or "").strip()
        return pid or str(self.ctx.get("persona_id") or "")

    def _scope(self) -> dict[str, Any]:
        """按人格库取作用域。每个请求带自己的库，多标签页看不同库不会互相串。

        启动库直接用 run.py 装配好的那套（通道、智能体都连着它）；
        其它库现开一套可查看可编辑的上下文，**不碰线上智能体**——
        改别的库的 wiki 不该把正在跟人聊天的那个换掉。
        """
        pid = self._persona_id()
        cache: dict[str, Any] = self.ctx.setdefault("scope_cache", {})
        hit = cache.get(pid)
        if hit is not None:
            return hit

        cfg: Config = self.ctx["cfg"]
        boot_id = str(self.ctx.get("persona_id") or "")
        lib = self.ctx.get("library")
        entry = lib.get(pid) if (lib is not None and pid and pid != boot_id) else None

        # 没指定库 / 指定的就是启动库 / 根本没有注册表（老调用方式、离线测试）
        # 一律用启动时装配好的那套，行为与多库改造前完全一致。
        # 不缓存：启动库的 seed / sim / distill 可能被中途替换（测试注入、配置热更新），
        # 缓存一份就会永远读到旧对象。
        if entry is None and (not pid or pid == boot_id or lib is None):
            return {
                "id": pid or boot_id,
                "persona": self.ctx["persona"],
                "memory": self.ctx["memory"],
                "seed": self.ctx.get("seed"),
                "sim": self.ctx.get("sim"),
                "distill": self.ctx.get("distill"),
                "boot": True,
            }

        if entry is None:
            raise ValueError(f"人格库未登记：{pid}")
        if not entry.path.is_dir():
            raise ValueError(f"人格库目录不存在：{entry.path}")

        persona = Persona(entry.path)
        llm = getattr(self.ctx.get("agent"), "llm", None)
        # 这个库的完整配置：全局打底 + 库内 prefs 覆盖。
        # 素材台 / 蒸馏 / 试聊都要用这一份，否则在当前库造料却带着别人的口吻。
        scoped = copy.deepcopy(cfg)
        library.apply_prefs(scoped, entry.path)
        sim = Studio(scoped, persona, llm) if scoped.sim.enabled and llm is not None else None
        distill = (
            Distiller(scoped, sim, llm)
            if scoped.distill.enabled and sim is not None and llm is not None
            else None
        )
        scope = {
            "id": pid,
            "cfg": scoped,
            "persona": persona,
            "memory": Memory(library.sessions_path(entry.path)),
            "seed": Seed(
                entry.path,
                library.resolve_history(entry.path, cfg.path("data") / "identity_history"),
            ),
            "sim": sim,
            "distill": distill,
            "boot": False,
        }
        cache[pid] = scope
        return scope

    def _scope_agent(self):
        """当前库的智能体。

        启动库直接用 run.py 装配好的那个（连着 QQ）；其它库现造一个只做试聊的，
        因为 preview 不经过通道，不需要 sender 真把消息发出去。
        """
        scope = self._scope()
        if scope.get("boot"):
            return self.ctx.get("agent")
        agent = scope.get("agent")
        if agent is not None:
            return agent
        llm = getattr(self.ctx.get("agent"), "llm", None)
        if llm is None:
            return None
        # 用该库自己的那份配置，不要再去 deepcopy 全局的运行配置
        cfg = scope.get("cfg") or self.ctx["cfg"]

        async def _sink(out) -> None:  # 试聊不发消息
            return None

        agent = Agent(cfg, scope["persona"], scope["memory"], llm, _sink)
        scope["agent"] = agent
        return agent

    def _mem(self):
        """当前请求所属库的记忆库。"""
        return self._scope()["memory"]

    # ---------- 路由 ----------

    def do_GET(self) -> None:  # noqa: N802
        if not self._is_local():
            self._json({"error": "forbidden"}, 403)
            return
        path = self.path.split("?")[0]
        try:
            if path in {"/", "/index.html"}:
                self._serve_ui()
            elif path == "/api/personas":
                self._json(self._personas())
            elif path == "/api/status":
                self._json(self._status())
            elif path == "/api/config":
                self._json(self._config())
            elif path == "/api/logs":
                self._logs()
            elif path == "/api/persona":
                self._json(self._persona())
            elif path == "/api/identity":
                self._json(self._identity())
            elif path == "/api/ingest/state":
                self._json(self._ingest_state())
            elif path == "/api/ingest/job":
                self._json({"ok": True, **ingest.job(), "memory": self._mem_ingest_state()})
            elif path == "/api/ingest/history":
                self._json({"ok": True, "history": ingest.history(self._scope()["persona"].root)})
            elif path == "/api/ingest/lint":
                self._json(self._ingest_lint())
            elif path == "/api/shots/state":
                self._json(self._shots_state())
            elif path == "/api/sessions":
                mem = self._mem()
                self._json({"sessions": mem.list_sessions()})
            elif path == "/api/sessions/view":
                self._json(self._session_view())
            elif path == "/api/sim/drafts":
                self._json(self._sim_index())
            elif path == "/api/sim/draft":
                self._json(self._sim_get())
            elif path == "/api/distill/books":
                self._json({"ok": True, "books": self._distill().books(),
                            "root": str(self._distill().root)})
            elif path == "/api/distill/book":
                self._json(self._distill_get())
            else:
                self._json({"error": "not found"}, 404)
        except Exception as exc:  # noqa: BLE001
            log_.exception("GET %s 失败", path)
            self._json({"error": str(exc)}, 500)

    def do_POST(self) -> None:  # noqa: N802
        if not self._is_local():
            self._json({"error": "forbidden"}, 403)
            return
        path = self.path.split("?")[0]
        body = self._body()
        try:
            if path == "/api/personas":
                self._json(self._create_persona(body))
            elif path == "/api/personas/archive":
                self._json(self._archive_persona(body))
            elif path == "/api/personas/import":
                self._json(self._import_persona(body))
            elif path == "/api/config":
                self._json(self._save_config(body))
            elif path == "/api/memory/ingest":
                self._json(self._memory_ingest())
            elif path == "/api/ingest/run":
                self._json(self._ingest_run(body))
            elif path == "/api/ingest/undo":
                self._json(self._ingest_undo(body))
            elif path == "/api/shots/one":
                self._json(self._shots_one(body))
            elif path == "/api/chat":
                self._json(self._chat(body))
            elif path == "/api/chat/clear":
                bot = self._scope_agent()
                if bot is not None:
                    bot.clear_preview()
                self._json({"ok": True})
            elif path == "/api/sessions/delete":
                self._json(self._session_delete(body))
            elif path == "/api/sessions/compact":
                self._json(self._session_compact(body))
            elif path == "/api/persona/reload":
                self._scope()["persona"].reload()
                self._json({"ok": True, "persona": self._persona()})
            elif path == "/api/identity":
                self._json(self._save_identity(body))
            elif path == "/api/channel/start":
                self._json(self._channel_op("start"))
            elif path == "/api/channel/stop":
                self._json(self._channel_op("stop"))
            elif path == "/api/channel/switch":
                self._json(self._channel_switch(body))
            elif path == "/api/sim/cards":
                self._json(self._sim_cards(body))
            elif path == "/api/sim/draft":
                self._json(self._sim_new(body))
            elif path == "/api/sim/cand":
                self._json(self._sim_cand(body))
            elif path == "/api/sim/commit":
                self._json(self._sim_commit(body))
            elif path == "/api/sim/undo":
                self._json(self._sim_undo(body))
            elif path == "/api/sim/edit":
                self._json(self._sim_edit(body))
            elif path == "/api/sim/turn":
                self._json(self._sim_turn(body))
            elif path == "/api/sim/draft-card":
                self._json(self._sim_draft_card(body))
            elif path == "/api/sim/mode":
                self._json(self._sim_mode(body))
            elif path == "/api/sim/export":
                self._json(self._sim_export(body))
            elif path == "/api/sim/reactgen":
                self._json(self._sim_reactgen(body))
            elif path == "/api/sim/reaction":
                self._json(self._sim_reaction(body))
            elif path == "/api/sim/reaction/update":
                self._json(self._sim_reaction_update(body))
            elif path == "/api/sim/react/example":
                self._json(self._sim_react_example(body))
            elif path == "/api/sim/save-card":
                self._json(self._sim_save_card(body))
            elif path == "/api/sim/draw":
                self._json(self._sim_draw(body))
            elif path == "/api/sim/delete-card":
                self._json(self._sim_delete_card(body))
            elif path == "/api/distill/import":
                self._json(self._distill_import(body))
            elif path == "/api/distill/extract":
                self._json(self._distill_extract(body))
            elif path == "/api/distill/segment":
                self._json(self._distill_segment(body))
            elif path == "/api/distill/line":
                self._json(self._distill_line(body))
            elif path == "/api/distill/export":
                self._json(self._distill_export(body))
            elif path == "/api/distill/to-sim":
                self._json(self._distill_to_sim(body))
            elif path == "/api/distill/to-library":
                self._json(self._distill_to_library(body))
            elif path == "/api/distill/react":
                self._json(self._distill_react(body))
            elif path == "/api/distill/react/update":
                self._json(self._distill_react_update(body))
            elif path == "/api/distill/react/export":
                self._json(self._distill_react_export(body))
            else:
                self._json({"error": "not found"}, 404)
        except ValueError as exc:
            # 参数问题（母本为空、超长等）属预期分支，不该刷一屏 traceback
            log_.warning("POST %s 被拒：%s", path, exc)
            self._json({"error": str(exc)}, 400)
        except Exception as exc:  # noqa: BLE001
            log_.exception("POST %s 失败", path)
            self._json({"error": str(exc)}, 500)

    # ---------- 各接口实现 ----------

    def _serve_ui(self) -> None:
        if not UI_FILE.is_file():
            self._json({"error": f"缺少界面文件：{UI_FILE}"}, 500)
            return
        data = UI_FILE.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    # ---------- 人格库总端 ----------

    def _personas(self) -> dict[str, Any]:
        """总端列表：每个库的填充度、会话概览、是否正在运行、是否归档。"""
        lib = self.ctx.get("library")
        if lib is None:
            raise RuntimeError("人格库注册表未初始化")
        lib.load()  # 每次重读，别处手工建的库也能立刻出现
        active = str(self.ctx.get("persona_id") or "")
        items: list[dict[str, Any]] = []
        for entry in lib.entries(include_archived=True):
            item = entry.to_dict()
            try:
                item.update(library.stats(entry.path))
            except OSError as exc:
                item["error"] = str(exc)
            item["active"] = entry.id == active
            item["max_pages"] = len(library.WIKI_PAGES)
            # 漂移状态一人一份，总端顺带列出来，方便确认切库真的换了人
            item["drift"] = library.drift_summary(entry.path)
            items.append(item)
        return {"ok": True, "active": active, "personas": items}

    def _create_persona(self, body: dict[str, Any]) -> dict[str, Any]:
        """建库：生成空骨架并登记。已存在的目录一律不碰。"""
        cfg: Config = self.ctx["cfg"]
        name = str(body.get("name") or "").strip()
        dirname = str(body.get("dir") or "").strip() or name
        summary = str(body.get("summary") or "").strip()
        parent = str(body.get("parent") or "").strip()

        err = library.validate_name(dirname)
        if err:
            return {"ok": False, "error": err}
        if len(summary) > 200:
            return {"ok": False, "error": "一句话人设太长了，200 字以内"}

        # 不指定父目录时，默认跟**启动库做邻居**——人格库是"一个人一个目录"，
        # 藏在程序目录里（程序目录下的 personas/）最难找。只有启动库本身
        # 就在项目目录里的时候，才退回项目内的 personas/。
        if parent:
            base = Path(parent)
        else:
            boot_dir = Path(str(cfg.bot.persona_dir or ""))
            base = boot_dir.parent if boot_dir.parent.is_dir() else (cfg.root / "personas")
        root = base / dirname
        lib = self.ctx.get("library")
        if lib is not None and lib.exists(dirname):
            return {"ok": False, "error": f"已有人格库叫「{dirname}」，换个目录名"}

        try:
            # 新库不该从空配置开始，否则点连接只会提示缺凭证。
            # 拿当前生效的整份配置打底，用户想分开再自己进库改。
            entry = library.create_persona(
                root,
                name=name or dirname,
                summary=summary,
                prefs=library.prefs_from_config(
                    cfg, name or dirname, summary, ALL_PATHS
                ),
            )
        except (ValueError, OSError) as exc:
            return {"ok": False, "error": str(exc)}

        if lib is not None:
            lib.register(entry.path, name=entry.name, summary=entry.summary)
        return {
            "ok": True,
            "persona": entry.to_dict(),
            "files": len(library.WIKI_PAGES) + 5,
            "hint": "骨架已生成：wiki 八页空着，身份母本还是八节占位。进主界面从「身份」填起。",
        }

    def _archive_persona(self, body: dict[str, Any]) -> dict[str, Any]:
        """归档 / 取消归档：只在注册表记一笔，磁盘上的文件一个都不动。"""
        pid = str(body.get("id") or "").strip()
        if not pid:
            return {"ok": False, "error": "缺 id"}
        want = bool(body.get("archived", True))
        if want and pid == str(self.ctx.get("persona_id") or ""):
            return {"ok": False, "error": "当前运行中的库不能归档，先重启到别的库"}
        lib = self.ctx.get("library")
        if lib is None:
            return {"ok": False, "error": "注册表未初始化"}
        try:
            entry = lib.archive(pid) if want else lib.unarchive(pid)
        except KeyError as exc:
            return {"ok": False, "error": str(exc)}
        return {
            "ok": True,
            "persona": entry.to_dict(),
            "hint": "已从总端隐藏，文件仍在原位置，随时可以取消归档。"
            if want
            else "已恢复到总端。",
        }

    def _import_persona(self, body: dict[str, Any]) -> dict[str, Any]:
        """把一个已经存在的人格库目录登记进总端。

        不复制、不移动、不改写任何已有文件。只补两样它可能缺的东西：
        空会话库 data/sessions.db 和 preferences.toml —— 缺了 db 它会按回退规则
        跟启动库共用同一个会话库，两个角色的对话就混在一起了。
        """
        raw_path = str(body.get("path") or "").strip().strip('"')
        if not raw_path:
            return {"ok": False, "error": "缺目录路径"}
        cfg: Config = self.ctx["cfg"]
        root = Path(raw_path).expanduser()
        if not root.is_absolute():
            root = cfg.root / root
        if not root.is_dir():
            return {"ok": False, "error": f"目录不存在：{root}"}
        if not ((root / "index.md").is_file() or (root / "wiki").is_dir()):
            return {"ok": False, "error": "这个目录不像人格库：既没有 index.md，也没有 wiki/ 目录"}
        lib = self.ctx.get("library")
        if lib is None:
            return {"ok": False, "error": "注册表未初始化"}
        if lib.exists(root.name):
            return {"ok": False, "error": f"注册表里已经有「{root.name}」了"}

        created_db = created_prefs = False
        db = library.sessions_path(root)
        if not db.is_file():
            library.ensure_db(root)  # 建库逻辑只留一处，和 create_persona 共用
            created_db = True
        if not library.prefs_path(root).is_file():
            library.write_prefs(
                root,
                library.prefs_from_config(cfg, display_name=root.name, paths=ALL_PATHS),
            )
            created_prefs = True

        entry = lib.register(root, name=str(body.get("name") or "").strip() or root.name)
        notes: list[str] = []
        if created_db:
            notes.append("已建空会话库 data/sessions.db")
        if created_prefs:
            notes.append("已生成 preferences.toml（通道凭证、模型、触发等整份配置取自当前运行的那套，可在配置页改）")
        return {
            "ok": True,
            "persona": entry.to_dict(),
            "hint": f"已登记「{entry.name}」"
            + ("：" + "；".join(notes) if notes else "，原有文件一个没动"),
        }

    def _channel_op(self, op: str) -> dict:
        """启停 QQ 通道。失败不抛异常，把原因回给界面。"""
        if not self._scope().get("boot"):
            return {
                "ok": False,
                "error": "通道连的是运行中的那个人格库。想换人，点顶部的「切到此库运行」。",
            }
        ch = self.ctx.get("channel")
        if not ch or op not in ch:
            return {"ok": False, "error": "当前模式不支持启停通道"}
        try:
            return self._run(ch[op](), timeout=90)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}

    def _channel_switch(self, body: dict[str, Any]) -> dict[str, Any]:
        """把线上运行的人格库换成另一个（停通道 → 换人 → 重连）。"""
        pid = str(body.get("id") or "").strip() or self._persona_id()
        if not pid:
            return {"ok": False, "error": "缺人格库 id"}
        ch = self.ctx.get("channel") or {}
        fn = ch.get("switch")
        if not callable(fn):
            return {"ok": False, "error": "当前模式不支持换库"}
        try:
            return self._run(fn(pid), timeout=180)
        except Exception as exc:  # noqa: BLE001
            log_.exception("换库失败：%s", pid)
            return {"ok": False, "error": str(exc)}

    def _status(self) -> dict[str, Any]:
        cfg: Config = self.ctx["cfg"]
        agent = self.ctx.get("agent")
        adapter = self.ctx.get("adapter")
        scope = self._scope()
        persona = scope["persona"]
        memory = self._mem()

        pages = persona.status()
        filled = sum(1 for p in pages.values() if p.filled)
        mem = memory.stats()

        ch_state: dict[str, Any] = {}
        ch = self.ctx.get("channel")
        if ch and "status" in ch:
            try:
                ch_state = self._run(ch["status"](), timeout=5) or {}
            except Exception:  # noqa: BLE001
                ch_state = {}

        ing_state: dict[str, Any] = {}
        ing = self.ctx.get("ingest")
        if ing and "status" in ing:
            try:
                ing_state = self._run(ing["status"](), timeout=5) or {}
            except Exception:  # noqa: BLE001
                ing_state = {}

        return {
            "model": cfg.llm.model,
            "base_url": cfg.llm.base_url,
            "llm_ready": bool(cfg.llm.resolve_key()),
            "connected": False,  # 工坊不接通道，试聊直连模型
            "running": bool(ch_state.get("running")),
            "adapter": "试聊",
            "persona": {
                "id": scope["id"],
                "name": scope["id"],
                "boot": bool(scope["boot"]),
                "root": str(persona.root),
                "filled": filled,
                "total": len(pages),
                "summary_empty": persona.self_summary_is_empty,
            },
            "runtime": {
                "id": str(ch_state.get("persona_id") or self.ctx.get("persona_id") or ""),
                "boot": scope["id"] == str(ch_state.get("persona_id") or self.ctx.get("persona_id") or ""),
                "switchable": callable((self.ctx.get("channel") or {}).get("switch")),
            },
            "memory": mem,
            "ingest": ing_state,
            "stats": dict(getattr(agent, "stats", {}) or {}),
            "started": self.ctx.get("started"),
            "version": self.ctx.get("version", ""),
        }

    def _memory_ingest(self) -> dict[str, Any]:
        """手动触发一轮记忆归档。触发后立刻返回，进度看 /api/status 的 ingest 段。

        一轮归档要挨个会话调模型，几分钟都可能，HTTP 请求等不起，
        更不能让浏览器以为点坏了。
        """
        ing = self.ctx.get("ingest")
        if not ing or "start" not in ing:
            return {"ok": False, "error": "归档器没接上（记忆层没开或归档器加载失败）"}
        running = str(self.ctx.get("persona_id") or "")
        viewing = self._persona_id()
        # 归档器读的是运行中那个库的会话库，给别的库点按钮会"点了个空"
        if viewing and running and viewing != running:
            return {
                "ok": False,
                "error": f"正在运行的是「{running}」，整理只作用于它。要整理「{viewing}」先切过去",
            }
        try:
            return self._run(ing["start"](), timeout=8) or {"ok": False, "error": "没有回应"}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}

    def _mem_ingest_state(self) -> dict[str, Any]:
        """记忆归档的状态。挂在同一个轮询口上，界面就不用开两条轮询。"""
        ing = self.ctx.get("ingest")
        if not ing or "status" not in ing:
            return {"available": False}
        try:
            return self._run(ing["status"](), timeout=5) or {"available": False}
        except Exception:  # noqa: BLE001
            return {"available": False, "error": "状态没取到"}

    def _ingest_lint(self) -> dict[str, Any]:
        """体检：确定性 lint，随时可点，不改任何东西。

        一键归纳跑完会自动扫一遍，这里留个手动口子 —— 手改过 wiki 之后
        想确认有没有写坏，不用为了看一眼就去跑一轮归纳。
        """
        from . import lint as lint_mod

        try:
            root = self._scope()["persona"].root
            return lint_mod.run(root)
        except Exception as exc:  # noqa: BLE001
            log_.exception("体检失败")
            return {"ok": False, "error": str(exc), "summary": "", "items": []}

    # ---------- 截图入库 ----------

    def _shots_model(self) -> str:
        """这一轮用哪个模型看图。留空 `vision_model` 就跟着聊天模型走。"""
        cfg: Config = self.ctx["cfg"]
        return (cfg.llm.vision_model or cfg.llm.model or "").strip()

    def _shots_state(self) -> dict[str, Any]:
        """扫一遍 raw/assets。零副作用，不建目录、不写水位。"""
        try:
            root = self._scope()["persona"].root
            sc = shots.scan(root)
        except Exception as exc:  # noqa: BLE001
            log_.exception("扫截图失败")
            return {"ok": False, "error": str(exc)}
        out = sc.to_dict()
        out["ok"] = True
        out["model"] = self._shots_model()
        return out

    def _shots_one(self, body: dict[str, Any]) -> dict[str, Any]:
        """转写一张并落稿。**一次一张**：前端循环调，进度能逐张刷出来。

        不批量处理是因为十张就要等二十秒，HTTP 先超时了；而且单张失败只废这一张，
        不会让整批白跑。落出来的稿一律 `审：未过` + `情境：—`，被归纳闸门挡住。
        """
        name = str(body.get("name") or "").strip()
        if not name:
            return {"ok": False, "error": "没给图片名"}

        root = self._scope()["persona"].root
        hit = next((s for s in shots.scan(root).items if s.name == name), None)
        if hit is None:
            return {"ok": False, "error": f"raw/assets 里没有 {name}"}
        if hit.status == shots.STATUS_DONE:
            return {"ok": False, "error": f"{name} 已经转过了（{hit.md or '非聊天图'}）"}

        model = self._shots_model()
        if not model:
            return {"ok": False, "error": "config.toml 的 [llm] 里 model 和 vision_model 都是空的"}

        cfg: Config = self.ctx["cfg"]
        try:
            tr = asyncio.run(vision.describe(hit.path, LLMClient(cfg.llm), model=model))
        except (vision.VisionError, LLMError) as exc:
            return {"ok": False, "error": str(exc)}

        try:
            got = shots.commit(root, [(hit, tr)])
        except Exception as exc:  # noqa: BLE001
            log_.exception("落截图稿失败")
            return {"ok": False, "error": str(exc)}

        row = got[0] if got else {}
        out: dict[str, Any] = {
            "ok": True,
            "name": name,
            "kind": tr.kind,
            "turns": tr.readable,
            "unknown": tr.unknown,
            "note": tr.note,
            "md": row.get("md", ""),
        }
        if tr.kind == vision.KIND_NOT_CHAT:
            out["skipped"] = True
            out["note"] = out["note"] or "不是聊天截图，只记水位、不产稿"
        return out

    def _ingest_state(self) -> dict[str, Any]:
        """一键归纳的队列状态：待收几份、拦住几份、每份为什么。

        每次请求真扫一遍 raw/（二十几个文件算 sha 是毫秒级），不做缓存 ——
        缓存只会让人刚丢进去一份素材、界面上还是旧数字，以为程序没反应。
        """
        try:
            root = self._scope()["persona"].root
            scan = ingest.scan(root)
        except Exception as exc:  # noqa: BLE001
            log_.exception("扫 raw 失败")
            return {"ok": False, "error": str(exc)}
        out = scan.to_dict()
        out["ok"] = True
        out["job"] = {**ingest.job(), "memory": self._mem_ingest_state()}
        return out

    def _ingest_run(self, body: dict[str, Any]) -> dict[str, Any]:
        """开跑一键归纳。立刻返回，进度靠 /api/ingest/job 轮询。

        一轮要挨份调模型，二十四份素材跑十几分钟都正常，HTTP 请求等不起。
        """
        scope = self._scope()
        root = scope["persona"].root
        llm = getattr(self.ctx.get("agent"), "llm", None)
        if llm is None or not getattr(llm, "ready", False):
            return {"ok": False, "error": "模型没配好（缺 API Key）。归纳要读规程 + 素材 + 现有页，必须有模型"}
        loop: asyncio.AbstractEventLoop | None = self.ctx.get("loop")
        if loop is None or loop.is_closed():
            return {"ok": False, "error": "主循环不可用"}

        cfg: Config = scope.get("cfg") or self.ctx["cfg"]
        then_memory = bool(body.get("then_memory"))

        def _kick_memory() -> str | None:
            """在 loop 上把记忆归档开起来。成功返回 None，失败返回原因。

            只用于「素材已经是最新、没什么可收」那条分支 —— 后端这次没有
            正在跑的协程可以顺手 await，所以得单独丢一个过去。
            """
            start = (self.ctx.get("ingest") or {}).get("start")
            if start is None:
                return "记忆归档没接上"
            try:
                fut = asyncio.run_coroutine_threadsafe(start(), loop)
            except Exception as exc:  # noqa: BLE001
                return str(exc)

            def _done(f: "asyncio.Future[dict[str, Any]]") -> None:
                if not f.cancelled() and f.exception() is not None:
                    log_.error("记忆归档没起来：%s", f.exception())

            fut.add_done_callback(_done)
            return None

        try:
            queue = ingest.pending(root)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"扫 raw 失败：{exc}"}
        if not queue:
            # 「两个都跑」但素材已经是最新：记忆归档该跑还是跑，别因为前半段没活就整条停下
            if then_memory and _kick_memory() is None:
                return {"ok": True, "queued": 0, "then_memory": True,
                        "message": "没有待归纳的素材，已接着整理记忆"}
            return {"ok": True, "queued": 0, "message": "没有待归纳的素材"}

        # 库内 [ingest] 段兜底：请求体没显式给就以 prefs 为准。
        # 前端不传 limit / dry_run，所以默认行为完全由库自己的 preferences.toml 决定。
        limit = int(body.get("limit") or 0) or int(getattr(cfg.ingest, "max_per_run", 0) or 0)
        dry_run = bool(body.get("dry_run")) or bool(getattr(cfg.ingest, "preview_first", False))
        auto_index = bool(getattr(cfg.ingest, "auto_index", True))
        run_id = time.strftime("%Y%m%d-%H%M%S")
        if not ingest.claim(run_id, str(root)):
            return {"ok": False, "error": "上一轮还没跑完，等它结束"}

        async def _run_then_memory() -> dict[str, Any]:
            """素材收完再接着整理记忆。

            串在后端而不是前端：浏览器一关就断的串接不算串接。这里只是
            "开跑"记忆归档（那个入口本身就是丢后台任务立刻返回），不等它跑完 ——
            一轮归档要挨个会话调模型，等下去会把素材这轮的返回值也拖住。
            """
            got = await ingest.run(
                root,
                llm,
                limit=limit,
                dry_run=dry_run,
                run_id=run_id,
                model=cfg.llm.ingest_model,
                max_tokens=cfg.llm.ingest_max_tokens,
                claimed=True,
                auto_index=auto_index,
            )
            if then_memory:
                start = (self.ctx.get("ingest") or {}).get("start")
                if start is None:
                    log_.warning("记忆归档没接上，跳过「两个都跑」的后半段")
                else:
                    try:
                        await start()
                    except Exception as exc:  # noqa: BLE001
                        log_.error("素材收完了，记忆归档没起来：%s", exc)
            return got

        coro = _run_then_memory()

        def _done(fut: "asyncio.Future[dict[str, Any]]") -> None:
            # 后台任务没人 await，异常不取出来会被 asyncio 静默吞掉
            if not fut.cancelled() and fut.exception() is not None:
                log_.error("一键归纳异常：%s", fut.exception())

        asyncio.run_coroutine_threadsafe(coro, loop).add_done_callback(_done)
        # 报出去的是这轮真要跑几份，不是扫出来几份。限了 2 份却告诉人"开跑：3 份"，
        # 进度条走到 2/2 结束，人第一反应是"它偷工减料了"。
        will = min(len(queue), limit) if limit and limit > 0 else len(queue)
        return {"ok": True, "run_id": run_id, "queued": will,
                "remaining": len(queue) - will,
                "dry_run": dry_run, "then_memory": then_memory}

    def _ingest_undo(self, body: dict[str, Any]) -> dict[str, Any]:
        """撤回一轮：文件字节级还原 + 素材名改回去 + 水位回退。"""
        root = self._scope()["persona"].root
        run_id = str(body.get("run_id") or "").strip()
        try:
            got = ingest.undo(root, run_id)
        except Exception as exc:  # noqa: BLE001
            log_.exception("撤回失败")
            return {"ok": False, "error": str(exc)}
        if got.get("ok"):
            log_.info("撤回完成：%s", got.get("stamps"))
        return got

    def _config(self) -> dict[str, Any]:
        raw = self.ctx.get("raw_config") or {}
        scope = self._scope()
        # 当前库的生效配置：启动库是 run.py 装配的那份，别的库是全局打底 + 库内覆盖
        base: Config = self.ctx["cfg"]
        cfg: Config = scope.get("cfg") or base
        root = scope["persona"].root
        prefs = library.read_prefs(root)
        values = {}
        for f in FIELDS:
            path = f["path"]
            if f.get("readonly"):
                # 启动库不是库的属性，是"现在在跑谁"。显示当前库本身：
                # 点进哪个库就看到哪个库，不会误以为自己能在这里改开机默认。
                values[path] = str(root)
                continue
            # 随库字段读当前库的 preferences，全局字段读 config.toml。
            # 文件里有就用文件里的原值；没有就回退到当前生效值，
            # 否则前端拿到的是空输入框，一保存就提交空串
            if f.get("scope") == "persona":
                got = tomlwrite.get_path(prefs, path, None)
            else:
                got = tomlwrite.get_path(raw, path, None)
            values[path] = got if got is not None else _from_cfg(cfg, path)
        return {
            "fields": FIELDS,
            "values": values,
            "path": str(self.ctx.get("cfg_path", "")),
            "persona": {
                "id": scope["id"],
                "root": str(root),
                "prefs_path": str(library.prefs_path(root)),
                "boot": bool(scope["boot"]),
                "has_prefs": library.prefs_path(root).is_file(),
                "global_paths": sorted(library.GLOBAL_PATHS),
            },
            "has_key": bool(cfg.llm.resolve_key()),
            "key_env": cfg.llm.api_key_env,
            "key_from_env": not bool(cfg.llm.api_key) and bool(cfg.llm.resolve_key()),
        }

    def _save_config(self, body: dict[str, Any]) -> dict[str, Any]:
        raw = dict(self.ctx.get("raw_config") or {})
        scope = self._scope()
        root = scope["persona"].root
        prefs = library.read_prefs(root)
        changed: list[str] = []
        persona_changed: list[str] = []  # 生效值真变了的（提示与重启判据只看它）
        persona_wrote: list[str] = []    # 需要写进库内文件的（含"只是显式记一遍"）
        ignored: list[str] = []

        def _effective(path: str) -> Any:
            """这一项此刻的生效值：库内写着就用库内的，没写就是全局那份。"""
            got = tomlwrite.get_path(prefs, path, None)
            return got if got is not None else tomlwrite.get_path(raw, path, None)

        by_path = {f["path"]: f for f in FIELDS}
        for path, value in (body.get("values") or {}).items():
            spec = by_path.get(path)
            if spec is None:
                continue
            if spec.get("readonly"):
                # 启动库只由「切库」改写。配置页提交它一律不落盘：
                # 否则"打开 B 的配置页点保存"就把全局开机默认悄悄换成了 B。
                ignored.append(spec["label"])
                continue
            try:
                value = _coerce(spec, value)
            except (TypeError, ValueError) as exc:
                return {"ok": False, "error": f"{spec['label']}：{exc}"}

            if spec.get("scope") == "persona":
                before = _effective(path)
                if tomlwrite.get_path(prefs, path) != value:
                    tomlwrite.set_path(prefs, path, value)
                    persona_wrote.append(spec["label"])
                # 「库内文件里没写过」不等于「值变了」：新库的 preferences 是打底导出的，
                # 少写一项只是沿用全局值。按文件差异算的话，随便保存一次
                # 都会得到"整套配置都改了、需要重启"。
                if before != value:
                    persona_changed.append(spec["label"])
                continue

            if tomlwrite.get_path(raw, path) != value:
                tomlwrite.set_path(raw, path, value)
                changed.append(spec["label"])

        # 库级配置落在库自己的文件里，与全局配置互不覆盖
        if persona_wrote:
            library.write_prefs(root, prefs)
            log_.info("库级配置已保存：%s → %s", ", ".join(persona_wrote), root)

        cfg_path: Path = self.ctx["cfg_path"]
        wrote_global = bool(changed)
        if wrote_global:
            # 先留一份 .bak：配置写坏过的人都知道这行值钱
            try:
                if cfg_path.is_file():
                    cfg_path.with_suffix(".toml.bak").write_bytes(cfg_path.read_bytes())
            except OSError as exc:
                log_.warning("配置备份失败：%s", exc)
            cfg_path.write_text(tomlwrite.dumps(raw), encoding="utf-8")
            log_.info("配置已保存：%s → %s", ", ".join(changed), cfg_path)

        # 重新加载并热更新
        from . import config as config_mod

        cfg_path: Path = self.ctx["cfg_path"]
        is_boot = bool(scope["boot"])

        if is_boot:
            new_cfg = config_mod.load(cfg_path)
            # set_dir=False：刚在全局里改了 persona_dir 想换启动库时，别被库内路径盖回去
            library.apply_prefs(new_cfg, root, set_dir=False)
            self.ctx["cfg"] = new_cfg
            self.ctx["raw_config"] = raw
            agent = self.ctx.get("agent")
            if agent is not None:
                agent.apply_config(new_cfg)
            # 蒸馏台/素材台是在启动时用旧 cfg 构造的，只换 ctx 里的那一份不够：
            # 它们各自攥着老对象，改了"一段最多句数"再重新提取还是老结果。
            for key in ("distill", "sim"):
                svc = self.ctx.get(key)
                if svc is not None and hasattr(svc, "cfg"):
                    svc.cfg = new_cfg
        else:
            # 改的是"别人家"的库：文件写完就结束，绝不能热替换线上正在用的配置。
            # 该库的作用域缓存（persona/memory/sim/试聊 agent）失效，下次请求重建。
            new_cfg = self.ctx["cfg"]
            cache = self.ctx.get("scope_cache")
            if isinstance(cache, dict):
                cache.pop(scope["id"], None)

        # 只有真换了通道类型才必须重启（适配器对象是按通道构造的）。
        # 工坊不接通道：没有通道类型可换，改完即存即生效
        need_restart = False
        wrote_prefs = bool(persona_wrote)
        target = (
            str(cfg_path)
            if (changed and not wrote_prefs)
            else str(library.prefs_path(root))
        )
        nothing = not changed and not wrote_prefs
        if nothing and ignored:
            message = "「启动时加载的人格库」跟随切换，这里改不了；换库请用总端。"
        elif need_restart:
            message = f"已保存到「{scope['id']}」的 preferences.toml。切换通道类型需要重启程序。"
        elif is_boot:
            message = "已保存并即时生效。"
        else:
            message = "已保存到该人格库。切到此库运行时才会用到这套配置。"
        if ignored and not nothing:
            message += "（「启动时加载的人格库」跟随切换，已忽略）"
        return {
            "ok": True,
            "changed": changed + persona_wrote,
            "ignored": ignored,
            "need_restart": need_restart,
            "path": target,
            "wrote": {
                "prefs": wrote_prefs,
                "global": bool(changed),
                "boot": is_boot,
            },
            "saved_at": time.strftime("%H:%M:%S"),
            "has_key": bool(new_cfg.llm.resolve_key()),
            "message": message,
        }

    def _logs(self) -> None:
        query = self.path.split("?", 1)[1] if "?" in self.path else ""
        since = 0
        for part in query.split("&"):
            if part.startswith("since="):
                try:
                    since = int(part[6:])
                except ValueError:
                    since = 0
        items, total = log.RING.since(since)
        self._json({"items": items, "total": total})

    def _query_param(self, name: str) -> str:
        from urllib.parse import unquote

        query = self.path.split("?", 1)[1] if "?" in self.path else ""
        for part in query.split("&"):
            if part.startswith(name + "="):
                return unquote(part[len(name) + 1:])
        return ""

    # ---------- 会话记录：浏览 / 删除 / 手动压缩 ----------

    def _session_view(self) -> dict[str, Any]:
        mem = self._mem()
        sid = self._query_param("session")
        if not sid:
            return {"ok": False, "error": "缺 session 参数"}
        msgs, total = mem.session_messages(sid, limit=200)
        summary, upto = mem.get_summary(sid)

        # 群聊按发言人分组：同一个人的话归一处，界面才能分列表看。
        # 分组以内容前缀里的名字为准——老消息的 user_id 是空的，
        # 只认 user_id 的话，升级之前的历史一条都分不出来。
        members: list[dict[str, Any]] = []
        if sid.startswith("group:"):
            groups: dict[str, dict[str, Any]] = {}
            for m in msgs:
                if m.get("role") != "user":
                    continue
                name = _speaker_name(m.get("content") or "") or str(m.get("user_id") or "")
                if not name:
                    continue
                item = groups.setdefault(name, {"name": name, "user_id": "", "count": 0})
                item["count"] += 1
                if not item["user_id"] and m.get("user_id"):
                    item["user_id"] = str(m["user_id"])
            members = sorted(groups.values(), key=lambda x: -int(x["count"]))

        return {
            "ok": True,
            "session": sid,
            "messages": msgs,
            "total": total,
            "truncated": total > len(msgs),
            "summary": summary,
            "upto": upto,
            "members": members,
        }

    def _session_delete(self, body: dict[str, Any]) -> dict[str, Any]:
        mem = self._mem()
        sid = str(body.get("session") or "")
        if not sid:
            return {"ok": False, "error": "缺 session"}
        deleted = mem.clear_session(sid)
        log_.info("已删除会话记录 %s（%d 条消息与摘要）", sid, deleted)
        return {"ok": True, "deleted": deleted}

    def _session_compact(self, body: dict[str, Any]) -> dict[str, Any]:
        sid = str(body.get("session") or "")
        if not sid:
            return {"ok": False, "error": "缺 session"}
        agent = self._scope_agent()
        if agent is None:
            return {"ok": False, "error": "智能体尚未就绪，压不了"}
        if not agent.cfg.llm.resolve_key():
            return {"ok": False, "error": "缺少 API Key，压不了"}
        return self._run(agent.summarize_session(sid, force=True), timeout=180)

    def _persona(self) -> dict[str, Any]:
        persona = self._scope()["persona"]
        pages = persona.status()
        return {
            "root": str(persona.root),
            "summary": persona.summary,
            "summary_empty": persona.self_summary_is_empty,
            "routes": [{"situation": r.situation, "pages": r.pages} for r in persona._routes],
            "pages": [
                {"name": n, "filled": i.filled, "chars": i.chars} for n, i in pages.items()
            ],
        }

    # ---------- 身份母本 ----------

    def _seed(self) -> Seed:
        """母本通道：备份放当前库自己的 data/identity_history。

        启动库以 ctx["seed"] 为唯一真源（run.py 装配，测试也会中途注入），
        不走作用域缓存，否则先建的缓存会把后来注入的那份挡掉。
        """
        scope = self._scope()
        if scope.get("boot"):
            seed = self.ctx.get("seed")
            if seed is None:
                cfg: Config = self.ctx["cfg"]
                root = scope["persona"].root
                seed = Seed(
                    root,
                    library.resolve_history(root, cfg.path("data") / "identity_history"),
                )
                self.ctx["seed"] = seed
            return seed

        seed = scope.get("seed")
        if seed is None:
            cfg = self.ctx["cfg"]
            root = scope["persona"].root
            seed = Seed(
                root, library.resolve_history(root, cfg.path("data") / "identity_history")
            )
            scope["seed"] = seed
        return seed

    def _identity(self) -> dict[str, Any]:
        st = self._seed().status()
        st["template"] = TEMPLATE
        return st

    def _save_identity(self, body: dict[str, Any]) -> dict[str, Any]:
        text = body.get("content")
        if not isinstance(text, str):
            raise ValueError("缺少 content")
        st = self._seed().save(text)
        st["template"] = TEMPLATE
        return st

    # ---------- 素材台 ----------

    def _sim(self):
        studio = self._scope().get("sim")
        if studio is None:
            raise RuntimeError("素材台未启用（config.toml 里 [sim] enabled = false）")
        return studio

    def _llm_ready(self) -> bool:
        cfg: Config = self.ctx["cfg"]
        return bool(cfg.llm.resolve_key())

    def _sim_index(self) -> dict[str, Any]:
        st = self._sim()
        cfg: Config = self.ctx["cfg"]
        lib = st.library()
        return {
            "drafts": st.list_drafts(),
            "coverage": st.coverage(),
            "cards": [c.as_dict() for c in lib],
            "library_count": len(lib),
            "self_label": st.self_label,
            "root": str(st.root),
            "draft_dir": str(st.draft_dir),
            "card_dir": str(st.card_dir),
            "options": {
                "candidates": cfg.sim.candidates,
                "card_count": cfg.sim.card_count,
            },
            "relations": list(dims.RELATIONS),
            "categories": list(dims.CATEGORIES),
            "identity": self._seed_brief(),
        }

    def _seed_brief(self) -> dict[str, Any]:
        st = self._seed().status()
        return {
            "filled": st["filled"],
            "total": st["total"],
            "chars": len(self._seed().brief(1200)),
        }

    def _sim_get(self) -> dict[str, Any]:
        st = self._sim()
        query = self.path.split("?", 1)[1] if "?" in self.path else ""
        draft_id = ""
        for part in query.split("&"):
            if part.startswith("id="):
                draft_id = part[3:]
                break
        draft = st.get(draft_id)
        if draft is None:
            return {"ok": False, "error": "草稿不存在"}
        return {
            "ok": True,
            "draft": draft.as_dict(),
            "preview": st.preview(draft),
            "reaction_preview": st.preview_reaction(draft),
        }

    def _sim_cards(self, body: dict[str, Any]) -> dict[str, Any]:
        st = self._sim()
        if not self._llm_ready():
            return {"ok": False, "error": "缺少 API Key，先在「配置」页填上"}
        cfg: Config = self.ctx["cfg"]
        count = int(body.get("count") or cfg.sim.card_count or 3)
        relation = str(body.get("relation") or "").strip()
        if relation not in ("",) + dims.RELATIONS:
            return {"ok": False, "error": f"关系只能是 {' / '.join(dims.RELATIONS)}"}

        seed = self._seed()
        profile = seed.brief(1200)
        ident = seed.status()
        if relation == "拥有者" and len(profile) < 60:
            return {
                "ok": False,
                "error": "身份母本还没写，拥有者这一档造不出来——先去「身份」页写她是谁、你怎么跟她相处",
                "identity": {"filled": ident["filled"], "total": ident["total"], "chars": len(profile)},
            }
        avoid = [c.name for c in st.library()]
        started = time.monotonic()
        try:
            cards = self._run(
                generate_cards(
                    st.llm,
                    count,
                    avoid=avoid,
                    max_tokens=cfg.sim.card_max_tokens,
                    relation=relation,
                    profile=profile,
                ),
                timeout=180,
            )
        except Exception as exc:  # noqa: BLE001
            log_.warning("生成情境失败：%s", exc)
            return {"ok": False, "error": str(exc)}
        return {
            "ok": True,
            "cards": [c.as_dict() for c in cards],
            "elapsed_ms": int((time.monotonic() - started) * 1000),
            "identity": {
                "filled": ident["filled"],
                "total": ident["total"],
                "chars": len(profile),
            },
        }

    def _sim_draw(self, body: dict[str, Any]) -> dict[str, Any]:
        """从库里抽卡。库空时明确报出来，不静默返回空列表。"""
        st = self._sim()
        cfg: Config = self.ctx["cfg"]
        n = int(body.get("n") or cfg.sim.card_count or 3)
        cards = st.draw_cards(n)
        if not cards:
            return {"ok": False, "error": "库里还没有卡，先生成几张并收进库"}
        return {"ok": True, "cards": [c.as_dict() for c in cards], "total": len(st.library())}

    def _sim_delete_card(self, body: dict[str, Any]) -> dict[str, Any]:
        st = self._sim()
        ok = st.delete_card(str(body.get("id") or ""))
        return {"ok": ok, "error": "" if ok else "卡片不存在", "total": len(st.library())}

    def _sim_new(self, body: dict[str, Any]) -> dict[str, Any]:
        """开一稿。素材台不再对演，所以默认不落开场白——开场白只作为模拟条件的一份输入。"""
        st = self._sim()
        card = Card.from_dict(body.get("card") or {})
        if not card.name:
            return {"ok": False, "error": "卡片为空"}
        seed = bool(body.get("opening", False))
        draft = st.new_draft(card, str(body.get("mode") or "other"), opening=seed)
        return {"ok": True, "draft": draft.as_dict(), "preview": st.preview(draft)}

    def _sim_cand(self, body: dict[str, Any]) -> dict[str, Any]:
        """说话 + 出候选，合成一次往返。

        谁在演谁只看草稿的 mode（换下拉时会先落盘），不接受前端口头指定：
        以前两边各读一处，切完下拉后标签写「她本人」、存档还是「对方」，
        结果就是"我在演她，AI 却接着给她的话"。
        """
        st = self._sim()
        draft = st.get(str(body.get("id") or ""))
        if draft is None:
            return {"ok": False, "error": "草稿不存在"}
        if not self._llm_ready():
            return {"ok": False, "error": "缺少 API Key"}

        mine = draft.mode
        theirs = draft.ai_role
        text = str(body.get("text") or "").strip()
        if text:
            st.commit(draft, mine, text, "human")

        cfg: Config = self.ctx["cfg"]
        n = int(body.get("n") or cfg.sim.candidates or 3)
        started = time.monotonic()
        try:
            items = self._run(st.candidates(draft, theirs, n), timeout=180)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc), "draft": draft.as_dict(), "role": theirs}
        return {
            "ok": True,
            "candidates": items,
            "role": theirs,
            "draft": draft.as_dict(),
            "elapsed_ms": int((time.monotonic() - started) * 1000),
        }

    def _sim_mode(self, body: dict[str, Any]) -> dict[str, Any]:
        """改「我在演谁」。落盘之后，候选方向与界面标签都以它为准。"""
        st = self._sim()
        draft = st.get(str(body.get("id") or ""))
        if draft is None:
            return {"ok": False, "error": "草稿不存在"}
        draft = st.set_mode(draft, str(body.get("mode") or "other"))
        return {"ok": True, "draft": draft.as_dict(), "role": draft.ai_role}

    def _sim_commit(self, body: dict[str, Any]) -> dict[str, Any]:
        st = self._sim()
        draft = st.get(str(body.get("id") or ""))
        if draft is None:
            return {"ok": False, "error": "草稿不存在"}
        dropped = body.get("dropped") or []
        draft = st.commit(
            draft,
            # 没说就是"AI 那一方"：采纳的候选本来就只出这一边
            str(body.get("role") or draft.ai_role),
            str(body.get("text") or ""),
            str(body.get("source") or "human"),
            [str(x) for x in dropped] if isinstance(dropped, list) else [],
        )
        return {"ok": True, "draft": draft.as_dict()}

    def _sim_undo(self, body: dict[str, Any]) -> dict[str, Any]:
        st = self._sim()
        draft = st.get(str(body.get("id") or ""))
        if draft is None:
            return {"ok": False, "error": "草稿不存在"}
        draft = st.undo(draft)
        return {"ok": True, "draft": draft.as_dict()}

    def _sim_edit(self, body: dict[str, Any]) -> dict[str, Any]:
        """就地改写一句已经说过的话（含卡片开场白）。文本留空等于删掉这句。"""
        st = self._sim()
        draft = st.get(str(body.get("id") or ""))
        if draft is None:
            return {"ok": False, "error": "草稿不存在"}
        try:
            index = int(body.get("index"))
        except (TypeError, ValueError):
            return {"ok": False, "error": "句子下标不合法", "draft": draft.as_dict()}
        if not 0 <= index < len(draft.turns):
            return {"ok": False, "error": "这句已经不在了，刷新一下", "draft": draft.as_dict()}
        draft = st.edit_turn(draft, index, str(body.get("text") or ""))
        return {"ok": True, "draft": draft.as_dict()}

    def _sim_turn(self, body: dict[str, Any]) -> dict[str, Any]:
        """对话句子的结构操作：改字 / 换说话人 / 插入 / 挪位 / 删除。

        全都带 index，前端不必自己维护数组，改完把整份草稿发回去重画。
        """
        st = self._sim()
        draft = st.get(str(body.get("id") or ""))
        if draft is None:
            return {"ok": False, "error": "草稿不存在"}

        op = str(body.get("op") or "text")
        try:
            index = int(body.get("index", -1))
        except (TypeError, ValueError):
            index = -1
        text = str(body.get("text") or "")
        role = "self" if body.get("role") == "self" else "other"

        if op in {"text", "delete"}:
            if not 0 <= index < len(draft.turns):
                return {"ok": False, "error": "这句已经不在了，刷新一下", "draft": draft.as_dict()}
            draft = st.edit_turn(draft, index, text if op == "text" else "")
        elif op == "role":
            if not 0 <= index < len(draft.turns):
                return {"ok": False, "error": "这句已经不在了，刷新一下", "draft": draft.as_dict()}
            draft = st.set_turn_role(draft, index, role)
        elif op == "insert":
            if not text.strip():
                return {"ok": False, "error": "要插的那句是空的", "draft": draft.as_dict()}
            try:
                at = int(body.get("at", len(draft.turns)))
            except (TypeError, ValueError):
                at = len(draft.turns)
            draft = st.insert_turn(draft, at, role, text)
        elif op == "move":
            delta = 1 if str(body.get("dir") or "down") == "down" else -1
            draft = st.move_turn(draft, index, delta)
        else:
            return {"ok": False, "error": f"不认识的操作：{op}"}

        return {"ok": True, "draft": draft.as_dict()}

    def _sim_draft_card(self, body: dict[str, Any]) -> dict[str, Any]:
        """改正在演的这一稿的情境条件（关系 / 场合 / 卡点 / 对方 / 分类）。"""
        st = self._sim()
        draft = st.get(str(body.get("id") or ""))
        if draft is None:
            return {"ok": False, "error": "草稿不存在"}
        card = Card.from_dict(body.get("card") or {})
        if not card.name:
            return {"ok": False, "error": "卡片标题不能为空"}
        # 前端只送用户能改的那几个字段，id/origin/used 沿用原卡，免得把库里的记账抹掉
        card.id = card.id or draft.card.id
        card.origin = draft.card.origin if card.origin == "generated" else card.origin
        card.used = max(card.used, draft.card.used)
        card.created = draft.card.created or card.created
        card.saved_at = card.saved_at or draft.card.saved_at
        draft = st.set_card(draft, card)
        return {"ok": True, "draft": draft.as_dict(), "preview": st.preview(draft)}

    def _sim_export(self, body: dict[str, Any]) -> dict[str, Any]:
        """收稿。form="dialog" 出对话稿，form="reaction" 出反应卡。"""
        st = self._sim()
        draft = st.get(str(body.get("id") or ""))
        if draft is None:
            return {"ok": False, "error": "草稿不存在"}
        draft.note = str(body.get("note") or "")
        form = str(body.get("form") or "dialog").strip().lower()
        if form in {"reaction", "react", "style"}:
            # 转没转、有没有违规、过没过审，全由 export_reaction 自己判
            result = st.export(draft, form="reaction")
            return {
                "ok": True,
                **result,
                "draft_dir": str(st.draft_dir),
                "draft": draft.as_dict(),
            }
        if len(draft.turns) < 2:
            return {"ok": False, "error": "至少要有来有回一句，才值得收稿"}
        result = st.export(draft, form="dialog")
        return {"ok": True, **result, "draft_dir": str(st.draft_dir)}

    def _sim_reactgen(self, body: dict[str, Any]) -> dict[str, Any]:
        """按母本 + 刺激/档位直接出几张反应卡。素材台现在的主入口。

        没有情境卡这一步：不传场合、不传对方，模型也就编不出来。
        """
        st = self._sim()
        try:
            n = int(body.get("n") or 3)
        except (TypeError, ValueError):
            n = 3
        cards = self._run(
            st.generate_reactions(
                stimulus=str(body.get("stimulus") or ""),
                level=str(body.get("level") or ""),
                n=n,
            ),
            timeout=240,
        )
        return {"ok": True, "cards": cards}
    def _sim_reaction(self, body: dict[str, Any]) -> dict[str, Any]:
        """模拟反应卡。默认按母本 + 卡片条件推断；from_turns 时抽象已有台词（老稿/蒸馏段）。"""
        st = self._sim()
        draft = st.get(str(body.get("id") or ""))
        if draft is None:
            return {"ok": False, "error": "草稿不存在"}
        if body.get("from_turns"):
            return self._run(st.to_reaction(draft), timeout=180)
        return self._run(st.simulate(draft), timeout=180)

    def _sim_reaction_update(self, body: dict[str, Any]) -> dict[str, Any]:
        """改卡面 / 勾已过审 / 勾可作范例 / 填例句。改完重跑校验。"""
        st = self._sim()
        draft = st.get(str(body.get("id") or ""))
        if draft is None:
            return {"ok": False, "error": "草稿不存在"}
        reviewed = body.get("reviewed")
        exemplar = body.get("exemplar")
        example = body.get("example")
        draft = st.update_reaction(
            draft,
            body.get("card") if isinstance(body.get("card"), dict) else None,
            None if reviewed is None else bool(reviewed),
            None if exemplar is None else bool(exemplar),
            example if isinstance(example, str) else None,
        )
        return {
            "ok": True,
            "draft": draft.as_dict(),
            "preview": st.preview_reaction(draft),
        }

    def _sim_react_example(self, body: dict[str, Any]) -> dict[str, Any]:
        """给反应卡生成例句。只出句子不落盘：过不过、改不改，人说了算。"""
        st = self._sim()
        draft = st.get(str(body.get("id") or ""))
        if draft is None:
            return {"ok": False, "error": "草稿不存在"}
        try:
            example = self._run(st.generate_example(draft), timeout=120)
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}
        return {"ok": True, "example": example, "draft": draft.as_dict()}

    def _sim_save_card(self, body: dict[str, Any]) -> dict[str, Any]:
        st = self._sim()
        card = Card.from_dict(body.get("card") or {})
        if not card.name:
            return {"ok": False, "error": "卡片为空"}
        result = st.save_card(card)
        result["total"] = len(st.library())
        return result

    # ---------- 小说蒸馏器 ----------

    def _distill(self):
        d = self._scope().get("distill")
        if d is None:
            raise RuntimeError("蒸馏器未启用（config.toml 里 [distill] enabled = false）")
        return d

    def _distill_get(self) -> dict[str, Any]:
        book_id = self._query_param("id")
        if not book_id:
            return {"ok": False, "error": "缺 id"}
        try:
            offset = int(self._query_param("offset") or 0)
            limit = int(self._query_param("limit") or 200)
        except ValueError:
            offset, limit = 0, 200
        data = self._distill().get(book_id, offset, limit)
        data["llm_ready"] = self._llm_ready()
        return data

    def _distill_import(self, body: dict[str, Any]) -> dict[str, Any]:
        """接受 base64（浏览器读文件）或纯文本（命令行/测试）两种投喂方式。"""
        name = str(body.get("name") or "").strip()
        if body.get("b64"):
            raw = str(body["b64"])
            raw = raw.split(",", 1)[-1] if raw.startswith("data:") else raw
            try:
                data = base64.b64decode(raw, validate=False)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"文件解码失败：{exc}") from exc
            result = self._distill().import_bytes(name, data)
        else:
            result = self._distill().import_text(name, str(body.get("text") or ""))
        return {"ok": True, **result}

    def _distill_extract(self, body: dict[str, Any]) -> dict[str, Any]:
        aliases = body.get("aliases") or []
        if isinstance(aliases, str):
            aliases = [x.strip() for x in aliases.replace("，", ",").replace("、", ",").split(",")]
        result = self._distill().extract(
            str(body.get("book") or ""),
            str(body.get("character") or ""),
            [str(x).strip() for x in aliases if str(x).strip()],
            str(body.get("level") or "熟人"),
            int(body.get("limit") or 200),
            _parse_chapters(body.get("chapters")),
        )
        return result

    def _distill_segment(self, body: dict[str, Any]) -> dict[str, Any]:
        changes = body.get("changes")
        if not isinstance(changes, dict):
            changes = {k: v for k, v in body.items() if k not in {"book", "id"}}
        return self._distill().update_segment(
            str(body.get("book") or ""), str(body.get("id") or ""), changes
        )

    def _distill_line(self, body: dict[str, Any]) -> dict[str, Any]:
        try:
            index = int(body.get("index"))
        except (TypeError, ValueError):
            raise ValueError("句子下标不合法")
        return self._distill().update_line(
            str(body.get("book") or ""),
            str(body.get("id") or ""),
            index,
            str(body.get("role") or "self"),
        )

    def _distill_export(self, body: dict[str, Any]) -> dict[str, Any]:
        ids = body.get("ids") or []
        return self._distill().export(
            str(body.get("book") or ""), [str(x) for x in ids] if isinstance(ids, list) else []
        )

    def _distill_to_sim(self, body: dict[str, Any]) -> dict[str, Any]:
        return self._distill().to_sim(str(body.get("book") or ""), str(body.get("id") or ""))

    def _distill_to_library(self, body: dict[str, Any]) -> dict[str, Any]:
        return self._distill().to_library(str(body.get("book") or ""), str(body.get("id") or ""))

    def _distill_react(self, body: dict[str, Any]) -> dict[str, Any]:
        """转反应卡要调模型，是异步的，必须丢回主循环。"""
        return self._run(
            self._distill().to_reaction(str(body.get("book") or ""), str(body.get("id") or "")),
            timeout=180,
        )

    def _distill_react_update(self, body: dict[str, Any]) -> dict[str, Any]:
        reviewed = body.get("reviewed")
        exemplar = body.get("exemplar")
        example = body.get("example")
        return self._distill().update_reaction(
            str(body.get("book") or ""),
            str(body.get("id") or ""),
            body.get("card") if isinstance(body.get("card"), dict) else None,
            None if reviewed is None else bool(reviewed),
            None if exemplar is None else bool(exemplar),
            example if isinstance(example, str) else None,
        )

    def _distill_react_export(self, body: dict[str, Any]) -> dict[str, Any]:
        ids = body.get("ids") or []
        return self._distill().export_react(
            str(body.get("book") or ""), [str(x) for x in ids] if isinstance(ids, list) else []
        )

    def _chat(self, body: dict[str, Any]) -> dict[str, Any]:
        text = str(body.get("text") or "").strip()
        if not text:
            return {"ok": False, "error": "说点什么"}
        agent = self._scope_agent()
        if agent is None:
            return {"ok": False, "error": "智能体尚未就绪"}
        if not self.ctx["cfg"].llm.resolve_key():
            return {"ok": False, "error": "缺少 API Key，先在「模型」页填上"}
        try:
            result = self._run(agent.preview(text, remember=bool(body.get("remember"))))
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}
        return {"ok": True, **result}


def _parse_chapters(value: Any) -> list[int]:
    """解析章节范围。网页传 0 基下标列表，手打可以传 "3-8,12"（1 基）。

    空值返回 []，表示全书——不传这个参数的老调用行为不变。
    """
    if value is None or value == "" or value == []:
        return []
    if isinstance(value, list):
        out = []
        for item in value:
            try:
                out.append(int(item))
            except (TypeError, ValueError):
                continue
        return out
    out: list[int] = []
    for part in str(value).replace("，", ",").replace("、", ",").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            if "-" in part:
                a, b = part.split("-", 1)
                lo, hi = int(a), int(b)
                if lo > hi:
                    lo, hi = hi, lo
                out.extend(range(lo - 1, hi))
            else:
                out.append(int(part) - 1)
        except ValueError:
            continue
    return out


def _from_cfg(cfg: Any, path: str, default: Any = None) -> Any:
    """按表单路径从已加载的 Config 里取值，用于配置文件缺项时兜底。

    配置文件是用户手写的，新加的字段（比如 reply.max_chunks）在老的
    config.toml 里根本不存在。这时如果表单回 None，前端就渲染成空输入框，
    用户一点保存就把空串提交回来，int("") 直接抛 ValueError ——
    表现成"我什么都没改，保存却失败"。
    """
    parts = path.split(".")
    # 表单里叫 adapter.official.xxx，Config 上直接是 official.xxx
    if parts and parts[0] == "adapter":
        parts = parts[1:]
    cur: Any = cfg
    for part in parts:
        cur = getattr(cur, part, None)
        if cur is None:
            return default
    return cur


def _coerce(spec: dict[str, Any], value: Any) -> Any:
    t = spec["type"]
    if t == "bool":
        if isinstance(value, bool):
            return value
        s = str(value).strip().lower()
        if s in {"", "none", "null"}:
            # 空串当 False 会把开着的开关悄悄关掉，宁可报出来
            raise ValueError("不能留空")
        return s in {"1", "true", "yes", "on"}
    if t == "int":
        try:
            return int(str(value).strip())
        except (TypeError, ValueError):
            raise ValueError(f"要填整数，现在是 {value!r}") from None
    if t == "float":
        try:
            return float(str(value).strip())
        except (TypeError, ValueError):
            raise ValueError(f"要填数字，现在是 {value!r}") from None
    if t == "list":
        if isinstance(value, list):
            return [str(v).strip() for v in value if str(v).strip()]
        return [x.strip() for x in str(value).replace("\n", ",").split(",") if x.strip()]
    if t == "select":
        s = str(value)
        if spec.get("options") and s not in spec["options"]:
            raise ValueError(f"取值必须是 {spec['options']} 之一")
        return s
    return str(value)


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def start(ctx: dict[str, Any], host: str = "127.0.0.1", port: int = 8770) -> tuple[str, _Server]:
    """在后台线程启动 WebUI，返回 (url, server)。"""
    _Handler.ctx = ctx
    last_err: Exception | None = None
    for candidate in range(port, port + 12):
        try:
            server = _Server((host, candidate), _Handler)
        except OSError as exc:
            last_err = exc
            continue
        thread = threading.Thread(target=server.serve_forever, name="webui", daemon=True)
        thread.start()
        url = f"http://{host}:{candidate}"
        log_.info("WebUI 已启动：%s", url)
        return url, server
    raise RuntimeError(f"WebUI 端口全部占用（从 {port} 试到 {port + 11}）：{last_err}")
