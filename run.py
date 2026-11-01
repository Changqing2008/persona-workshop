"""人格工坊启动入口：总端 + 素材台 + 蒸馏台 + 试聊，不连任何通道。

与通道器程序的分工（见 人格工坊/contract/库契约.md）：
- 这里产出并维护人格库 / 记忆库，库是通用资产；
- QQ 聊天、DSH、其他智能体对话都是库的消费者，各自作为独立程序接入。

启动：
    python run.py              # 开控制台
    python run.py --check      # 自检：配置 / 人格库 / 库契约
    python run.py --port 8780 --no-open
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "tools"))

from workshop import __version__, config, library, log, webui  # noqa: E402
from workshop.agent import Agent  # noqa: E402
from workshop.distill import Distiller  # noqa: E402
from workshop.llm import LLMClient  # noqa: E402
from workshop.memory import Memory  # noqa: E402
from workshop.persona import Persona  # noqa: E402
from workshop.seed import Seed  # noqa: E402
from workshop.sim import Studio  # noqa: E402

# 记忆归档器是可选组件：缺依赖或语法错不该拖垮整个工坊。
try:
    import mem_ingest  # noqa: E402
except Exception as _exc:  # noqa: BLE001
    mem_ingest = None
    _INGEST_IMPORT_ERROR = str(_exc)
else:
    _INGEST_IMPORT_ERROR = ""


def _pid_alive(pid: int) -> bool:
    """进程是否还活着。判断上一次的 pid 文件是残留还是真有人在跑。"""
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            ok = kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
            return bool(ok) and code.value == STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _lock_state(path: Path) -> tuple[str, int]:
    """读锁文件：第一行 PID，第二行端口（旧版只写了 PID）。"""
    try:
        rows = [r.strip() for r in path.read_text(encoding="utf-8").splitlines()]
    except (PermissionError, OSError):
        return "", 0
    pid = rows[0] if rows and rows[0].isdigit() else ""
    port = int(rows[1]) if len(rows) > 1 and rows[1].isdigit() else 0
    return pid, port


def _port_state(host: str, port: int) -> str:
    """端口三态：alive / dead / unknown。

    只把**明确拒绝连接**当成 dead，其它一律 unknown——宁可保守地认为老实例还活着，
    也不能因为探测失手就让两个实例抢一个数据目录。

    这里只能用阻塞 connect：两个 Windows 坑都实测过——
    带 settimeout 时自己拒绝的连接会报成 TimeoutError；非阻塞 connect_ex 两次
    select 都拿不到拒连，只能一直 unknown。只有阻塞 connect 的异常是准的。
    代价是死端口要约 2 秒，所以只在拿锁时判一次，不拿它去扫端口。
    """
    if port <= 0:
        return "unknown"
    import socket

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.connect((host or "127.0.0.1", int(port)))
        return "alive"
    except ConnectionRefusedError:
        return "dead"
    except OSError:
        return "unknown"
    finally:
        s.close()


def remember_port(path: Path, port: int, pid: int | str | None = None) -> None:
    """把实际端口写进锁文件，下次有人启动就知道该把浏览器指到哪。

    pid 默认为自己；帮别人补记时要显式传持有者的 pid——锁是谁的就得写谁，
    否则真正的持有者退出时认不出这行 PID，会把锁当别人的留着不删。
    """
    try:
        path.write_text(f"{pid or os.getpid()}\n{int(port)}\n", encoding="utf-8")
    except OSError:
        pass


_PORT_LINE = re.compile(r"WebUI 已启动：https?://[^:\s]+:(\d+)")


def ports_from_log(log_path: Path, limit: int = 8) -> list[int]:
    """日志里最近报过的控制台端口，越新越靠前。

    比 netstat 可靠得多：实测这个环境里 `netstat -ano` 拿到的 TCP 表只有 76 行
    （全量几百行），经常漏掉目标进程，连 PID 4 的 445 都时有时无。日志是工坊
    自己写的，一条不缺，而且天然是"最近用过的端口优先"。

    只读末尾 64KB：日志会越滚越大，全读进来没必要。
    """
    try:
        with log_path.open("rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            fh.seek(max(0, size - 65536))
            tail = fh.read().decode("utf-8", "replace")
    except OSError:
        return []
    out: list[int] = []
    for m in reversed(_PORT_LINE.findall(tail)):
        port = int(m)
        if port not in out:
            out.append(port)
        if len(out) >= limit:
            break
    return out


def _ports_of_pid(pid: int) -> list[int]:
    """查这个进程在监听哪些端口（只靠系统自带的 netstat，不引第三方库）。

    比盲扫端口快得多，而且不受 `--port` 起在哪的影响——8780 附近扫 64 个口
    对 8799 这种还够，对 9000 那种就白扫。拿不到就返回空，交给日志和扫描兜底。
    """
    if pid <= 0 or os.name != "nt":
        return []
    import subprocess

    try:
        out = subprocess.run(
            ["netstat", "-ano", "-p", "TCP"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=8,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    ports: list[int] = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) < 5 or parts[0].upper() != "TCP":
            continue
        if parts[3].upper() != "LISTENING" or parts[4] != str(pid):
            continue
        port = parts[1].rpartition(":")[2]
        if port.isdigit() and int(port) not in ports:
            ports.append(int(port))
    return ports


def release_instance_lock(path: Path) -> None:
    """正常退出时删掉自己的锁。只删自己写的那份，别把别人的锁误删。"""
    pid, _port = _lock_state(path)
    if pid and pid != str(os.getpid()):
        return
    try:
        path.unlink()
    except OSError:
        pass


def _is_workshop(host: str, port: int) -> bool:
    """对面是不是工坊控制台。

    端口能连上不代表就是工坊（别的程序也可能占着），所以真发一个 HTTP 请求，
    认页面标题里的「人格工坊」。认错了会把浏览器指到一个陌生服务上。
    """
    import socket

    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(1.5)
        try:
            s.connect((host or "127.0.0.1", int(port)))
            s.sendall(b"GET / HTTP/1.0\r\nHost: 127.0.0.1\r\n\r\n")
            buf = b""
            while len(buf) < 8192:
                chunk = s.recv(4096)
                if not chunk:
                    break
                buf += chunk
        finally:
            s.close()
    except OSError:
        return False
    return b"<title>" in buf and "人格工坊".encode() in buf


def _scan_batch(host: str, ports: list[int], timeout: float) -> list[int]:
    """一批端口并发探测（单批别超过 500：Windows 的 select 就有这个上限）。"""
    import select
    import socket

    socks: dict[int, "socket.socket"] = {}
    for p in ports:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setblocking(False)
        try:
            s.connect_ex((host or "127.0.0.1", int(p)))
        except OSError:
            s.close()
            continue
        socks[p] = s

    alive: list[int] = []
    pending = dict(socks)
    end = time.time() + timeout
    try:
        while pending and time.time() < end:
            try:
                _, writable, errored = select.select(
                    [], list(pending.values()), list(pending.values()), 0.1
                )
            except OSError:
                break
            for s in set(writable) | set(errored):
                hit = next((p for p, x in pending.items() if x is s), None)
                if hit is None:
                    continue
                if s.getsockopt(socket.SOL_SOCKET, socket.SO_ERROR) == 0:
                    alive.append(hit)
                pending.pop(hit, None)
    finally:
        for s in socks.values():
            s.close()
    return alive


def scan_ports(host: str, ports: list[int], timeout: float = 0.8, batch: int = 200) -> list[int]:
    """一次性并发探测一批端口，返回在听的那些。

    串行探不行：Windows 上单个拒连端口要等 0.3 秒才给结论，64 个就是 20 秒。
    同时发起再 select 一次，总共几百毫秒。

    要分批：Windows 的 select 只认 512 个句柄，一口气传几百上千个会直接
    ValueError: too many file descriptors。分批后扫多宽的区间都一样稳。
    """
    ports = list(ports)
    alive: list[int] = []
    for i in range(0, len(ports), batch):
        alive.extend(_scan_batch(host, ports[i : i + batch], timeout))
    return sorted(set(alive))


def find_consoles(host: str, center: int, span: int = 64) -> list[int]:
    """在配置端口附近扫一圈，找出真正在跑的工坊控制台。

    旧版实例没把端口写进锁文件，只能靠扫。启动端口可能被 `--port` 改成任何值
    （实测有 8799 这种离默认值将近二十位的），所以不能只试默认那一个。
    """
    start = max(1, center - span // 2)
    candidates = scan_ports(host, list(range(start, start + span)))
    return [p for p in candidates if _is_workshop(host, p)]


def acquire_instance_lock(path: Path) -> tuple[bool, str]:
    """同一个数据目录只允许一个工坊实例（pid 文件 + 存活检测）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    holder, port = _lock_state(path)
    alive = _pid_alive(int(holder)) if holder else False
    # 进程活着但端口明确拒连：说明那个实例已经卡死，锁是残留的，让新实例接手
    if alive and _port_state("127.0.0.1", port) == "dead":
        alive = False
    if alive:
        return False, holder
    me = str(os.getpid())
    try:
        path.write_text(f"{me}\n", encoding="utf-8")
        got = path.read_text(encoding="utf-8").strip()
    except PermissionError:
        return False, ""
    except OSError:
        return True, ""
    if got and got != me:
        return False, got
    return True, ""


def _selfcheck(
    cfg: config.Config, persona: Persona, memory: Memory, test_llm: bool = True
) -> int:
    print("\n=== 自检（人格工坊）===")

    problems = cfg.validate()
    # 工坊不接通道：adapter 相关的缺项不算数
    problems = [p for p in problems if not p.startswith(("adapter", "通道"))]
    if problems:
        print("\n配置问题：")
        for p in problems:
            print(f"  ✗ {p}")
    else:
        print("\n配置： ✓ 通过")

    print("\n人格库：")
    print("  " + persona.describe().replace("\n", "\n  "))

    seed = Seed(
        persona.root,
        library.resolve_history(persona.root, cfg.path("data") / "identity_history"),
    )
    st = seed.status()
    if st["exists"]:
        print(f"\n身份母本：{st['chars']} 字 · 七节已写 {st['filled']}/{st['total']}")
    else:
        print("\n身份母本：未创建（控制台「身份」页可写，落到 raw/identity.md）")

    stats = memory.stats()
    print(f"\n记忆库：{stats['messages']} 条消息 / {stats['sessions']} 个会话 / {stats['users']} 个用户")

    # 库契约校验
    from workshop.lint import run as lint_run

    r = lint_run(persona.root)
    contract = [it for it in r["items"] if it["code"].startswith("contract")]
    if contract:
        print("\n库契约：")
        for it in contract:
            print(f"  {(it['level']).upper()} {it['page']} — {it['why']}")
    else:
        print("库契约： ✓ 符合 v1")

    print("\n归档器：", "可用" if mem_ingest is not None else f"不可用（{_INGEST_IMPORT_ERROR}）")

    # 模型连通性：试聊与归纳都靠它，配错 Key 时早说比在页面上猜好
    print("\n模型连通性：", end="")
    if not test_llm:
        print("跳过（--no-llm）")
    elif not cfg.llm.resolve_key():
        print(f"✗ 缺少 Key（环境变量 {cfg.llm.api_key_env}）")
    else:
        client = LLMClient(cfg.llm)
        try:
            # 预算别给太小：推理型模型会先把 token 花在思考上，
            # 16 个 token 连一句正文都轮不到，会误报成"模型不可用"。
            reply = asyncio.run(
                client.complete("你是连通性测试。", "只回复两个字：正常", max_tokens=256)
            )
            print(f"✓ {cfg.llm.model} → {reply.strip()[:40]}")
        except Exception as exc:  # noqa: BLE001
            print(f"✗ {exc}")

    print(f"版本：人格工坊 {__version__}")
    return 0


async def _run(
    cfg: config.Config,
    port: int | None = None,
    open_browser: bool = True,
    lock_path: Path | None = None,
) -> bool:
    """跑起来就一直挂着，返回 False 表示控制台压根没起来。"""
    persona_dir = Path(cfg.bot.persona_dir)
    library.apply_prefs(cfg, persona_dir)
    persona = Persona(persona_dir)
    memory = Memory(library.resolve_db(persona_dir, cfg.path(cfg.memory.db_path)))
    llm = LLMClient(cfg.llm)

    # ---------- 试聊：不经过任何通道，回复直接落在控制台聊天页 ----------
    async def sender(out) -> None:  # 试聊不发消息
        return None

    agent = Agent(cfg, persona, memory, llm, sender)

    # ---------- 归档回调（与 run.py 同源，去掉通道部分） ----------
    ingest_lock = asyncio.Lock()
    ingest_manual_task: asyncio.Task | None = None
    ingest_last: dict = {"at": 0.0, "rounds": 0, "events": 0, "note": ""}

    def _ingest_busy() -> bool:
        return ingest_lock.locked() or (
            ingest_manual_task is not None and not ingest_manual_task.done()
        )

    def _stats_lines() -> list[str]:
        try:
            path = persona_dir / "memory" / "ingest_stats.jsonl"
            return path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return []

    def _written_since(before: list[str]) -> int:
        total = 0
        for line in _stats_lines()[len(before):]:
            try:
                total += int(json.loads(line).get("written") or 0)
            except (ValueError, TypeError):
                continue
        return total

    def live_cfg() -> config.Config:
        cur = ctx.get("cfg") if ctx else None
        return cur if isinstance(cur, config.Config) else cfg

    async def ingest_once() -> dict:
        if mem_ingest is None:
            msg = f"归档器不可用：{_INGEST_IMPORT_ERROR}"
            ingest_last["note"] = msg
            return {"ok": False, "error": msg}
        cur = live_cfg()
        if not cur.memory.wiki_enabled:
            return {"ok": False, "error": "记忆层没开（memory.wiki_enabled = false）"}
        if ingest_lock.locked():
            return {"ok": False, "error": "上一轮还没跑完"}
        async with ingest_lock:
            src = cur.source_path or (cur.root / "config.toml")
            stats_before = _stats_lines()
            try:
                code = await mem_ingest.run(
                    mem_ingest.make_args(
                        persona=str(persona_dir),
                        config=str(src),
                        allow_new=cur.memory.ingest_allow_new,
                        rebuild_hub=True,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                log.get("ingest").error("归档失败：%s", exc)
                ingest_last["note"] = f"失败：{exc}"
                return {"ok": False, "error": str(exc)}
        ingest_last.update(
            at=time.time(),
            rounds=ingest_last["rounds"] + 1,
            events=_written_since(stats_before),
            note="完成",
        )
        log.get("ingest").info("归档完成：%s", persona_dir.name)
        return {"ok": code == 0}

    async def ingest_loop() -> None:
        await asyncio.sleep(120)
        while True:
            hours = max(0.0, float(live_cfg().memory.ingest_interval_hours or 0))
            if hours <= 0:
                await asyncio.sleep(300)
                continue
            try:
                await ingest_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                log.get("ingest").warning("归档任务异常：%s", exc)
            deadline = time.time() + hours * 3600
            while time.time() < deadline:
                await asyncio.sleep(min(60.0, max(1.0, deadline - time.time())))
                if float(live_cfg().memory.ingest_interval_hours or 0) != hours:
                    break

    async def ingest_status() -> dict:
        cur = live_cfg()
        return {
            "available": mem_ingest is not None,
            "enabled": bool(cur.memory.wiki_enabled and cur.memory.ingest_interval_hours > 0),
            "interval_hours": float(cur.memory.ingest_interval_hours or 0),
            "allow_new": bool(cur.memory.ingest_allow_new),
            "running": _ingest_busy(),
            "persona_id": persona_dir.name,
            **ingest_last,
        }

    async def ingest_start() -> dict:
        nonlocal ingest_manual_task
        if mem_ingest is None:
            return {"ok": False, "error": f"归档器不可用：{_INGEST_IMPORT_ERROR}"}
        if _ingest_busy():
            return {"ok": False, "error": "上一轮还没跑完，等它结束"}
        task = asyncio.create_task(ingest_once(), name="ingest-manual")
        ingest_manual_task = task

        def _done(t: asyncio.Task) -> None:
            nonlocal ingest_manual_task
            if ingest_manual_task is t:
                ingest_manual_task = None
            if not t.cancelled() and t.exception() is not None:
                log.get("ingest").error("手动归档异常：%s", t.exception())

        task.add_done_callback(_done)
        return {"ok": True}

    # ---------- 控制台 ----------
    webui_url = ""
    raw: dict = {}
    cfg_path = cfg.source_path or (cfg.root / "config.toml")
    if cfg_path.is_file():
        try:
            with cfg_path.open("rb") as fh:
                raw = tomllib.load(fh)
        except (OSError, tomllib.TOMLDecodeError) as exc:
            log.get("webui").warning("读取配置文件失败，面板保存将不可用：%s", exc)
    studio = Studio(cfg, persona, llm) if cfg.sim.enabled else None
    distiller = Distiller(cfg, studio, llm) if cfg.distill.enabled else None
    ctx: dict = {
        "loop": asyncio.get_running_loop(),
        "cfg": cfg,
        "library": library.Library(cfg.path(library.DEFAULT_INDEX_REL)),
        "persona_id": persona_dir.name,
        "memory_db": memory.db_path,
        "raw_config": raw,
        "cfg_path": cfg_path,
        "agent": agent,
        "persona": persona,
        "memory": memory,
        "adapter": None,  # 工坊不接通道
        "sim": studio,
        "distill": distiller,
        "ingest": {
            "run": ingest_once,
            "start": ingest_start,
            "status": ingest_status,
        },
        "started": time.time(),
        "version": __version__,
    }
    try:
        webui_url, _ = webui.start(ctx, cfg.webui.host, port or cfg.webui.port)
    except RuntimeError as exc:
        # 别让窗口一闪而过：返回码非零，启动器会停下来让人看清原因
        print(f"\n控制台启动失败：{exc}")
        print("端口被别的程序占着。换个端口：python run.py --port 8900")
        memory.close()
        return False

    # 把实际端口记进锁文件：下次重复启动时能直接把浏览器指到这里
    if lock_path is not None:
        from urllib.parse import urlparse

        try:
            remember_port(lock_path, urlparse(webui_url).port or 0)
        except ValueError:
            pass

    print("\n┌" + "─" * 66)
    print(f"│ 人格工坊 {__version__}")
    print(f"│ 控制台  {webui_url}")
    print(f"│ 模型    {cfg.llm.model} @ {cfg.llm.base_url}"
          + ("" if llm.ready else "（缺 API Key，试聊不可用，进控制台「模型」页填）"))
    print(f"│ 人格库  {persona.root}")
    print(f"│ 记忆库  {memory.db_path}")
    print("└" + "─" * 66)

    ingest_task: asyncio.Task | None = None
    if cfg.memory.wiki_enabled and float(cfg.memory.ingest_interval_hours or 0) > 0:
        if mem_ingest is None:
            print(f"· 定时归档未启动：归档器加载失败（{_INGEST_IMPORT_ERROR}）")
        else:
            ingest_task = asyncio.create_task(ingest_loop(), name="ingest")
            print(f"· 记忆归档：每 {cfg.memory.ingest_interval_hours:g} 小时一次")

    if open_browser:
        import webbrowser

        if not webbrowser.open(webui_url):
            print(f"\n· 浏览器没起来，手动复制这个地址：{webui_url}")

    try:
        while True:
            await asyncio.sleep(3600)
    except asyncio.CancelledError:
        pass
    finally:
        if ingest_task is not None:
            ingest_task.cancel()
        memory.close()
    return True


def main() -> int:
    ap = argparse.ArgumentParser(
        prog="run.py",
        description="人格工坊：库 + 素材台 + 蒸馏台 + 试聊（通道无关）",
    )
    ap.add_argument("-c", "--config", help="配置文件路径（默认找 config.toml）")
    ap.add_argument("--no-open", action="store_true", help="不自动打开浏览器")
    ap.add_argument("--port", type=int, help="控制台端口（默认 8770）")
    ap.add_argument("--check", action="store_true", help="只自检，不启动")
    ap.add_argument("--debug", action="store_true", help="输出调试日志")
    ap.add_argument("--no-llm", action="store_true", help="自检时不探测模型（省一次调用）")
    args = ap.parse_args()

    try:
        cfg = config.load(args.config)
    except FileNotFoundError as exc:
        print(f"错误：{exc}")
        print("提示：复制 config.example.toml 为 config.toml 后填写。")
        return 2

    log_path = cfg.path(cfg.bot.log_file)
    log.setup("DEBUG" if args.debug else cfg.bot.log_level, log_path)

    if not cfg.bot.persona_dir:
        print("错误：config.toml 里 [bot] persona_dir 没填（人格库路径）")
        return 2

    library.apply_prefs(cfg, Path(cfg.bot.persona_dir))

    # 首次启动：persona_dir 指向的目录还不存在，按模板生成一套空骨架并登记。
    # 不建的话新人 clone 下来是一屏 lint error（README 承诺过会自动建）。
    # create_persona 对"已存在且非空"的目录直接拒绝，不会覆盖任何人的库。
    persona_root = Path(cfg.bot.persona_dir)
    if not persona_root.exists():
        try:
            entry = library.create_persona(persona_root)
            library.Library(cfg.path(library.DEFAULT_INDEX_REL)).register(
                entry.path, name=entry.name, summary=entry.summary
            )
            print(f"  首次启动：已创建空人格库「{entry.name}」")
            print("  控制台打开后，从「身份」页开始填。")
        except (ValueError, OSError) as exc:
            print(f"  自动建库没成功：{exc}")
            print("  可以在控制台里手动「新建人格库」。")

    # 单实例：工坊与通道器可以同时在跑，但工坊自己只能有一个
    lock_path = cfg.path(cfg.memory.db_path).parent / "gongfang.lock"
    ok, holder = acquire_instance_lock(lock_path)
    if not ok:
        _pid, run_port = _lock_state(lock_path)
        holder_pid = int(holder) if holder.isdigit() else 0
        who = f"PID {holder}" if holder else "另一个进程"
        print(f"\n已经有一个工坊实例在运行（{who}），本次不再启动。")

        # 锁里记的端口先验一下：端口能连上不等于对面是工坊，
        # 把浏览器指到一个陌生服务上，用户看到的就是"打不开"。
        if run_port and not _is_workshop(cfg.webui.host, run_port):
            print(f"  锁里记的 {run_port} 上不是工坊控制台（可能被别的程序占了），重新找……")
            run_port = 0

        if not run_port:
            print("  正在找它在听哪个端口……")
            log_path = cfg.path(cfg.bot.log_file)
            # 三级兜底：日志（最可靠）→ netstat（环境相关）→ 扫端口（最慢但最稳）
            hits = [p for p in ports_from_log(log_path) if _is_workshop(cfg.webui.host, p)]
            if not hits:
                hits = [p for p in _ports_of_pid(holder_pid) if _is_workshop(cfg.webui.host, p)]
            if not hits:
                # 拿不到（旧版实例、非 Windows、或 pid 已失效）就扫配置端口附近
                hits = find_consoles(cfg.webui.host, cfg.webui.port)
            run_port = hits[0] if hits else 0
            if run_port and holder_pid:
                # 帮旧实例把端口补进锁文件，下次再点启动就不用再找了
                remember_port(lock_path, run_port, pid=holder_pid)
            if len(hits) > 1:
                print("  找到不止一个，下面是全部：")
                for p in hits:
                    print(f"    http://127.0.0.1:{p}/")
        if run_port:
            url = f"http://127.0.0.1:{run_port}/"
            print(f"  控制台就在：{url}")
            if not args.no_open:
                import webbrowser

                print("  正在用浏览器打开它……")
                if not webbrowser.open(url):
                    print(f"  浏览器没起来，手动复制这个地址：{url}")
            print("  想重开就先关掉那个实例（关掉它的窗口，或在任务管理器里结束那个 python）。")
        else:
            print("  但没找到它在监听哪个端口，可能已经卡住或退出了。")
            print("  把它关掉再启动一次即可 —— 下次启动会自动记住端口。")
        return 3

    if args.check:
        persona = Persona(Path(cfg.bot.persona_dir))
        memory = Memory(library.resolve_db(Path(cfg.bot.persona_dir), cfg.path(cfg.memory.db_path)))
        try:
            return _selfcheck(cfg, persona, memory, test_llm=not args.no_llm)
        finally:
            memory.close()
            release_instance_lock(lock_path)

    started = True
    try:
        started = asyncio.run(
            _run(cfg, port=args.port, open_browser=not args.no_open, lock_path=lock_path)
        )
    except KeyboardInterrupt:
        print("\n收到中断，退出。")
    finally:
        # 退出就把锁摘掉，别留个"有实例在跑"的假消息给下一次启动
        release_instance_lock(lock_path)
    return 0 if started else 4


if __name__ == "__main__":
    raise SystemExit(main())
