"""启动器测试：单实例锁 + 端口定位。

这道闸的价值是一次双击到底能不能把浏览器指到真正在跑的那个控制台。
踩过的坑都在这儿钉住：
- 锁文件只写 PID 的旧版本，第二次启动必须还能找到端口（用户报过两次"打不开"）
- 帮旧实例补记端口时，pid 得写持有者的，不能写成自己
- 端口能连上不代表是工坊，别的程序也可能占着
- 控制台起不来不能返回 0，否则启动器窗口一闪而过，人什么也看不到
"""

from __future__ import annotations

import os
import socket
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import run  # noqa: E402

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


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class _Plain(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        body = b"<html><head><title>other</title></head><body>no</body></html>"
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a) -> None:
        return


class _Workshop(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        body = "<html><head><title>人格工坊 控制台</title></head></html>".encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a) -> None:
        return


def _serve(handler) -> tuple[ThreadingHTTPServer, int]:
    srv = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, srv.server_address[1]


# ---------------------------------------------------------------- 锁文件解析
def test_lock_state(tmp: Path) -> None:
    print("\n锁文件解析")
    lock = tmp / "a.lock"

    lock.write_text("27844\n8799\n", encoding="utf-8")
    check("新版：PID + 端口", run._lock_state(lock) == ("27844", 8799), str(run._lock_state(lock)))

    lock.write_text("27844\n", encoding="utf-8")
    check("旧版只写 PID：端口记 0", run._lock_state(lock) == ("27844", 0), str(run._lock_state(lock)))

    lock.write_text("27844\r\n8799\r\n", encoding="utf-8", newline="")
    check("CRLF 换行也认", run._lock_state(lock) == ("27844", 8799), str(run._lock_state(lock)))

    lock.write_text("", encoding="utf-8")
    check("空文件不炸", run._lock_state(lock) == ("", 0), str(run._lock_state(lock)))

    lock.unlink()
    check("文件不存在不炸", run._lock_state(lock) == ("", 0), str(run._lock_state(lock)))

    lock.write_text("not-a-pid\nxx\n", encoding="utf-8")
    check("垃圾内容当没有", run._lock_state(lock) == ("", 0), str(run._lock_state(lock)))


# ---------------------------------------------------------------- 端口回写
def test_remember(tmp: Path) -> None:
    print("\n端口回写")
    lock = tmp / "b.lock"

    run.remember_port(lock, 8799)
    rows = lock.read_text(encoding="utf-8").split()
    check("默认记自己的 PID", rows == [str(os.getpid()), "8799"], str(rows))

    run.remember_port(lock, 8799, pid=27844)
    rows = lock.read_text(encoding="utf-8").split()
    check("帮别人补记时写持有者 PID", rows == ["27844", "8799"], str(rows))

    run.remember_port(lock, 0, pid="27844")
    check("端口 0 也照写，不抛异常", run._lock_state(lock) == ("27844", 0), str(run._lock_state(lock)))


# ---------------------------------------------------------------- 摘锁
def test_release(tmp: Path) -> None:
    print("\n摘锁只摘自己的")
    lock = tmp / "c.lock"

    lock.write_text(f"{os.getpid()}\n", encoding="utf-8")
    run.release_instance_lock(lock)
    check("自己的锁：删掉", not lock.exists())

    lock.write_text("27844\n8799\n", encoding="utf-8")
    run.release_instance_lock(lock)
    check("别人的锁：留着", lock.exists())

    lock.unlink()
    run.release_instance_lock(lock)
    check("没有锁：不炸", True)


# ---------------------------------------------------------------- 拿锁
def test_acquire(tmp: Path) -> None:
    print("\n单实例锁")
    lock = tmp / "d.lock"

    ok, who = run.acquire_instance_lock(lock)
    check("没有锁：拿到", ok is True, f"{ok} {who}")
    run.release_instance_lock(lock)

    lock.write_text("999999\n", encoding="utf-8")
    ok, _who = run.acquire_instance_lock(lock)
    check("死进程留下的锁：接管", ok is True)
    check("接管后写的是自己的 PID", run._lock_state(lock)[0] == str(os.getpid()), str(run._lock_state(lock)))
    run.release_instance_lock(lock)

    lock.write_text(f"{os.getpid()}\n", encoding="utf-8")
    ok, who = run.acquire_instance_lock(lock)
    check("自己还活着：被挡", ok is False and who == str(os.getpid()), f"{ok} {who}")

    # 进程活着、端口明确拒连 = 上一个实例卡死了，锁是残留的，让新实例接手
    lock.write_text(f"{os.getpid()}\n{_free_port()}\n", encoding="utf-8")
    ok, _who = run.acquire_instance_lock(lock)
    check("活着但端口拒连：当成卡死，接管", ok is True)
    check("接管后写的是自己的 PID", run._lock_state(lock)[0] == str(os.getpid()), str(run._lock_state(lock)))
    run.release_instance_lock(lock)

    lock.write_text(f"{os.getpid()}\n", encoding="utf-8")
    ok, _who = run.acquire_instance_lock(lock)
    check("旧版锁没端口：判不准，宁可不抢", ok is False)
    run.release_instance_lock(lock)


# ---------------------------------------------------------------- 日志取端口
def test_ports_from_log(tmp: Path) -> None:
    print("\n从日志找端口")
    log = tmp / "e.log"
    rows = [
        "2026-10-29 23:02:38,366 INFO  webui        WebUI 已启动：http://127.0.0.1:8780",
        "2026-10-30 12:16:01,607 INFO  webui        WebUI 已启动：http://127.0.0.1:8799",
        "2026-10-30 12:16:53,967 INFO  webui        WebUI 已启动：http://127.0.0.1:8799",
        "2026-10-31 11:52:27,391 INFO  webui        WebUI 已启动：http://127.0.0.1:8780",
    ]
    log.write_text("\n".join(rows) + "\n", encoding="utf-8")
    got = run.ports_from_log(log)
    check("去重且最近的在前", got == [8780, 8799], str(got))

    log.write_text("没有任何端口信息的日志\n随便几行\n", encoding="utf-8")
    check("没有记录：空列表", run.ports_from_log(log) == [])

    log.write_text("", encoding="utf-8")
    check("空日志：空列表", run.ports_from_log(log) == [])

    missing = tmp / "nope.log"
    check("日志不存在：空列表，不炸", run.ports_from_log(missing) == [])

    # 只读末尾 64KB：日志再大也不能整份读进内存
    big = tmp / "big.log"
    big.write_text("填充\n" * 40000 + rows[1] + "\n", encoding="utf-8")
    check("大日志只读尾巴也能找到", run.ports_from_log(big) == [8799], str(run.ports_from_log(big)))

    log.write_text("\n".join(rows + rows + rows), encoding="utf-8")
    check("limit 生效", len(run.ports_from_log(log, limit=1)) == 1)


# ---------------------------------------------------------------- 定位进程端口
def test_ports_of_pid() -> None:
    print("\n按 PID 找端口")
    if os.name != "nt":
        check("非 Windows 直接返回空（交给扫描兜底）", run._ports_of_pid(os.getpid()) == [])
        return
    # netstat 在这个环境里会漏行（实测 TCP 表只给出 76 行），所以不断言"必须找到"，
    # 只保证不炸、不返回脏数据。真正的定位靠日志那一级。
    hits = run._ports_of_pid(os.getpid())
    check("不抛异常，返回整数列表", isinstance(hits, list) and all(isinstance(p, int) for p in hits), str(hits))
    check("不存在的 PID 返回空", run._ports_of_pid(999999) == [], str(run._ports_of_pid(999999)))
    check("PID 0 返回空", run._ports_of_pid(0) == [])


# ---------------------------------------------------------------- 认出工坊
def test_recognise() -> None:
    print("\n端口上是不是工坊")
    srv_w, port_w = _serve(_Workshop)
    srv_p, port_p = _serve(_Plain)
    try:
        check("工坊页面：认出来", run._is_workshop("127.0.0.1", port_w) is True)
        check("别家网页：不认", run._is_workshop("127.0.0.1", port_p) is False)
        check("没人听：不认", run._is_workshop("127.0.0.1", _free_port()) is False)
        check("端口 0：不认", run._is_workshop("127.0.0.1", 0) is False)
    finally:
        srv_w.shutdown()
        srv_p.shutdown()


# ---------------------------------------------------------------- 扫端口
def test_scan() -> None:
    print("\n批量扫端口")
    srv, port = _serve(_Workshop)
    try:
        alive = run.scan_ports("127.0.0.1", list(range(port, port + 8)))
        check("扫到在听的那个", port in alive, str(alive))
        found = run.find_consoles("127.0.0.1", port, span=16)
        check("附近扫出工坊", port in found, str(found))
        check("扫的是空区段：一个都不给", run.scan_ports("127.0.0.1", [_free_port() for _ in range(4)]) == [])
    finally:
        srv.shutdown()

    srv2, port2 = _serve(_Plain)
    try:
        check("在听但不是工坊：不进候选", run.find_consoles("127.0.0.1", port2, span=8) == [])
    finally:
        srv2.shutdown()


# ---------------------------------------------------------------- 退出码
def _main_with(
    argv: list[str],
    lock: tuple[bool, str],
    run_result: bool | None = None,
    lock_port: int = 8799,
    is_workshop=...,
    found: list[int] | None = None,
) -> tuple[int, str]:
    """跑一遍 main()，把锁、网络探测全换成假的。

    真实环境里 8799 上确实有个工坊在跑，不换掉这些探测，测试结论会随
    "用户此刻有没有开着工坊"而变 —— 那种测试等于没测。
    """
    import io
    from contextlib import redirect_stdout

    old_argv = sys.argv
    old_lock = run.acquire_instance_lock
    old_state = run._lock_state
    old_find = run.find_consoles
    old_ports = run._ports_of_pid
    old_logports = run.ports_from_log
    old_ws = run._is_workshop
    old_aret = run.asyncio.run
    sys.argv = ["run.py", *argv]
    run.acquire_instance_lock = lambda p: lock
    run._lock_state = lambda p: ("27844", lock_port)
    hits = [8799] if found is None else list(found)
    run.find_consoles = lambda *a, **k: list(hits)
    run._ports_of_pid = lambda pid: list(hits)
    run.ports_from_log = lambda *a, **k: list(hits)
    if is_workshop is not ...:
        run._is_workshop = is_workshop
    if run_result is not None:
        # 关掉协程，不然会冒 RuntimeWarning: coroutine was never awaited
        run.asyncio.run = lambda coro: (coro.close(), run_result)[1]
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            code = run.main()
    finally:
        sys.argv = old_argv
        run.acquire_instance_lock = old_lock
        run._lock_state = old_state
        run.find_consoles = old_find
        run._ports_of_pid = old_ports
        run.ports_from_log = old_logports
        run._is_workshop = old_ws
        run.asyncio.run = old_aret
    return code, buf.getvalue()


def test_exit_codes() -> None:
    print("\n退出码与提示")
    code, out = _main_with(["--no-open"], (False, "27844"))
    check("已有实例且端口已知：3", code == 3, str(code))
    check("提示里给出能点的地址", "http://127.0.0.1:8799/" in out, out)

    code, out = _main_with(["--no-open"], (False, "27844"), lock_port=0)
    check("锁里没端口：也能找到 8799", "8799" in out, out)
    check("同一种情况仍返回 3", code == 3, str(code))

    code, out = _main_with(["--no-open"], (False, "27844"), is_workshop=lambda h, p: False)
    check("端口被别的程序占：不往那儿指", "正在找它在听哪个端口" in out or "重新找" in out, out)

    code, out = _main_with(["--no-open"], (False, ""), lock_port=0, found=[])
    check("找不到任何控制台：说清楚让它重开", "没找到" in out, out)

    code, _out = _main_with(["--no-open"], (True, ""), run_result=False)
    check("控制台起不来：4（窗口要留住）", code == 4, str(code))

    code, _out = _main_with(["--no-open"], (True, ""), run_result=True)
    check("正常退出：0", code == 0, str(code))


def main() -> int:
    print("=" * 60)
    print("启动器测试")
    print("=" * 60)
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        test_lock_state(tmp)
        test_remember(tmp)
        test_release(tmp)
        test_acquire(tmp)
        test_ports_from_log(tmp)
    test_ports_of_pid()
    test_recognise()
    test_scan()
    test_exit_codes()
    print("\n" + "=" * 60)
    print(f"通过 {PASS} · 失败 {FAIL}")
    print("=" * 60)
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
