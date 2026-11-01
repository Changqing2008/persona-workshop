"""一键归纳的提示词拼装。

单独一模块，只为让 `ingest.py` 专心做校验和落盘 —— 提示词改起来是常态，
校验逻辑不是。这里**只读**库，不写任何文件。

拼法：系统提示（角色 + 铁律 + 本类素材的固定映射表 + 补丁 schema）
    + 用户消息（成品头 + 素材全文 + 可用页现存内容）。
"""

from __future__ import annotations

from pathlib import Path

from . import ingest_rules as R

PAGE_HINT = {
    "identity.md": "身份背景：名字 / 设定 / 关系三档事实 / 想要与怕 / 经历脉络",
    "traits.md": "核心性格：稳定主特质 + 情境触发特质（条件 → 表现）",
    "speech.md": "语言风格：语气语域 / 口癖 / 句式节奏 / 消息体量 / 标点 / 称呼 / 禁忌词",
    "reactions.md": "情境反应：`### 情境：<刺激> / <场景>` 下写条件 → 回应 → 不做",
    "values.md": "价值观：母本「价值观与立场」一节的搬运与细化，不从对话里推新立场",
    "relationships.md": "关系三档：只写六个旋钮（称呼 / 直接度 / 话量 / 解释不解释 / 主动追问 / 情绪露出）",
    "boundaries.md": "边界红线：不做的事 / 不聊的话题 / 一致性硬约束",
    "exemplars.md": "范例语料：按情境分节，一句原话一条，标 source 与 confidence",
}

PRIORITY = {
    R.KIND_REACTION: ("reactions.md", "speech.md", "relationships.md", "exemplars.md"),
    R.KIND_DIALOG: (
        "exemplars.md", "reactions.md", "speech.md", "traits.md",
        "values.md", "relationships.md", "identity.md", "boundaries.md",
    ),
}

PAGES_BUDGET = 20000
"""提示词里现有页内容的总字数上限。超了后面的页只给目录。

反应卡四个页全量约 24K 字（reactions 12K + speech 5.5K + relationships 6.5K），
在 128K 上下文的模型里不算贵；但对话稿要看的页多，得有个闸。
"""

MATERIAL_LIMIT = 12000
"""素材正文上限。野生素材很少有超过这个长度的，真超了截断并告知模型。"""


SYSTEM = """你是「归纳AI」。任务是把一份 raw 素材收进人格库的 wiki/ 页。

你不是在写人物小传，是在写「这个人怎么说话、怎么反应」。wiki 存的是**行为**，不是百科事实。

## 铁律（违反即作废，程序会逐条拦）

1. **只准追加**。没有删除、没有替换。schema 里只有 append / merge / conflict / skip 四种动作。
   已有条目哪怕你觉得写得不好，也不许覆盖；只能是「在它下面补一条」或「新开一条并列」。
2. **每条必须带 quote**：从素材原文里**逐字复制**一小段，一个标点都不许改。
   正文引文至少 8 字；抄成品头的某一行也算（`话量：少（一句一条）` 这种短句可以，
   但必须整行原样，改了标点顺序就作废）。抄不出来，说明这条是你编的，那就不要写。
   程序会拿 quote 回素材里搜，搜不到整条作废。
3. **text 只写条目正文**，可以是多行 `- ` 项目符号。不许写 `#` 开头的标题 ——
   页名与节标题由程序加，你越界就是改结构。
4. **不许性格断言**：不写「她总是 / 一向 / 从来 / 本质 / 其实很 / 这人 / 天生 / 根本」。
   要写就写行为：「被追问就一句『不必』，不再解释」。
5. **素材没说的不补**。不猜动机、不补背景、不美化。宁可这一份只出 1 条，也不许凑 5 条。
6. **抽象标签必须跟着行为写**。整条读下来，读者应该能照着做出来，而不是只知道她很"冷"。
7. **text 里只写结论**，不写过程说明。「素材这一栏没记，不补」「与某条相抵，等人工裁」
   这类话放进 `why`（那是给人看的），写进 wiki 的每一句都会变成她的人设。
   素材缺的字段**跳过就完了**，不用报告你跳过了什么。

## 输出格式（只输出 JSON，不要解释、不要 markdown 代码块以外的话）

```json
{
  "items": [
    {
      "op": "append",
      "page": "reactions.md",
      "section": "高频情境反应",
      "title": "情境：被关心 / 对方把她手里的事接过去",
      "text": "- 条件：对方主动替她把事接走，不问她要不要\\n- 回应：先愣一下，然后一句「我自己来」，不再解释\\n- 不做：不说谢谢，不解释为什么不用",
      "quote": "从素材里逐字抄的一段",
      "confidence": "medium",
      "why": "一句话说清这条为什么该进这页"
    }
  ],
  "note": "整份都收不出东西时，在这里写原因；有 items 就留空"
}
```

字段规则：

- `op`：
  - `append` = 新条目（最常见）
  - `merge` = 并进页里已有的某条下面，`title` 必须**原样**给已有的标题
  - `conflict` = 与页里既有说法矛盾。两说并存待人工裁，不许覆盖旧的
  - `skip` = 素材里这块不值得收（只写 `why`，不写 text）
- `page`：只能填下面列出的页，页名一字不差
- `section`：`##` 级标题，**优先从「现有页面」里原样抄一个已有的**。只有两个例外：
  - 对话稿往 `exemplars.md` 收原话：按情境建新节，写成 `## <情境>（<档位>）`，与页里已有节同构
  - 反应样式稿往 `exemplars.md` 收范例：固定 `## 行为范例`（那一节由你建）
  其余自己编节名一律作废 —— 页的结构是人定的，不是你定的
- `title`：可空。填了就是 `###` 级小标题，同一节里同名的会并进去（新情境靠它区分）
- `confidence`：`high`（多处证据一致或人工原话）/ `medium`（单处但有代表性）/ `low`（推断、孤立）。
  **程序会按素材类型压天花板**，你自报 high 也可能被降成 medium，别为了显得确定就写 high
- `why`：给人工看的一句话理由，不进 wiki
"""


REACTION_RULES = """
## 这份素材是「反应样式稿」

它没有对白、没有对象、没有具体场景。它只说「面对这类刺激，她怎么回」。

**固定映射，不许自由归页**：

| 进哪页 | 写什么 |
|---|---|
| `reactions.md` | `刺激` + `场景` + `情绪` + `回应` + `露出` + `不做`，合成一条条件 → 回应。**一律写进页里已有的 `## 高频情境反应`**，新情境靠 `title`（`### 情境：<刺激> / <场景>`）区分 |
| `speech.md` | `话量 / 句长` → 句式节奏，`语言` → 口癖与标点 |
| `relationships.md` | **仅当 `档位` 不是 `—`**：那一节的六个旋钮（称呼 / 直接度 / 话量 / 解释不解释 / 主动追问 / 情绪露出）。**每条 `- ` 行都必须以旋钮名开头**，写成散文会被拦 |
| `exemplars.md` | **仅当成品头有 `范例：已核`**，且只进 `## 行为范例`，写成「条件 → 回应」，不抄句子 |

**三条禁令**：

1. 不许写进 `identity.md` / `boundaries.md`。
2. 不许出现专名、称谓、职业、地名、具体事件。写「被连续追问」，不写「被长辈在书房追问」。
   「不喊称呼」是规则，可以写；「喊他老师」是称谓，会被拦。
3. 与页里既有条目或真人原话冲突时，以既有为准，用 `op: conflict`，不许覆盖。

`档位` 只能写三个固定词（拥有者 / 熟人 / 陌生人），不能写成「她师父」。
"""

DIALOG_RULES = """
## 这份素材是「对话稿」

它是一段真实或接近真实的对话，可能有成品头（`用途 / 分类 / 情境 / 张力 / 对方 / 构成 / 来源 / 批注`）。

- **先读成品头再读正文**：头决定「往哪页、算不算硬证据」，正文决定「怎么写」。
- `分类` 决定用途；两处打架时**以分类为准**。头里没有的字段不报错、不反推。
- `情境` 首段是关系档（拥有者 / 熟人 / 陌生人）。**首段是 `—` 表示还没归过档**：
  这一段不进 `relationships.md`，不要替它贴一档。
- `构成` 里 `人工原话 N 句` 才算 hard 证据；`原文 N 句`（小说）最高 medium；
  只有 `AI 采纳` 的当补充语料，不写成硬人设；`批注` 里有「推定」按 low 收。
- **范例语料优先**：最像本人的原话进 `exemplars.md`，一句一条，标 source。
  这比任何"她很毒舌"的描述都值钱。
- `exemplars.md` 的节名按情境起（`## <情境>（<档位>）`，与页里已有节同构）。
  **不要用 `## 行为范例`** —— 那是反应样式稿的补位区，写「条件 → 回应」不抄句子；
  对话稿的原话要一句一条、原样引用。
  建节之前先把页里现有的 `##` 节名过一遍：只要有一节的措辞能容纳这条原话，就原样抄它。
  宁可并进一个宽一点的节，也不要另起一节 —— 近义节攒多了只能靠人手工合并，机器识别不了。
- `reactions.md` 照页里已有条目的字段名写（`- 典型反应：` / `- 情绪强度：` / `- 证据摘录：` / `- 来源：`），
  不要用反应样式稿那套「条件 / 回应 / 不做」—— 两套字段混在一页里，人一眼就看出来。
- 同一份素材里矛盾的两面，不要合成一句，写成「在 X 情境 → Y 表现」的条件规则。
- `values.md` 只搬运母本 `raw/identity.md` 的「价值观与立场」一节，**不新增立场**。
  母本没写的宁可不写；对话里与母本相反的，用 `op: conflict` 记一条并写清出处，不许覆盖。
"""


def _read(root: Path, page: str) -> str:
    path = root / "wiki" / page
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _headings(text: str) -> str:
    """只给目录：`## 节` 与 `### 子节`。用于预算不够时的降级。"""
    rows = []
    for line in text.splitlines():
        s = line.rstrip()
        if s.startswith("## ") or s.startswith("### "):
            rows.append(s)
    return "\n".join(rows)


def _slice_section(text: str, heading: str) -> str:
    """取某个 `## 节` 的正文（含标题行）。取不到返回空串。"""
    rows = text.splitlines()
    target = heading.strip()
    start = None
    for i, line in enumerate(rows):
        if line.rstrip() == target:
            start = i
            break
    if start is None:
        return ""
    end = len(rows)
    for i in range(start + 1, len(rows)):
        if rows[i].startswith("## "):
            end = i
            break
    return "\n".join(rows[start:end]).strip()


def pages_block(root: Path, asset_kind: str, level: str, budget: int = PAGES_BUDGET) -> str:
    """拼「现有页面」段。按类别优先级塞，塞不下就降级成目录。"""
    order = PRIORITY.get(asset_kind, PRIORITY[R.KIND_DIALOG])
    allowed = R.allowed_pages(asset_kind)
    parts: list[str] = []
    used = 0
    for page in order:
        if page not in allowed:
            continue
        text = _read(root, page)
        if not text.strip():
            continue
        body = text
        # 关系页只给这一档：三档一起给，模型会串档（agents.md 第八节）
        if page == "relationships.md" and level in R.LEVELS:
            sec = ""
            for heading in R.REL_SECTION_ALIASES.get(level, ()):
                sec = _slice_section(text, heading)
                if sec:
                    break
            # 取不到这一档时给空壳，绝不退回整页：整页 = 三档一起进提示词
            body = sec or f"{R.REL_SECTIONS[level]}\n（这一档还没写）"
        hint = PAGE_HINT.get(page, "")
        if used + len(body) > budget:
            body = _headings(text)
            note = "（预算不够，这里只给目录；要写就写已有的节名）"
        else:
            note = ""
        used += len(body)
        parts.append(f"### {page} — {hint}\n{note}\n{body.strip()}")
    return "\n\n".join(parts)


def render_head(head: dict[str, str]) -> str:
    if not head:
        return "（无成品头）"
    return "\n".join(f"{k}：{v}" for k, v in head.items() if str(v).strip())


def build_user(root: Path, asset) -> str:
    """用户消息：素材 + 现有页面。asset 是 `ingest.Asset`（鸭子类型，避免循环 import）。"""
    text = asset.text or ""
    if len(text) > MATERIAL_LIMIT:
        text = text[:MATERIAL_LIMIT] + "\n\n（素材过长已截断）"
    kind_label = R.KIND_LABEL.get(asset.kind, asset.kind)
    allowed = "\n".join(f"- `{p}` — {PAGE_HINT.get(p, '')}" for p in R.allowed_pages(asset.kind))
    return (
        f"== 素材：{asset.plain} ==\n"
        f"类型：{kind_label}｜关系档：{asset.level}｜置信天花板：{asset.cap}\n"
        f"成品头：\n{render_head(asset.head)}\n\n"
        f"--- 素材原文 ---\n{text.strip()}\n\n"
        f"--- 可写的页（只能填这些） ---\n{allowed}\n\n"
        f"--- 现有页面内容 ---\n{pages_block(root, asset.kind, asset.level)}\n"
    )


def build_messages(root: Path, asset) -> list[dict[str, str]]:
    kind_rules = REACTION_RULES if asset.kind == R.KIND_REACTION else DIALOG_RULES
    return [
        {"role": "system", "content": SYSTEM + "\n" + kind_rules},
        {"role": "user", "content": build_user(root, asset)},
    ]
