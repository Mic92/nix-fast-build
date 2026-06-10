"""Terminal/CI output helpers shared by the build log renderers."""

import os
import re
import unicodedata

# ANSI escape sequences: CSI, OSC (incl. unterminated), other ESC forms.
ANSI_RE = re.compile(
    r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07\x1b]*(?:\x07|\x1b\\)?|[@-Z\\-_])"
)
# Mega-lines (minified JS etc.) choke terminals and CI web UIs.
MAX_LINE_LEN = 4096
SGR_RESET = "\x1b[0m"
# ESC that does not start an SGR sequence (used after ANSI_RE filtering).
# Must accept exactly what repl() in sanitize_line keeps: a CSI sequence
# (incl. intermediate bytes) with final byte m.
_STRAY_ESC_RE = re.compile(r"\x1b(?!\[[0-?]*[ -/]*m)")
_TRAILING_PARTIAL_SGR_RE = re.compile(r"\x1b[^m]*$")


def sanitize_line(s: str) -> str:
    """Make one captured log line safe to re-emit.

    - emulate carriage-return overwrite (progress bars: keep final state)
    - keep SGR color sequences, strip everything else ANSI (cursor moves,
      clear-screen, OSC title changes -- a build log must not be able to
      take over our terminal)
    - if we kept any SGR, append a reset so unbalanced color can't bleed
      into subsequent lines
    - expand tabs, drop other control chars
    - cap length

    CI command injection: GitHub/Forgejo only interpret ::commands:: at
    the start of a line. Renderers prefix every build line with "attr> ",
    and this function guarantees no CR/LF survives, so build output can
    never fabricate a line start to smuggle ::endgroup::/::add-mask::.
    """
    if "\r" in s:
        s = next((p for p in reversed(s.split("\r")) if p), "")
    kept_sgr = False

    def repl(m: re.Match[str]) -> str:
        nonlocal kept_sgr
        seq = m.group(0)
        if seq.startswith("\x1b[") and seq.endswith("m"):
            kept_sgr = True
            return seq
        return ""

    s = ANSI_RE.sub(repl, s)
    s = s.expandtabs(4)
    s = "".join(ch for ch in s if ch == "\x1b" or not ch < " ")
    # After the substitution, every legitimate ESC starts a kept SGR
    # (\x1b[...m). A stray or partial ESC would pair with our own output
    # at display time, so drop it.
    s = _STRAY_ESC_RE.sub("", s)
    if len(s) > MAX_LINE_LEN:
        # Don't cut an SGR in half: an unterminated CSI makes the
        # terminal swallow the text that follows.
        s = _TRAILING_PARTIAL_SGR_RE.sub("", s[:MAX_LINE_LEN])
        s += " …[line truncated]"
    if kept_sgr:
        s += SGR_RESET
    return s


def char_width(ch: str) -> int:
    """Display cells of one char: CJK/emoji are 2, combining marks 0."""
    if unicodedata.combining(ch):
        return 0
    return 2 if unicodedata.east_asian_width(ch) in "WF" else 1


def clip_ansi(s: str, width: int) -> str:
    """Clip to display cells, passing ANSI sequences through unbroken."""
    out: list[str] = []
    cells = 0
    i = 0
    while i < len(s):
        m = ANSI_RE.match(s, i)
        if m:
            out.append(m.group(0))
            i = m.end()
            continue
        w = char_width(s[i])
        if cells + w > width:
            break
        out.append(s[i])
        cells += w
        i += 1
    return "".join(out)


def subseq_match(query: str, candidate: str) -> bool:
    """fzf-style subsequence match, case-insensitive."""
    it = iter(candidate.lower())
    return all(ch in it for ch in query.lower())


def trunc_middle(s: str, n: int) -> str:
    if len(s) <= n:
        return s
    half = (n - 1) // 2
    return s[:half] + "…" + s[-(n - 1 - half) :]


def want_color(isatty: bool) -> bool:
    """Color decision: NO_COLOR > CLICOLOR_FORCE/FORCE_COLOR > isatty.

    Actions CI counts as color-capable: its log viewer renders ANSI,
    but the job's stderr is a pipe, so isatty alone would disable it.
    """
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("CLICOLOR_FORCE") or os.environ.get("FORCE_COLOR"):
        return True
    return isatty or fold_markers()


def fold_markers() -> bool:
    """Whether to emit ::group:: fold markers for collapsible log sections.

    GitHub Actions, Forgejo Actions and Gitea Actions all set
    GITHUB_ACTIONS=true and are indistinguishable from inside a job.
    GitHub and Forgejo (v7+) render the markers as folds; Gitea prints
    them as text, which is harmless.
    """
    return os.environ.get("GITHUB_ACTIONS") == "true"
