#!/usr/bin/env python3
"""Pre-publication check for this repository.

Fails if anything that must not be public is present: private paths, host or
network identifiers, credentials, workflow internals, oversized files, a stray
virtualenv, a missing licence file, or a broken link in the README -- and, since
the 2026-09 publication pass, three content rules: no reference to the
implementation, paper or repository the method came from (OTHER-IMPL), no
affiliation, programme or premises (AFFILIATION), no ownership assertion
or author name (OWNERSHIP), each applied to file names as well as contents, and
no URL to another party's repository (OTHER-REPO-URL) beyond the vendor install
sources listed in THIRD_PARTY_NOTICES.md. Verbatim third-party licence texts under
third_party_licenses/ are exempt from the content rules by resolved path.

    python tools/publish_check.py [--root .] [-v]

Exit code 0 means clean.

NOTE ON THE PATTERNS. Every forbidden token is assembled from fragments at
runtime and is never printed. This file ships publicly; spelling out the words
it forbids would leak exactly what it exists to keep out. Failures name the
file, the line and a rule id -- never the matched string.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

MAX_FILE_BYTES = 10 * 1024 * 1024

TEXT_SUFFIXES = {
    ".py", ".md", ".txt", ".toml", ".cfg", ".ini", ".json", ".csv", ".yaml",
    ".yml", ".sh", ".xml", ".example", ".gitignore",
}
SKIP_DIRS = {".git", ".pytest_cache", "node_modules"}


def _j(*parts: str) -> str:
    return "".join(parts)


def _rules() -> list[tuple[str, re.Pattern[str], str]]:
    """(rule id, compiled pattern, human description). Needles are assembled."""
    home = _j("/ho", "me/")
    return [
        # --- absolute home paths, any user ---
        ("HOME-PATH", re.compile(re.escape(home) + r"[A-Za-z0-9._-]+"),
         "an absolute home path"),

        # --- private workspace / sibling-checkout / umbrella-project names ---
        ("PRIVATE-WORKSPACE", re.compile(
            _j("Eli", "The", "Intern"), re.I), "a private workspace name"),
        ("PRIVATE-PROJECT", re.compile(
            r"\b" + _j("PRI", "SM") + r"([ _-]?" + _j("LE", "NS") + r"|[ _-]MASTER)?\b",
            re.I), "a private umbrella-project name"),
        ("PRIVATE-REPO", re.compile(
            r"\b(" + _j("Video", "2Real") + r"|" + _j("Loop", "daed5") + r"|"
            + _j("pri", "sm", "_pipe", "line") + r"|" + _j("Project", "-PRI", "SM")
            + r"|" + _j("MONO", "LITH ") + r")", re.I),
         "another private repository"),
        ("SIBLING-CHECKOUT", re.compile(
            r"\.\./(" + _j("REST", "3D") + r"|" + _j("sam-3d-", "objects")
            + r"|" + _j("Robo", "JuDo") + r"|" + _j("mj", "lab") + r")\b", re.I),
         "a sibling-checkout relative path"),

        # --- employment identifiers ---
        ("EMPLOYER", re.compile(
            r"\b(" + _j("Info", "sys") + r"|" + _j("In", "Step") + r"|"
            + _j("ARC ", "Humanoid") + r")\b", re.I), "an affiliation identifier"),
        ("PERSON", re.compile(r"\b" + _j("Rajavard", "han") + r"\b", re.I),
         "a third party's name"),

        # --- private machines ---
        ("HOSTNAME", re.compile(
            r"\b(" + _j("arc", "05") + r"|" + _j("arc", "06") + r"|"
            + _j("inspi", "ron", "-76", "30") + r"|" + _j("ti", "tan", "-50", "90") + r")\b", re.I),
         "a private hostname"),

        # --- network / hardware addresses ---
        ("TAILSCALE-IP", re.compile(r"\b100\.(8[0-9]|9[0-9]|1[0-2][0-9])\.\d{1,3}\.\d{1,3}\b"),
         "a private mesh-network address"),
        ("LAN-IP", re.compile(r"\b(172\.(1[6-9]|2\d|3[01])|10)\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"),
         "a private LAN address"),
        ("MAC-ADDR", re.compile(r"\b(?:[0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}\b"),
         "a MAC or BLE address"),

        # --- credentials ---
        ("API-KEY", re.compile(r"\bAIza[0-9A-Za-z_\-]{30,}\b"),
         "what looks like a Google API key"),
        ("OPENAI-KEY", re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"),
         "what looks like a secret key"),
        ("HF-TOKEN", re.compile(r"\bhf_[A-Za-z0-9]{30,}\b"),
         "what looks like a Hugging Face token"),
        ("ASSIGNED-SECRET", re.compile(
            r"(?i)\b(api[_-]?key|secret|password|passwd|token)\b\s*[:=]\s*"
            r"[\"']?[A-Za-z0-9_\-]{16,}[\"']?"),
         "a credential assigned to a literal value"),

        # --- the implementation / paper the method came from (R1) ---
        ("OTHER-IMPL", re.compile(
            r"\b" + _j("rest", r"[\s_-]?", "3d") + r"\b|" + _j("ma et", r"\s+", "al") + r"\b|"
            + _j("2605", r"\.", "30338") + r"|" + _j("ar", "xiv") + r"|"
            + r"\b" + _j("3e06", "390") + r"\b|"
            + r"\b(reference|upstream|external|published|prior|original) "
            + r"(implementation|run|checkout|reference|method|paper|baseline|pipeline|exporter)s?\b|"
            + r"behaviou?ral baseline|"
            + r"\bthe other (reconstruction|implementation|pipeline|exporter)\b|"
            + r"\bseparately reconstructed\b|\blike for like\b|"
            + _j("CC", r"[ -]?", "BY", r"[ -]?", "NC") + r"|" + _j("gjk", r"\.py") + r" was a port", re.I),
         "a reference to the implementation or paper the method came from"),
        # --- the name of the retired side-by-side comparison ---
        ("LEFTOVER-TOKEN", re.compile(r"\b" + _j("par", "ity") + r"(?:_|\d|\b)", re.I),
         "a leftover token from a retired comparison"),
        # --- affiliation, programme, premises (R2) ---
        ("AFFILIATION", re.compile(
            r"\b(" + _j("intern", "ship") + r"|" + _j("in", "tern") + r"s?|"
            + _j("emplo", "yer") + r"|" + _j("men", "tor") + r"s?|" + _j("collea", "gue") + r"s?|"
            + _j("supervised ", "lab") + r"|" + _j("lab ", "session") + r"|"
            + _j("lab ", "workstation") + r"|" + _j("lab ", "network") + r"|"
            + _j("lab ", "hostname") + r"|" + _j("the ", "office") + r"(?![ _-]?lounge)|"
            + _j("the ", "company") + r"|" + _j("enterprise ", "wi-fi") + r"|"
            + _j("progra", "mme") + r"s?|" + _j("institu", "tion") + r"s?)\b"
            + r"|(?<!isaac )(?<!isaac-)\b" + _j("the ", "lab") + r"\b", re.I),
         "an affiliation, programme or premises"),
        # --- a URL to another party's repository (R1); vendor install sources
        # named in THIRD_PARTY_NOTICES.md are allowed by owner, assembled here ---
        ("OTHER-REPO-URL", re.compile(
            r"https?://(www\.)?" + _j("git", "hub") + r"\.com/(?!("
            + _j("micro", "soft") + r"|" + _j("facebook", "research") + r"|"
            + _j("hugging", "face") + r")/)[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", re.I),
         "a URL to another party's repository"),
        # --- ownership assertions and the author's name (R3) ---
        ("OWNERSHIP", re.compile(
            r"\b(" + _j("own ", "time") + r"|" + _j("own ", "equipment") + r"|"
            + _j("own ", "resources") + r"|work (made )?" + _j("for ", "hire") + r"|"
            + _j("Elie", "zer") + r"|" + _j("David", "son") + r"|"
            + _j("El", "i") + r"(?:'s)?|" + _j("own ", "(prior )?work") + r"|"
            + _j("the author's ", "own") + r"|" + _j("not a work", "( made)? for") + r")\b|"
            + _j("copyright", r"\s*(\(c\)|\u00a9|(\(c\)\s*)?(19|20)\d\d)"), re.I),
         "an ownership assertion or the author's name"),

        # --- agent-workflow internals ---
        ("WORKFLOW-INTERNAL", re.compile(
            r"(?<![A-Za-z0-9_])(\." + _j("super", "powers") + r"/|"
            + _j("docs/super", "powers") + r"/|" + _j("skill-obser", "vations")
            + r"/|\.claude/|\.cursor/)"),
         "an agent-workflow internal path"),
    ]


FORBIDDEN_NAMES = [
    ("FORBIDDEN-FILE", _j("CLAU", "DE.md")),
    ("FORBIDDEN-FILE", ".env"),
    ("FORBIDDEN-FILE", ".mcp.json"),
    ("FORBIDDEN-FILE", _j("HAND", "OFF.md")),
    ("FORBIDDEN-FILE", _j("PRO", "MPT.md")),
]
FORBIDDEN_DIRS = [
    # Byte-code caches embed the absolute build path of the machine that wrote
    # them. They are gitignored, so a push is safe -- but a zip or rsync of the
    # working directory is not, and that is how an export is usually handed over.
    ("FORBIDDEN-DIR", "__pycache__"),
    ("FORBIDDEN-DIR", ".claude"),
    ("FORBIDDEN-DIR", ".cursor"),
    ("FORBIDDEN-DIR", _j(".super", "powers")),
    ("FORBIDDEN-DIR", _j("skill-obser", "vations")),
]
FORBIDDEN_NAME_PATTERNS = [
    ("FORBIDDEN-FILE", re.compile(r"\.pyc$")),
    ("FORBIDDEN-FILE", re.compile(r"(?i)^" + _j("MORNING_RE", "PORT"))),
    ("FORBIDDEN-FILE", re.compile(r"(?i)^" + _j("NEXT-SESS", "ION"))),
    ("FORBIDDEN-FILE", re.compile(r"(?i)" + _j("HAND", "OFF") + r".*\.md$")),
    ("FORBIDDEN-FILE", re.compile(r"(?i)" + _j("PRO", "MPT") + r".*\.md$")),
]

REQUIRED_FILES = ["LICENSE", "THIRD_PARTY_NOTICES.md", "README.md"]

# Verbatim third-party licence texts are not this project's prose; they are
# exempted from the content rules by resolved directory, never by substring.
EXEMPT_DIRS = ("third_party_licenses",)

# Documented placeholders that stand in for a redacted address. This file's own
# rule table is exempted by resolved path in check_content, never by substring:
# a substring exemption would let any line anywhere opt out of every rule.
ALLOW_SUBSTRINGS = ("<G1_IP>", "<LAN_IP>", "<BLE_ADDR>")


class Report:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.checked = 0

    def fail(self, rule: str, where: str, detail: str) -> None:
        self.failures.append(f"[{rule}] {where}: {detail}")


def walk(root: Path):
    for p in sorted(root.rglob("*")):
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        yield p


def check_content(root: Path, rep: Report, verbose: bool) -> None:
    rules = _rules()
    this_file = (root / "tools" / "publish_check.py").resolve()
    for p in walk(root):
        if not p.is_file():
            continue
        # A file with no suffix (LICENSE, NOTICE) is prose; a binary one fails
        # the strict decode below and is skipped there.
        if p.suffix and p.suffix.lower() not in TEXT_SUFFIXES and p.name != ".gitignore":
            continue
        if p.resolve() == this_file:
            continue                      # its own rule table is not a leak
        if any(part in EXEMPT_DIRS for part in p.relative_to(root).parts):
            continue                      # a verbatim third-party licence
        try:
            text = p.read_text(encoding="utf-8", errors="strict")
        except (UnicodeDecodeError, OSError):
            continue
        rep.checked += 1
        rel = p.relative_to(root)
        for lineno, line in enumerate(text.splitlines(), 1):
            if any(a in line for a in ALLOW_SUBSTRINGS):
                continue
            for rule, pat, desc in rules:
                if pat.search(line):
                    rep.fail(rule, f"{rel}:{lineno}", desc)
        if verbose:
            print(f"  scanned {rel}")


# Content rules that also apply to a file's or directory's NAME: a file called
# after the other implementation, an affiliation or the author would ship its
# needle in the tree listing even if every line inside it were clean.
NAME_RULE_PREFIXES = ("OTHER-IMPL", "LEFTOVER-TOKEN", "AFFILIATION", "OWNERSHIP",
                      "PRIVATE-", "EMPLOYER", "PERSON", "HOSTNAME")


def check_names(root: Path, rep: Report) -> None:
    name_rules = [r for r in _rules() if r[0].startswith(NAME_RULE_PREFIXES)]
    for p in walk(root):
        rel = p.relative_to(root)
        raw = str(rel)
        split = re.sub(r"[/_.\-]+", " ", raw)     # so word boundaries see each fragment
        for rule, pat, desc in name_rules:
            if pat.search(raw) or pat.search(split):
                rep.fail(rule, str(rel), f"a path that names {desc}")
                break
        if p.is_dir():
            for rule, name in FORBIDDEN_DIRS:
                if p.name == name:
                    rep.fail(rule, str(rel), "a directory that must not ship")
            continue
        for rule, name in FORBIDDEN_NAMES:
            if p.name == name:
                rep.fail(rule, str(rel), "a file that must not ship")
        for rule, pat in FORBIDDEN_NAME_PATTERNS:
            if pat.search(p.name):
                rep.fail(rule, str(rel), "a file whose name matches a forbidden pattern")


def check_no_virtualenv(root: Path, rep: Report) -> None:
    for p in walk(root):
        if p.is_file() and p.name == "pyvenv.cfg":
            rep.fail("VIRTUALENV", str(p.relative_to(root)),
                     "a virtualenv marker -- a virtualenv has been copied in")
        if p.is_dir() and re.fullmatch(r"python3\.\d+", p.name) and "lib" in p.parts:
            rep.fail("VIRTUALENV", str(p.relative_to(root)),
                     "an installed-library tree")


def check_sizes(root: Path, rep: Report) -> None:
    for p in walk(root):
        if p.is_file() and p.stat().st_size > MAX_FILE_BYTES:
            mb = p.stat().st_size / 1024 / 1024
            rep.fail("FILE-TOO-BIG", str(p.relative_to(root)), f"{mb:.1f} MB > 10 MB")


def check_required(root: Path, rep: Report) -> None:
    for name in REQUIRED_FILES:
        if not (root / name).is_file():
            rep.fail("MISSING-REQUIRED", name, "required file is absent")


_LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")


def check_links(root: Path, rep: Report) -> None:
    for md in sorted(root.rglob("*.md")):
        if any(part in SKIP_DIRS for part in md.parts):
            continue
        rel = md.relative_to(root)
        for lineno, line in enumerate(md.read_text(encoding="utf-8").splitlines(), 1):
            for target in _LINK.findall(line):
                target = target.split(" ")[0].strip()
                if target.startswith(("http://", "https://", "mailto:", "#")):
                    continue
                path = target.split("#")[0]
                if not path:
                    continue
                dest = (md.parent / path).resolve()
                try:
                    dest.relative_to(root.resolve())
                except ValueError:
                    rep.fail("LINK-ESCAPES", f"{rel}:{lineno}",
                             "a relative link points outside the repository")
                    continue
                if not dest.exists():
                    rep.fail("LINK-BROKEN", f"{rel}:{lineno}",
                             f"relative link does not resolve: {path}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Pre-publication check.")
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    root = Path(args.root).resolve()
    rep = Report()
    check_content(root, rep, args.verbose)
    check_names(root, rep)
    check_no_virtualenv(root, rep)
    check_sizes(root, rep)
    check_required(root, rep)
    check_links(root, rep)

    print(f"publish_check: scanned {rep.checked} text files under {root.name}/")
    if rep.failures:
        print(f"FAIL -- {len(rep.failures)} problem(s):")
        for f in rep.failures:
            print("  " + f)
        return 1
    print("PASS -- nothing that must not be public was found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
