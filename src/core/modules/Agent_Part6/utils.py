import re
from typing import List, Dict

_BLANK_RE = re.compile(r'_{2,}|\[BLANK\]|\<BLANK\>', flags=re.I)
_HEADER_RE = re.compile(r'^\s*(To:|From:|Subject:|Date:)\b', flags=re.I)
_BLOCK_NUM_LINE = re.compile(r'^\s*(\d{1,3})\.\s*$', flags=re.M)
_OPTION_LINE = re.compile(r'^\s*\(?[A-Da-d]\)?[\.\)]\s+.*')  
_GARBAGE_TOKEN_RE = re.compile(r'\(\(+\d+\)+\)|Choices:?', flags=re.I)

def _split_lines_keep(text: str) -> List[str]:
    return [ln for ln in text.replace('\r\n','\n').split('\n')]

def _is_header_line(line: str) -> bool:
    return bool(_HEADER_RE.match(line))

def _is_choice_block_start(line: str) -> bool:
    return bool(re.match(r'^\s*\d{1,3}\.\s*$', line))

def _clean_choice_lines(lines: List[str]) -> List[str]:
    cleaned = []
    seen = set()
    for ln in lines:
        if not ln or ln.strip()=="":
            continue
        ln2 = _GARBAGE_TOKEN_RE.sub('', ln).strip()
        ln2 = re.sub(r'^\s*\d{1,3}\.\s*', '', ln2).strip()
        if not ln2:
            continue
        if ln2 not in seen:
            cleaned.append(ln2)
            seen.add(ln2)
    return cleaned

def _collect_choice_blocks(lines: List[str], start_idx: int) -> Dict[str, List[str]]:
    blocks = {}
    i = start_idx
    n = len(lines)
    current_num = None
    current_buf = []
    while i < n:
        ln = lines[i].rstrip()
        m = _BLOCK_NUM_LINE.match(ln)
        if m:
            if current_num is not None:
                blocks[current_num] = _clean_choice_lines(current_buf)
            current_num = m.group(1)
            current_buf = []
        else:
            current_buf.append(ln)
        i += 1
    if current_num is not None:
        blocks[current_num] = _clean_choice_lines(current_buf)
    return blocks

def clean_and_extract_passage_simple(passage: str) -> str:
    """
    Simpler & robust: if a line has a blank, return that line and any immediate following
    option lines (A/B/C/D). If no inline options present, fallback to parsing numeric choice blocks.
    """
    if not passage:
        return ""

    lines = _split_lines_keep(passage)
    # find first numeric choice block if exists
    first_choice_idx = None
    for idx, ln in enumerate(lines):
        if _is_choice_block_start(ln):
            first_choice_idx = idx
            break

    # header_lines: keep leading To/From/Subject/Date lines
    header_lines = []
    i = 0
    while i < len(lines) and _is_header_line(lines[i]):
        header_lines.append(lines[i].rstrip())
        i += 1

    # split body and choice lines
    if first_choice_idx is not None:
        body_lines = lines[:first_choice_idx]
        choice_lines = lines[first_choice_idx:]
    else:
        body_lines = lines
        choice_lines = []

    out_parts = []
    if header_lines:
        out_parts.extend([ln.rstrip() for ln in header_lines])
        out_parts.append("")

    # scan body_lines: for each line that contains a blank, capture it and any following option lines
    used_body_indices = set()
    for idx, ln in enumerate(body_lines):
        if _BLANK_RE.search(ln):
            # capture this line
            out_parts.append(ln.strip())
            used_body_indices.add(idx)
            # capture immediate following option lines (A/B/C/D), stop on first non-option or blank line
            j = idx + 1
            found_opt = False
            while j < len(body_lines):
                next_ln = body_lines[j].strip()
                if _OPTION_LINE.match(next_ln):
                    out_parts.append(next_ln)
                    used_body_indices.add(j)
                    found_opt = True
                    j += 1
                    continue
                # also accept lines like "A) attends" without parentheses, handled by regex above
                # stop if next line is empty or another header/metadata line
                break
            out_parts.append("")  # blank line after each question block

    # If no inline options were found for any question, fall back to parsing numeric choice blocks
    if not any(_OPTION_LINE.match(l.strip()) for l in body_lines):
        if choice_lines:
            choices_map = _collect_choice_blocks(choice_lines, 0)
            # append cleaned choice blocks
            if choices_map:
                for num in sorted(choices_map, key=lambda x: int(x)):
                    out_parts.append(f"{num}.")
                    for o in choices_map[num]:
                        out_parts.append(o)
        else:
            # nothing to append
            pass
    else:
        # if some inline options were captured already, we keep them and do NOT duplicate numeric blocks
        pass
    return "\n".join(out_parts).strip()