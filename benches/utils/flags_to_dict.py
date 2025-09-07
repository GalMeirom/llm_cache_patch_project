# flags_to_dict.py
import json
import sys


def _coerce(v: str):
    vl = v.lower()
    if vl in ("true", "yes", "on"):
        return True
    if vl in ("false", "no", "off"):
        return False
    try:
        if vl.startswith(("0x", "-0x")):
            return int(v, 16)
        return int(v)
    except ValueError:
        try:
            return float(v)
        except ValueError:
            return v


def parse_flags(argv) -> dict:
    out, pos = {}, []
    i = 0
    n = len(argv)
    while i < n:
        tok = argv[i]

        # --key=value or --key value or --flag
        if tok.startswith("--"):
            keyval = tok[2:]
            if "=" in keyval:
                k, v = keyval.split("=", 1)
                out[k] = _coerce(v)
            else:
                nxt = argv[i + 1] if i + 1 < n else None
                if nxt is None or nxt.startswith("-"):
                    out[keyval] = True
                else:
                    out[keyval] = _coerce(nxt)
                    i += 1

        # -k=value, -k value, or -abc (booleans)
        elif tok.startswith("-") and tok != "-":
            body = tok[1:]
            if "=" in body:
                k, v = body.split("=", 1)
                # if multiple letters before "=", treat as one key
                out[k] = _coerce(v)
            else:
                nxt = argv[i + 1] if i + 1 < n else None
                if len(body) == 1 and nxt is not None and not nxt.startswith("-"):
                    out[body] = _coerce(nxt)
                    i += 1
                else:
                    for ch in body:
                        out[ch] = True
        else:
            pos.append(tok)
        i += 1

    if pos:
        out["_positional"] = pos
    return out


if __name__ == "__main__":
    # skip script name
    result = parse_flags(sys.argv[1:])
    print(json.dumps(result, indent=2))
