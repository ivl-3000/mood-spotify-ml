import argparse
import base64
import pathlib
import sys
import time
import zlib
import requests


def _download(url: str, timeout_s: int) -> bytes:
    resp = requests.get(url, timeout=timeout_s)
    resp.raise_for_status()
    return resp.content


def _render_via_mermaid_ink(mermaid_src: str, timeout_s: int) -> bytes:
    # mermaid.ink expects raw DEFLATE (wbits=-15) then base64-url without padding
    compressed = zlib.compress(mermaid_src.encode("utf-8"), level=9)
    # Convert to raw DEFLATE by stripping zlib header/footer if needed
    # zlib header is 2 bytes, footer is 4 bytes (adler32)
    if len(compressed) > 6 and compressed[0] == 0x78:
        raw_deflate = compressed[2:-4]
    else:
        # Fallback: attempt raw DEFLATE via zlib with wbits=-15
        raw_deflate = zlib.compressobj(level=9, wbits=-15).compress(mermaid_src.encode("utf-8"))
        raw_deflate += zlib.compressobj(level=9, wbits=-15).flush()
    b64 = base64.urlsafe_b64encode(raw_deflate).decode("ascii").rstrip("=")
    url = f"https://mermaid.ink/img/{b64}"
    return _download(url, timeout_s)


def render_mermaid_to_png(input_path: pathlib.Path, output_path: pathlib.Path, kroki_url: str, timeout_s: int = 180, retries: int = 3) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mermaid_src = input_path.read_text(encoding="utf-8")

    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(
                kroki_url.rstrip("/") + "/mermaid/png",
                data=mermaid_src.encode("utf-8"),
                headers={"Content-Type": "text/plain; charset=utf-8"},
                timeout=timeout_s,
            )
            resp.raise_for_status()
            output_path.write_bytes(resp.content)
            return
        except Exception as e:  # requests exceptions
            last_err = e
            if attempt < retries:
                sleep_s = 5 * attempt
                print(f"Attempt {attempt}/{retries} failed for {input_path.name}: {e}. Retrying in {sleep_s}s...")
                time.sleep(sleep_s)
            else:
                print(f"Kroki failed for {input_path.name}: {e}. Falling back to mermaid.ink ...")
    # Fallback to mermaid.ink
    try:
        png_bytes = _render_via_mermaid_ink(mermaid_src, timeout_s)
        output_path.write_bytes(png_bytes)
        return
    except Exception as e:
        last_err = e
    assert last_err is not None
    raise last_err


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Render Mermaid diagrams to PNG via Kroki with mermaid.ink fallback")
    parser.add_argument("inputs", nargs="+", help="Input .mmd files")
    parser.add_argument("--outdir", default="docs/diagrams/img", help="Output directory for PNGs")
    parser.add_argument("--kroki", default="https://kroki.io", help="Kroki base URL")
    parser.add_argument("--timeout", type=int, default=180, help="HTTP timeout seconds")
    parser.add_argument("--retries", type=int, default=3, help="Retry count on failure")
    args = parser.parse_args(argv)

    outdir = pathlib.Path(args.outdir)

    for in_file in args.inputs:
        in_path = pathlib.Path(in_file)
        if not in_path.exists():
            print(f"Skipping missing file: {in_file}")
            continue
        out_name = in_path.stem + ".png"
        out_path = outdir / out_name
        print(f"Rendering {in_path} -> {out_path}")
        render_mermaid_to_png(in_path, out_path, args.kroki, timeout_s=args.timeout, retries=args.retries)

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
