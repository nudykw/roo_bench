"""Quick diagnostic test: checks what Ollama returns in streaming vs non-streaming mode."""

import json
import sys

import requests

OLLAMA_URL = "http://aorus-cachyos-server:11434"
MODEL = "gemma4:latest"
PROMPT = "Say 'hello world' only."
NUM_CTX = 512
NUM_PREDICT = 20


def test_stream_true():
    print("\n" + "=" * 60)
    print("TEST 1: stream=True")
    print("=" * 60)
    payload = {
        "model": MODEL,
        "prompt": PROMPT,
        "stream": True,
        "options": {"num_ctx": NUM_CTX, "num_predict": NUM_PREDICT},
        "think": False,
    }
    response = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=60, stream=True)
    print(f"  HTTP status: {response.status_code}")
    lines = []
    for line in response.iter_lines():
        if line:
            lines.append(line.decode("utf-8"))
    print(f"  Number of lines from iter_lines(): {len(lines)}")
    full_response_text = ""
    for i, line_str in enumerate(lines):
        try:
            obj = json.loads(line_str)
            tok = obj.get("response", "")
            if tok:
                full_response_text += tok
            done = obj.get("done", False)
            if done:
                print(f"  [done=true] line[{i}]: eval_count={obj.get('eval_count')}, response={repr(tok)}")
            elif tok:
                print(f"  [token]     line[{i}]: response={repr(tok[:50])}")
        except json.JSONDecodeError:
            print(f"  [PARSE ERR] line[{i}]: {line_str[:100]!r}")
    print(f"  Collected response_text length: {len(full_response_text)}")
    print(f"  Collected response_text: {full_response_text!r}")
    return full_response_text


def test_stream_false():
    print("\n" + "=" * 60)
    print("TEST 2: stream=False")
    print("=" * 60)
    payload = {
        "model": MODEL,
        "prompt": PROMPT,
        "stream": False,
        "options": {"num_ctx": NUM_CTX, "num_predict": NUM_PREDICT},
        "think": False,
    }
    response = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=60, stream=False)
    print(f"  HTTP status: {response.status_code}")
    data = response.json()
    response_text = data.get("response", "")
    eval_count = data.get("eval_count", 0)
    total_duration = data.get("total_duration", 0)
    tps = eval_count / (total_duration / 1e9) if total_duration > 0 else 0
    print(f"  eval_count: {eval_count}")
    print(f"  total_duration: {total_duration / 1e9:.2f}s")
    print(f"  TPS: {tps:.2f}")
    print(f"  response_text length: {len(response_text)}")
    print(f"  response_text: {response_text!r}")
    return response_text


def test_stream_true_raw_bytes():
    """Check raw bytes to see exact format of streaming response."""
    print("\n" + "=" * 60)
    print("TEST 3: stream=True - raw iter_content()")
    print("=" * 60)
    payload = {
        "model": MODEL,
        "prompt": PROMPT,
        "stream": True,
        "options": {"num_ctx": NUM_CTX, "num_predict": NUM_PREDICT},
        "think": False,
    }
    response = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=60, stream=True)
    print(f"  HTTP status: {response.status_code}")
    chunks = list(response.iter_content(chunk_size=None))
    print(f"  Number of iter_content() chunks: {len(chunks)}")
    all_bytes = b"".join(chunks)
    print(f"  Total bytes received: {len(all_bytes)}")
    lines = all_bytes.split(b"\n")
    non_empty = [l for l in lines if l.strip()]
    print(f"  Lines after split('\\n'): {len(lines)} total, {len(non_empty)} non-empty")
    full_text = ""
    for i, line in enumerate(non_empty[:5]):
        line_str = line.decode("utf-8")
        try:
            obj = json.loads(line_str)
            tok = obj.get("response", "")
            full_text += tok
            print(f"  line[{i}]: done={obj.get('done')} response={repr(tok[:30])}")
        except json.JSONDecodeError:
            print(f"  line[{i}]: PARSE_ERR {line_str[:80]!r}")
    if len(non_empty) > 5:
        print(f"  ... ({len(non_empty) - 5} more lines)")
    print(f"  First 200 raw bytes: {all_bytes[:200]!r}")


if __name__ == "__main__":
    ollama_url = OLLAMA_URL
    model = MODEL
    for arg in sys.argv[1:]:
        if arg.startswith("--url="):
            ollama_url = arg[6:]
            OLLAMA_URL = ollama_url
        elif arg.startswith("--model="):
            model = arg[8:]
            MODEL = model

    print(f"Ollama URL: {OLLAMA_URL}")
    print(f"Model: {MODEL}")
    print(f"Prompt: {PROMPT!r}")

    try:
        text1 = test_stream_true()
        text2 = test_stream_false()
        test_stream_true_raw_bytes()

        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"  stream=True  response text: {len(text1)} chars")
        print(f"  stream=False response text: {len(text2)} chars")
        if len(text2) > 0 and len(text1) == 0:
            print("  ✅ CONCLUSION: stream=False works, stream=True does NOT capture text (SSH buffering issue)")
        elif len(text1) > 0 and len(text2) > 0:
            print("  ✅ CONCLUSION: Both modes work for text capture")
        else:
            print("  ❌ CONCLUSION: Neither mode captures response text - check model/connection")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
