/**
 * Whistle STT engine, running in a Web Worker so transcription never blocks
 * the chat UI.
 *
 * Protocol (postMessage in, postMessage out):
 *   {type:"init"}            -> {type:"ready"} | {type:"error", message}
 *   {type:"chunk", pcm}      -> {type:"partial", committed, pending}
 *     pcm: Float32Array of 16 kHz mono samples (~1 s per call, transferred).
 *     committed: words two consecutive passes agreed on, joined.
 *     pending: the unconfirmed tail (may be revised by the next chunk).
 *   {type:"stop"}            -> {type:"final", text}
 *
 * Two integration facts discovered by the headless Node test, not the docs:
 *   - This is a WASM_BIGINT build: byte lengths passed to needle_load must
 *     be BigInt, or the call throws "Cannot convert to a BigInt".
 *   - Module.HEAPF32 is not exposed on this build; PCM bytes are copied
 *     through HEAPU8 instead (same memory, no conversion).
 */

/// <reference lib="webworker" />

import createNeedle from "./needle.cjs";

// eslint-disable-next-line @typescript-eslint/no-explicit-any
let M: any = null;
let outPtr = 0;
const OUT_CAP = 65536;

type Word = { word: string; start: number; end: number };
let committed: Word[] = [];
let lastStart = -1;

function readOut(): string {
  return M.UTF8ToString(outPtr) as string;
}

function takeCommitted(res: {
  words?: Word[];
}): { committed: string; pending: string; tail: string } {
  // Words are deduplicated by start time, so this is correct whether the
  // engine reports per-call or cumulative words: already-seen words are
  // skipped either way, and a repeated stream after stop() starts clean.
  for (const w of res.words ?? []) {
    if (typeof w.start === "number" && w.start > lastStart) {
      committed.push(w);
      lastStart = w.start;
    }
  }
  return {
    committed: committed.map((w) => w.word).join(" "),
    pending: (res as { pending?: string }).pending ?? "",
    tail: (res as { text?: string }).text ?? "",
  };
}

async function init(): Promise<void> {
  const [wasmBinary, cact] = await Promise.all([
    fetch("/stt/needle.wasm").then((r) => {
      if (!r.ok) throw new Error(`needle.wasm: HTTP ${r.status}`);
      return r.arrayBuffer();
    }),
    fetch("/stt/whistle.cact").then((r) => {
      if (!r.ok) throw new Error(`whistle.cact: HTTP ${r.status}`);
      return r.arrayBuffer();
    }),
  ]);
  M = await createNeedle({ wasmBinary: new Uint8Array(wasmBinary) });
  const bytes = new Uint8Array(cact);
  const cp = M._malloc(bytes.byteLength);
  M.HEAPU8.set(bytes, cp);
  const rc: number = M._needle_load(cp, BigInt(bytes.byteLength));
  M._free(cp);
  if (rc < 0) {
    throw new Error(M.UTF8ToString(M._needle_last_error()) || "needle_load failed");
  }
  outPtr = M._malloc(OUT_CAP);
  committed = [];
  lastStart = -1;
}

function processChunk(pcm: Float32Array): { committed: string; pending: string } {
  const pp: number = M._malloc(pcm.length * 4);
  M.HEAPU8.set(new Uint8Array(pcm.buffer, pcm.byteOffset, pcm.byteLength), pp);
  M.ccall(
    "needle_stream_transcribe_process",
    "number",
    ["number", "number", "string", "string", "number", "number"],
    // English forced: Whistle has no Hindi model, and short clinical
    // utterances misdetect more often than they code-switch.
    [pp, pcm.length, "en", null, outPtr, OUT_CAP],
  );
  M._free(pp);
  const res = JSON.parse(readOut() || "{}");
  const { committed: text, pending } = takeCommitted(res);
  return { committed: text, pending };
}

function stop(): string {
  M.ccall("needle_stream_transcribe_stop", "number", ["number", "number"], [
    outPtr,
    OUT_CAP,
  ]);
  const res = JSON.parse(readOut() || "{}");
  const { committed: text, tail } = takeCommitted(res);
  // stop() reports the tail as committed words; if it arrived only as text
  // (no word entries), append it unless it duplicates what we have.
  let finalText = text;
  if (tail && !text.endsWith(tail)) {
    finalText = (text ? text + " " : "") + tail;
  }
  committed = [];
  lastStart = -1;
  return finalText.trim();
}

self.onmessage = async (e: MessageEvent) => {
  const msg = e.data as { type: string; pcm?: Float32Array; id?: number };
  try {
    if (msg.type === "init") {
      await init();
      self.postMessage({ type: "ready", id: msg.id });
    } else if (msg.type === "chunk") {
      if (!M || !msg.pcm) return;
      const { committed: text, pending } = processChunk(msg.pcm);
      self.postMessage({ type: "partial", committed: text, pending });
    } else if (msg.type === "stop") {
      if (!M) {
        self.postMessage({ type: "final", text: "", id: msg.id });
        return;
      }
      self.postMessage({ type: "final", text: stop(), id: msg.id });
    }
  } catch (err) {
    self.postMessage({
      type: "error",
      id: msg.id,
      message: err instanceof Error ? err.message : String(err),
    });
  }
};
