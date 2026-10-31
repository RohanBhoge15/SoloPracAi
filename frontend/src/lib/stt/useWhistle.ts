/**
 * useWhistle: offline mic dictation for the chat composer.
 *
 * Press mic -> the engine loads once (~17 MB fetch, cached by the
 * browser afterwards) -> every ~1 s of speech is sent to the worker ->
 * committed words plus the pending tail stream back and are composed
 * into `live` text. Press again (or stop) -> the tail flushes and `final`
 * is delivered.
 *
 * Audio path: getUserMedia -> AudioContext at 16 kHz -> ScriptProcessor.
 * ScriptProcessor is deprecated but universal (Chrome, Safari, Firefox);
 * an AudioWorklet would need a second bundled module for no audible gain
 * on 1 s chunks. If the context cannot run at 16 kHz, samples are
 * linearly resampled rather than fed at the wrong rate, which would read
 * as chipmunk speech to the model.
 *
 * Nothing leaves the device: no fetch after the two asset loads, no key,
 * no usage meter. A failed load or a denied mic reports through onError
 * and leaves the text input exactly as it was.
 */

import { useCallback, useEffect, useRef, useState } from "react";

export type SttState = "idle" | "loading" | "listening" | "error";

const CHUNK_SAMPLES = 16000; // ~1 s, the cadence the stream API expects
const INIT_TIMEOUT_MS = 60000;

function resampleLinear(input: Float32Array, fromRate: number): Float32Array {
  if (fromRate === 16000) return input;
  const ratio = fromRate / 16000;
  const outLen = Math.floor(input.length / ratio);
  const out = new Float32Array(outLen);
  for (let i = 0; i < outLen; i++) {
    const pos = i * ratio;
    const i0 = Math.floor(pos);
    const frac = pos - i0;
    const a = input[i0] ?? 0;
    const b = input[i0 + 1] ?? a;
    out[i] = a + (b - a) * frac;
  }
  return out;
}

export function useWhistle(opts: {
  onLive: (text: string) => void;
  onDone: (text: string) => void;
  onError: (message: string) => void;
}) {
  const [state, setState] = useState<SttState>("idle");
  const workerRef = useRef<Worker | null>(null);
  const audioRef = useRef<{
    ctx: AudioContext;
    source: MediaStreamAudioSourceNode;
    proc: ScriptProcessorNode;
    stream: MediaStream;
  } | null>(null);
  const accRef = useRef<Float32Array>(new Float32Array(0));
  const liveRef = useRef("");
  const rpcRef = useRef(
    new Map<number, { resolve: (v: string) => void; reject: (e: Error) => void }>(),
  );
  const idRef = useRef(0);
  const optsRef = useRef(opts);
  optsRef.current = opts;

  const getWorker = useCallback(() => {
    if (!workerRef.current) {
      workerRef.current = new Worker(new URL("./whistle.worker.ts", import.meta.url), {
        type: "module",
      });
      workerRef.current.onmessage = (e: MessageEvent) => {
        const msg = e.data as {
          type: string;
          id?: number;
          committed?: string;
          pending?: string;
          text?: string;
          message?: string;
        };
        if (msg.type === "partial") {
          // Committed words are final; the pending tail may be revised by
          // the next chunk, so it is shown but never stored.
          const live = [msg.committed, msg.pending].filter(Boolean).join(" ");
          liveRef.current = live;
          optsRef.current.onLive(live);
        } else if ((msg.type === "ready" || msg.type === "final") && msg.id != null) {
          rpcRef.current.get(msg.id)?.resolve(msg.text ?? "");
          rpcRef.current.delete(msg.id);
        } else if (msg.type === "error") {
          if (msg.id != null && rpcRef.current.has(msg.id)) {
            rpcRef.current.get(msg.id)?.reject(new Error(msg.message ?? "worker error"));
            rpcRef.current.delete(msg.id);
          } else {
            optsRef.current.onError(msg.message ?? "transcription failed");
            setState("error");
          }
        }
      };
    }
    return workerRef.current;
  }, []);

  const callWorker = useCallback(
    (type: "init" | "stop", timeoutMs: number): Promise<string> =>
      new Promise((resolve, reject) => {
        const id = ++idRef.current;
        rpcRef.current.set(id, { resolve, reject });
        try {
          getWorker().postMessage({ type, id });
        } catch (e) {
          rpcRef.current.delete(id);
          reject(e instanceof Error ? e : new Error(String(e)));
          return;
        }
        window.setTimeout(() => {
          if (rpcRef.current.has(id)) {
            rpcRef.current.delete(id);
            reject(new Error(`${type} timed out`));
          }
        }, timeoutMs);
      }),
    [getWorker],
  );

  const teardownAudio = useCallback(() => {
    const a = audioRef.current;
    audioRef.current = null;
    accRef.current = new Float32Array(0);
    if (!a) return;
    try {
      a.proc.onaudioprocess = null;
      a.source.disconnect();
      a.proc.disconnect();
      void a.ctx.close();
    } catch {
      // Teardown is best-effort; a half-closed graph must not throw.
    }
    for (const track of a.stream.getTracks()) {
      try {
        track.stop();
      } catch {
        // A track that refuses to stop is already gone.
      }
    }
  }, []);

  const start = useCallback(async () => {
    if (state === "loading" || state === "listening") return;
    setState("loading");
    liveRef.current = "";
    try {
      await callWorker("init", INIT_TIMEOUT_MS);
    } catch (e) {
      setState("error");
      optsRef.current.onError(
        e instanceof Error ? `Speech engine failed to load: ${e.message}` : "Speech engine failed to load",
      );
      return;
    }
    let stream: MediaStream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate: 16000,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
        },
      });
    } catch {
      setState("error");
      optsRef.current.onError("Microphone unavailable. Check the browser permission and try again.");
      return;
    }
    try {
      const Ctx: typeof AudioContext | undefined =
        window.AudioContext ??
        (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
      if (!Ctx) throw new Error("this browser has no Web Audio");
      const ctx = new Ctx({ sampleRate: 16000 });
      const source = ctx.createMediaStreamSource(stream);
      const proc = ctx.createScriptProcessor(4096, 1, 1);
      const worker = getWorker();
      const rate = ctx.sampleRate;
      proc.onaudioprocess = (e) => {
        const input = e.inputBuffer.getChannelData(0);
        const mono = resampleLinear(input.slice(), rate);
        const acc = accRef.current;
        const joined = new Float32Array(acc.length + mono.length);
        joined.set(acc);
        joined.set(mono, acc.length);
        let off = 0;
        while (joined.length - off >= CHUNK_SAMPLES) {
          const piece = joined.slice(off, off + CHUNK_SAMPLES);
          off += CHUNK_SAMPLES;
          worker.postMessage({ type: "chunk", pcm: piece }, [piece.buffer]);
        }
        accRef.current = joined.slice(off);
      };
      source.connect(proc);
      // ScriptProcessor only fires while the graph pulls audio, which
      // requires a path to the destination. A zero-gain node keeps it
      // silent: without it Chrome never calls onaudioprocess, and without
      // the gain the doctor hears their own mic back (feedback).
      const sink = ctx.createGain();
      sink.gain.value = 0;
      proc.connect(sink);
      sink.connect(ctx.destination);
      audioRef.current = { ctx, source, proc, stream };
      setState("listening");
    } catch (e) {
      for (const track of stream.getTracks()) {
        try {
          track.stop();
        } catch {
          // Already gone.
        }
      }
      setState("error");
      optsRef.current.onError(e instanceof Error ? e.message : "Audio setup failed");
    }
  }, [callWorker, getWorker, state]);

  const stop = useCallback(async () => {
    if (state !== "listening") return "";
    // Stop capturing first so no more chunks queue behind the flush.
    teardownAudio();
    let finalText = liveRef.current;
    try {
      finalText = await callWorker("stop", 30000);
    } catch {
      // The flush failed; keep whatever was committed so far rather than
      // discarding a whole dictation.
    }
    liveRef.current = "";
    setState("idle");
    optsRef.current.onDone(finalText.trim());
    return finalText.trim();
  }, [callWorker, state, teardownAudio]);

  // Unmount mid-dictation must not leave the mic open.
  useEffect(
    () => () => {
      teardownAudio();
      workerRef.current?.terminate();
      workerRef.current = null;
    },
    [teardownAudio],
  );

  return { state, start, stop };
}
