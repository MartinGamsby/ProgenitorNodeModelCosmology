"""Synthesize the soundtrack locally (no samples, no licences): a seamless ambient pad loop plus
three sound effects. Writes video/public/audio/{music,boom,whoosh,ding}.wav.  npm run music"""
import os
import wave

import numpy as np

SR = 44100
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "public", "audio")
rng = np.random.default_rng(7)


def write(name, x, stereo=True):
    x = np.asarray(x, float)
    if x.ndim == 1:
        x = np.stack([x, x], 1)
    x = x / (np.max(np.abs(x)) + 1e-9) * 0.89
    os.makedirs(OUT, exist_ok=True)
    with wave.open(os.path.join(OUT, name), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((x * 32767).astype("<i2").tobytes())
    print("wrote", name, f"{len(x) / SR:.1f}s")


def midi(n):
    return 440.0 * 2 ** ((n - 69) / 12)


def lowpass(x, cutoff):
    a = np.exp(-2 * np.pi * cutoff / SR)
    y = np.empty_like(x)
    acc = 0.0
    for i, v in enumerate(x):  # one-pole; fine for short sfx
        acc = (1 - a) * v + a * acc
        y[i] = acc
    return y


def reverb(x, wet=0.35, seconds=3.5):
    """Exponentially decaying noise impulse response (FFT convolution)."""
    n = int(seconds * SR)
    ir = rng.standard_normal((n, 2)) * np.exp(-np.linspace(0, 7, n))[:, None]
    ir[: int(0.01 * SR)] = 0
    if x.ndim == 1:
        x = np.stack([x, x], 1)
    L = len(x) + n
    nfft = 1 << (L - 1).bit_length()
    y = np.stack([np.fft.irfft(np.fft.rfft(x[:, c], nfft) * np.fft.rfft(ir[:, c], nfft), nfft)[:L]
                  for c in range(2)], 1)
    y /= np.max(np.abs(y)) + 1e-9
    xx = np.vstack([x / (np.max(np.abs(x)) + 1e-9), np.zeros((n, 2))])
    return (1 - wet) * xx + wet * y


def pad():
    # Am(add9) - Fmaj7 - C(add9) - Em7, 8 s each, two passes = 64 s seamless loop.
    chords = [[57, 64, 67, 71, 72], [53, 60, 64, 69, 72], [48, 55, 62, 64, 67], [52, 59, 62, 67, 71]]
    bass = [45, 41, 36, 40]
    dur, n_loop = 8.0, 2
    total = dur * len(chords) * n_loop
    N = int(total * SR)
    t = np.arange(N) / SR
    out = np.zeros((N, 2))
    seg = int(dur * SR)
    fade = int(2.5 * SR)
    for k in range(len(chords) * n_loop):
        notes = chords[k % 4]
        start = k * seg
        idx = (np.arange(start - fade, start + seg + fade)) % N  # wrap => seamless loop
        tt = np.arange(len(idx)) / SR
        env = np.minimum(1, np.minimum(tt / (fade / SR), (tt[-1] - tt) / (fade / SR))) ** 2
        for j, n in enumerate(notes + [bass[k % 4]]):
            f = midi(n)
            amp = 0.5 if j == len(notes) else 0.22
            for c, det in enumerate((-0.12, 0.12)):  # stereo detune
                ph = 2 * np.pi * f * (1 + det / 100) * (tt + start / SR)
                trem = 1 + 0.25 * np.sin(2 * np.pi * (0.07 + 0.03 * j) * (tt + start / SR) + j)
                v = np.sin(ph) + 0.18 * np.sin(2 * ph) + 0.06 * np.sin(3 * ph)
                np.add.at(out[:, c], idx, amp * env * trem * v)
    # shimmer: sparse high bell notes from the chord, every 2 s
    for k in range(int(total / 2)):
        notes = chords[(k // 4) % 4]
        n = notes[rng.integers(len(notes))] + 12
        s = int(k * 2 * SR + rng.uniform(0, 0.6) * SR)
        L = int(3 * SR)
        tt = np.arange(L) / SR
        b = np.sin(2 * np.pi * midi(n) * tt) * np.exp(-tt * 1.6) * 0.05
        pan = rng.uniform(0.2, 0.8)
        idx = (s + np.arange(L)) % N
        np.add.at(out[:, 0], idx, b * (1 - pan))
        np.add.at(out[:, 1], idx, b * pan)
    wet = reverb(out, wet=0.45, seconds=4.0)
    # wrap the reverb tail into the start so the loop stays seamless
    loop = wet[:N].copy()
    tail = wet[N:]
    loop[: len(tail)] += tail
    write("music.wav", loop * 0.6)


def boom():
    L = int(4.5 * SR)
    t = np.arange(L) / SR
    f = 30 + 70 * np.exp(-t * 3)
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 1.1)
    noise = lowpass(rng.standard_normal(L), 900) * np.exp(-t * 4) * 0.8
    write("boom.wav", reverb(body + noise, wet=0.3, seconds=2.5))


def whoosh():
    L = int(1.4 * SR)
    t = np.arange(L) / SR
    env = np.sin(np.pi * np.clip(t / 1.2, 0, 1)) ** 2
    n = rng.standard_normal(L)
    lo = lowpass(n, 500)
    hi = lowpass(n, 3000)
    sweep = np.clip(t / 1.2, 0, 1)
    x = (lo * (1 - sweep) + (hi - lo) * sweep) * env
    write("whoosh.wav", reverb(x * 0.6, wet=0.25, seconds=1.2))


def ding():
    L = int(3 * SR)
    t = np.arange(L) / SR
    f0 = midi(84)
    x = sum(a * np.sin(2 * np.pi * f0 * r * t) * np.exp(-t * d)
            for r, a, d in [(1, 1, 1.5), (2.76, 0.4, 3), (5.4, 0.2, 5), (8.9, 0.1, 8)])
    write("ding.wav", reverb(x * 0.5, wet=0.3, seconds=2))


if __name__ == "__main__":
    pad()
    boom()
    whoosh()
    ding()
