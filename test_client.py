#!/usr/bin/env python3
"""
Fake-device test client for Sarah's WebSocket endpoint.

Sends the hello / start_turn / <binary PCM frames> / end_turn sequence
against a real .wav file, prints every status/transcript/error message
the server sends back, and writes whatever audio comes back to reply.wav.

Requires: pip install websockets

Usage:
    python test_ws_client.py --wav sample.wav
    python test_ws_client.py --wav sample.wav --server ws://192.168.1.50:9000/ws
"""

import argparse
import asyncio
import json
import struct
import uuid
import wave

import websockets

CHUNK_MS = 100  # how much audio to send per binary frame


def read_wav_as_pcm16(path: str) -> tuple[bytes, int, int]:
    """Returns (raw_pcm_bytes, sample_rate, channels). Expects 16-bit PCM."""
    with wave.open(path, "rb") as wf:
        if wf.getsampwidth() != 2:
            raise ValueError(
                f"{path} is {wf.getsampwidth()*8}-bit, need 16-bit PCM"
            )
        sample_rate = wf.getframerate()
        channels = wf.getnchannels()
        pcm = wf.readframes(wf.getnframes())
    return pcm, sample_rate, channels


def write_wav(path: str, pcm: bytes, sample_rate: int, channels: int) -> None:
    with wave.open(path, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)


async def run(server: str, wav_path: str, device_id: str, out_path: str) -> None:
    pcm, sample_rate, channels = read_wav_as_pcm16(wav_path)
    print(f"[client] loaded {wav_path}: {len(pcm)} bytes, "
          f"{sample_rate} Hz, {channels}ch")
    if sample_rate != 16000 or channels != 1:
        print("[client] WARNING: server/STT expects 16kHz mono — "
              "this file may not transcribe well")

    bytes_per_chunk = int(sample_rate * channels * 2 * (CHUNK_MS / 1000))
    bytes_per_chunk -= bytes_per_chunk % 2  # keep it even (16-bit samples)

    turn_id = uuid.uuid4().hex[:8]
    reply_pcm = bytearray()
    reply_info = {"sample_rate": 24000, "channels": 1}
    receiving_audio = False

    # ping_interval=None: disable the client's own keepalive pings. The
    # server blocks synchronously for the whole STT->LLM->TTS turn and
    # won't process control frames until it's done, so the client's
    # default 20s ping/pong timeout can fire mid-turn on slower hardware
    # and kill the connection before any response comes back.
    async with websockets.connect(server, max_size=None, ping_interval=None) as ws:
        # 1. hello / hello_ack
        await ws.send(json.dumps({
            "type": "hello",
            "device_id": device_id,
            "protocol_version": 1,
            "audio": {"sample_rate": sample_rate, "encoding": "pcm_s16le",
                      "channels": channels},
        }))
        ack = json.loads(await ws.recv())
        print(f"[server] {ack}")
        if ack.get("type") != "hello_ack":
            print("[client] no hello_ack, aborting")
            return

        # 2. start_turn
        await ws.send(json.dumps({"type": "start_turn", "turn_id": turn_id,
                                  "trigger": "test_script"}))
        print(f"[client] start_turn {turn_id}")

        # 3. stream PCM in chunks
        n_chunks = 0
        for i in range(0, len(pcm), bytes_per_chunk):
            await ws.send(pcm[i:i + bytes_per_chunk])
            n_chunks += 1
        print(f"[client] streamed {n_chunks} audio frames")

        # 4. end_turn
        await ws.send(json.dumps({"type": "end_turn", "turn_id": turn_id}))
        print(f"[client] end_turn {turn_id}")

        # 5. read everything back until tts_end or error
        while True:
            msg = await ws.recv()
            if isinstance(msg, (bytes, bytearray)):
                reply_pcm.extend(msg)
                continue

            data = json.loads(msg)
            mtype = data.get("type")
            print(f"[server] {data}")

            if mtype == "tts_start":
                receiving_audio = True
                reply_info["sample_rate"] = data.get("sample_rate", 24000)
                reply_info["channels"] = data.get("channels", 1)
            elif mtype == "tts_end":
                break
            elif mtype == "error":
                break

    if reply_pcm:
        write_wav(out_path, bytes(reply_pcm), reply_info["sample_rate"],
                  reply_info["channels"])
        print(f"[client] saved reply audio -> {out_path} "
              f"({len(reply_pcm)} bytes)")
    else:
        print("[client] no audio came back")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wav", required=True, help="path to a 16kHz mono 16-bit WAV to send")
    parser.add_argument("--server", default="ws://localhost:9000/ws")
    parser.add_argument("--device-id", default="test-client")
    parser.add_argument("--out", default="reply.wav", help="where to save the returned audio")
    args = parser.parse_args()

    asyncio.run(run(args.server, args.wav, args.device_id, args.out))


if __name__ == "__main__":
    main()