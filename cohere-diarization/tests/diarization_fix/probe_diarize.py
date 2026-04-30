"""Quick diarization probe — calls /diarize/path and prints segment summary."""
import asyncio
import json
import sys
import aiohttp


async def main(wav_path: str):
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300)) as session:
        async with session.post(
            "http://127.0.0.1:8000/diarize/path",
            json={"wav_path": wav_path},
        ) as resp:
            async for raw in resp.content:
                line = raw.strip()
                if not line:
                    continue
                d = json.loads(line)
                t = d.get("type")
                if t == "progress":
                    print(f"  [{d['step']}] {d['completed']}/{d['total']}", flush=True)
                elif t == "result":
                    segs = d["segments"]
                    profiles = d.get("profiles", {})
                    speakers: dict = {}
                    for seg in segs:
                        speakers.setdefault(seg["speaker"], []).append(seg)
                    print(f"\nTotal segments : {len(segs)}")
                    print(f"Unique speakers: {sorted(speakers.keys())}")
                    if profiles:
                        print("\nSPEAKER VOICE PROFILES:")
                        for spk in sorted(profiles.keys()):
                            p = profiles[spk]
                            print(f"  {spk}: pitch={p.get('pitch_hz',0):.0f}Hz (±{p.get('pitch_std',0):.0f}Hz)  "
                                  f"energy={p.get('energy_rms',0):.4f}  "
                                  f"speech={p.get('total_speech_sec',0):.0f}s  "
                                  f"gender={p.get('gender_hint','?')}")
                    print()
                    for seg in segs:
                        dur = seg["end"] - seg["start"]
                        print(f"  {seg['speaker']:12s}  {seg['start']:7.2f}s - {seg['end']:7.2f}s  ({dur:.1f}s)")
                elif t == "error":
                    print(f"ERROR: {d.get('error')}", file=sys.stderr)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: probe_diarize.py <wav_or_mp4_path>")
        sys.exit(1)
    asyncio.run(main(sys.argv[1]))
