#!/usr/bin/env python3
"""
Voiceprint management CLI - Extract segments and refine voiceprints.

Usage:
    # Extract segments using existing voiceprints for speaker identification:
    python voiceprint_mgmt.py extract --audio meeting.mp4 --voiceprints voiceprints.json --output segments/

    # Extract segments for a speaker from diarization output:
    python voiceprint_mgmt.py extract --audio meeting.mp4 --diarize diarization.json --speaker "SPEAKER_00" --output segments/

    # Extract a single segment by time:
    python voiceprint_mgmt.py extract --audio meeting.mp4 --start 00:05:30 --end 00:06:15 --speaker "John" --output segments/

    # Reassign segments to a different speaker (e.g., fix misidentified speaker):
    python voiceprint_mgmt.py reassign segments/"Nico Bruhl" "NewSpeaker"

    # Refine voiceprint after reviewing/correcting segments:
    python voiceprint_mgmt.py refine --voiceprints voiceprints.json --speaker "SpeakerName" --segments segments/

    # Mass refine - process multiple speakers from folder structure:
    python voiceprint_mgmt.py mass_refine segments/ --voiceprints voiceprints.json
    python voiceprint_mgmt.py mass_refine segments/ --voiceprints voiceprints.json --skip-existing  # Skip existing

    # Create new voiceprint from audio segment:
    python voiceprint_mgmt.py create meeting.mp4 00:05:30 00:06:15 "John"

    # Add more samples to existing voiceprint:
    python voiceprint_mgmt.py add voiceprints.json "John" meeting1.mp4 00:05:30 00:06:15 meeting2.mp4 00:10:00 00:11:00
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from voiceprint_utils import (
    extract_speaker_segments,
    refine_voiceprint_from_segments,
    parse_time,
    format_time,
    ensure_wav,
    load_audio_segment,
    extract_embedding,
    compute_pitch,
    compute_energy,
    init_embedding_session,
    load_voiceprints,
    save_voiceprints,
)
from server import Settings


def format_time_short(seconds: float) -> str:
    """Format seconds as MM-SS."""
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}-{secs:02d}"


def generate_segment_hash(audio_path: str) -> str:
    """Generate a short 6-char alphanumeric hash from audio filename."""
    import hashlib
    # Hash only the filename (stem)
    key = Path(audio_path).stem
    hash_int = int(hashlib.md5(key.encode()).hexdigest(), 16)
    chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    result = []
    for _ in range(6):
        hash_int, idx = divmod(hash_int, 32)
        result.append(chars[idx])
    return "".join(result)


def cmd_extract(args):
    """Extract audio segments matching a speaker."""
    # Check for conflicting/invalid parameter combinations

    # --voiceprints is standalone mode
    if args.voiceprints:
        if args.diarize or args.start or args.end:
            print("[ERROR] Cannot combine --voiceprints with --diarize or --start/--end")
            sys.exit(1)
        if args.speaker:
            print("[ERROR] --speaker is not needed with --voiceprints (speakers are auto-identified)")
            sys.exit(1)
    # --diarize requires --speaker
    elif args.diarize:
        if not args.speaker:
            print("[ERROR] --speaker is required with --diarize")
            sys.exit(1)
        if args.start or args.end:
            print("[ERROR] Cannot combine --diarize with --start/--end")
            sys.exit(1)
    # --start/--end requires --speaker
    elif args.start or args.end:
        if not args.start or not args.end:
            print("[ERROR] Both --start and --end are required together")
            sys.exit(1)
        if not args.speaker:
            print("[ERROR] --speaker is required with --start/--end")
            sys.exit(1)
    else:
        print("[ERROR] Must specify one of: --voiceprints, --diarize, or --start/--end")
        sys.exit(1)

    if args.voiceprints:
        from voiceprint_utils import (
            load_voiceprints, init_embedding_session, ensure_wav, identify_speakers_in_audio
        )
        import soundfile as sf

        voiceprints = load_voiceprints(Path(args.voiceprints))
        if not voiceprints:
            print(f"[ERROR] No voiceprints found in {args.voiceprints}")
            sys.exit(1)

        print(f"[INFO] Loaded {len(voiceprints)} voiceprints: {list(voiceprints.keys())}")

        settings = Settings()
        embedding_session = init_embedding_session(settings)

        audio_path = Path(args.audio)
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        wav_path = ensure_wav(audio_path, output_dir)
        print(f"[INFO] Identifying speakers in {audio_path.name}...")

        segments = identify_speakers_in_audio(
            audio_path=str(wav_path),
            voiceprints=voiceprints,
            embedding_session=embedding_session,
            vad_threshold=args.vad_threshold or 0.5,
            vad_min_speech_ms=args.vad_min_speech or 250,
            match_threshold=args.match_threshold or 0.4,
            single_speaker_threshold=args.single_speaker_threshold or 0.8,
        )

        if wav_path != audio_path:
            wav_path.unlink()

        if not segments:
            print("[WARN] No speaker segments identified")
            print("  This may mean:")
            print("  - Audio has no clear speech")
            print("  - Speakers don't match existing voiceprints")
            print("  - VAD threshold too high (try --vad-threshold 0.3)")
            sys.exit(0)

        # Group by speaker
        speakers = {}
        for seg in segments:
            spk = seg["speaker"]
            if spk not in speakers:
                speakers[spk] = []
            speakers[spk].append(seg)

        # Identify unknown speakers (not in original voiceprints) and assign SPEAKER1, SPEAKER2, etc.
        known_speakers = set(voiceprints.keys())
        unknown_speakers = [s for s in speakers.keys() if s not in known_speakers]
        unknown_speakers.sort()

        speaker_rename = {}
        for i, spk in enumerate(unknown_speakers):
            speaker_rename[spk] = f"SPEAKER{i + 1}"

        if speaker_rename:
            print(f"[INFO] Unknown speakers found, assigning names: {list(speaker_rename.values())}")
            # Rename in segments
            for old_name, new_name in speaker_rename.items():
                if old_name in speakers:
                    speaker_rename[old_name] = new_name
                    speakers[new_name] = speakers.pop(old_name)

        print(f"[INFO] Identified {len(speakers)} speakers: {list(speakers.keys())}")

        # Save identification results to JSON for review
        review_file = output_dir / "identification_review.json"
        with open(review_file, "w", encoding="utf-8") as f:
            json.dump({
                "audio_file": str(audio_path),
                "voiceprints_used": list(voiceprints.keys()),
                "segments": segments,
                "speaker_summary": {
                    spk: {
                        "segments": len(segs),
                        "total_duration": round(sum(s["end"] - s["start"] for s in segs), 1),
                        "avg_confidence": round(sum(s["confidence"] for s in segs) / len(segs), 3)
                    }
                    for spk, segs in speakers.items()
                }
            }, f, indent=2)

        print(f"[INFO] Saved review file: {review_file}")

        # Extract segments for each speaker
        extracted_count = 0
        audio_hash = generate_segment_hash(str(audio_path))
        for speaker_name, speaker_segments in speakers.items():
            speaker_dir = output_dir / speaker_name
            speaker_dir.mkdir(exist_ok=True)

            # Save speaker segments info
            with open(speaker_dir / "segments.json", "w", encoding="utf-8") as f:
                json.dump(speaker_segments, f, indent=2)

            for i, seg in enumerate(speaker_segments):
                start = seg["start"]
                end = seg["end"]
                duration = end - start

                if duration < (args.min_duration or 1.5):
                    continue

                # Extract audio chunk
                conf = int(seg["confidence"] * 100)
                out_file = speaker_dir / f"{audio_hash}_{format_time_short(start)}_{conf:02d}_{duration:02.0f}.wav"
                cmd = [
                    "ffmpeg", "-y", "-i", str(audio_path),
                    "-ss", str(start), "-t", str(duration),
                    "-ar", "16000", "-ac", "1",
                    "-acodec", "pcm_s16le", "-loglevel", "error",
                    str(out_file)
                ]
                subprocess.run(cmd, check=True)
                extracted_count += 1

            print(f"  - {speaker_name}: {len(speaker_segments)} segments, {extracted_count} extracted to {speaker_dir}")

        print(f"[SUCCESS] Extracted {extracted_count} segments to {output_dir}")
        print(f"[INFO] Review {review_file} to verify speaker identification")
        print(f"[INFO] Remove incorrect segments from speaker folders, then run:")
        print(f"       python voiceprint_mgmt.py refine --voiceprints {args.voiceprints} --speaker <name> --segments {output_dir}/<speaker>/")
        return

    if args.diarize:
        segments = extract_speaker_segments(
            audio_file=args.audio,
            diarization_file=args.diarize,
            speaker_name=args.speaker,
            output_dir=args.output,
            min_duration=args.min_duration,
            min_confidence=args.min_confidence,
        )
    elif args.start and args.end:
        from voiceprint_utils import ensure_wav, load_audio_segment
        start_sec = parse_time(args.start)
        end_sec = parse_time(args.end)
        duration = end_sec - start_sec

        if duration < (args.min_duration or 1.5):
            print(f"[ERROR] Segment too short: {duration:.1f}s")
            sys.exit(1)

        audio_path = Path(args.audio)
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        wav_path = ensure_wav(audio_path, output_dir)
        waveform, sr = load_audio_segment(str(wav_path), start_sec, end_sec)

        out_file = output_dir / f"segment_{args.speaker}_{args.start}_{args.end}.wav"
        import soundfile as sf
        sf.write(str(out_file), waveform, sr)

        if wav_path != audio_path:
            wav_path.unlink()

        print(f"[SUCCESS] Extracted segment to {out_file}")
        return
    else:
        print("[ERROR] Provide either --voiceprints, --diarize, or --start/--end")
        sys.exit(1)

    if segments:
        print(f"[SUCCESS] Extracted {len(segments)} segments to {args.output}")
    else:
        print("[ERROR] No segments extracted")
        sys.exit(1)


def cmd_reassign(args):
    """Reassign segments from one speaker to another."""
    source_dir = Path(args.source)
    target_name = args.target
    output_dir = Path(args.output) if args.output else source_dir.parent

    if not source_dir.exists():
        print(f"[ERROR] Source directory not found: {source_dir}")
        sys.exit(1)

    # Find all wav files in source
    wav_files = sorted(source_dir.glob("*.wav"))
    if not wav_files:
        print(f"[ERROR] No .wav files found in {source_dir}")
        sys.exit(1)

    # Create target directory
    target_dir = output_dir / target_name
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Reassigning {len(wav_files)} segments to '{target_name}'")

    # Copy and rename files
    for i, wav in enumerate(wav_files):
        out_file = target_dir / f"{target_name}_{i+1:03d}.wav"
        import shutil
        shutil.copy2(wav, out_file)
        print(f"  - {wav.name} -> {out_file.name}")

    # Update/create segments.json
    segments_json = target_dir / "segments.json"
    segments_data = []
    for i, wav in enumerate(wav_files):
        # Try to extract timing from filename (e.g., segment_001_10.50_15.20.wav)
        name = wav.stem
        parts = name.split("_")
        start, end = None, None
        if len(parts) >= 3:
            try:
                start = float(parts[-2])
                end = float(parts[-1])
            except ValueError:
                pass

        segments_data.append({
            "file": f"{target_name}_{i+1:03d}.wav",
            "start": start,
            "end": end,
        })

    with open(segments_json, "w", encoding="utf-8") as f:
        json.dump(segments_data, f, indent=2)

    print(f"[SUCCESS] Reassigned {len(wav_files)} segments to {target_dir}")
    print(f"[INFO] To refine, run:")
    print(f"       python voiceprint_mgmt.py refine --voiceprints <path> --speaker {target_name} --segments {target_dir}/")


def cmd_refine(args):
    """Refine voiceprint from extracted segments."""
    voiceprints_file = Path(args.voiceprints)
    if not voiceprints_file.exists():
        print(f"[ERROR] Voiceprints file not found: {voiceprints_file}")
        sys.exit(1)

    segments_dir = Path(args.segments)
    if not segments_dir.exists():
        print(f"[ERROR] Segments directory not found: {segments_dir}")
        sys.exit(1)

    result = refine_voiceprint_from_segments(
        voiceprints_file=voiceprints_file,
        speaker_name=args.speaker,
        segments_dir=segments_dir,
        min_duration=args.min_duration,
    )

    print(f"[SUCCESS] Refined voiceprint for '{args.speaker}'")


def cmd_create(args):
    """Create a new voiceprint from audio segment."""
    from server import state, ensure_embedding_model
    import onnxruntime as ort

    start_sec = parse_time(args.start)
    end_sec = parse_time(args.end)

    if end_sec <= start_sec:
        print("[ERROR] End time must be after start time")
        sys.exit(1)

    duration = end_sec - start_sec
    if duration < 1.5:
        print(f"[ERROR] Segment must be at least 1.5 seconds (got {duration:.1f}s)")
        sys.exit(1)

    audio_path = Path(args.file)
    if not audio_path.exists():
        print(f"[ERROR] File not found: {audio_path}")
        sys.exit(1)

    print(f"[INFO] Loading {audio_path}...")

    settings = Settings()
    embedding_session = init_embedding_session(settings)

    wav_path = ensure_wav(audio_path)
    print(f"[INFO] Loading segment: {format_time(start_sec)} - {format_time(end_sec)} ({duration:.1f}s)")
    waveform, sample_rate = load_audio_segment(str(wav_path), start_sec, end_sec)

    print(f"[INFO] Extracting embedding...")
    embedding = extract_embedding(waveform, sample_rate, embedding_session)

    print(f"[INFO] Computing features...")
    pitch_result = compute_pitch(waveform, sample_rate)
    if isinstance(pitch_result, tuple):
        pitch, pitch_std = pitch_result
    else:
        pitch, pitch_std = pitch_result, 0.0
    energy = compute_energy(waveform)

    voiceprints = load_voiceprints(Path(args.output))

    if args.name in voiceprints and not args.overwrite:
        print(f"[ERROR] Speaker '{args.name}' exists. Use --overwrite to replace.")
        if wav_path != audio_path:
            wav_path.unlink()
        sys.exit(1)

    voiceprints[args.name] = {
        "pitch_hz": round(pitch, 1) if pitch > 0 else 0.0,
        "pitch_std": round(pitch_std, 1),
        "energy_rms": round(energy, 4),
        "total_speech_sec": round(duration, 1),
        "embedding": embedding,
    }

    save_voiceprints(voiceprints, Path(args.output))

    print(f"[SUCCESS] Saved voiceprint for '{args.name}' to {args.output}")
    print(f"  - Duration: {duration:.1f}s, Pitch: {pitch:.1f}Hz")

    if wav_path != audio_path and wav_path.exists():
        wav_path.unlink()


def cmd_add(args):
    """Add/r
efine voiceprint with additional samples."""
    from server import state, ensure_embedding_model
    import onnxruntime as ort
    import soundfile as sf

    voiceprints_path = Path(args.voiceprints)
    if not voiceprints_path.exists():
        print(f"[ERROR] Voiceprints file not found: {args.voiceprints}")
        sys.exit(1)

    voiceprints = load_voiceprints(voiceprints_path)

    if args.speaker not in voiceprints:
        print(f"[ERROR] Speaker '{args.speaker}' not found")
        print(f"  Available: {list(voiceprints.keys())}")
        sys.exit(1)

    existing = voiceprints[args.speaker]
    print(f"[INFO] Existing voiceprint for '{args.speaker}':")
    print(f"  - Duration: {existing.get('total_speech_sec', 0):.1f}s")
    print(f"  - Pitch: {existing.get('pitch_hz', 0):.1f} Hz")

    settings = Settings()
    embedding_session = init_embedding_session(settings)

    all_embeddings = []
    all_pitches = []
    all_energies = []
    total_duration = 0.0

    i = 0
    segments = args.segments
    while i < len(segments):
        file_path = Path(segments[i])
        start_sec = parse_time(segments[i + 1])
        end_sec = parse_time(segments[i + 2])
        i += 3

        if not file_path.exists():
            print(f"[WARN] Skipping missing file: {file_path}")
            continue

        duration = end_sec - start_sec
        if duration < 1.5:
            print(f"[WARN] Skipping short segment ({duration:.1f}s)")
            continue

        print(f"[INFO] Processing: {file_path.name} {format_time(start_sec)}-{format_time(end_sec)}")

        try:
            wav_path = ensure_wav(file_path)
            waveform, sr = load_audio_segment(str(wav_path), start_sec, end_sec)

            emb = extract_embedding(waveform, sr, embedding_session)
            pitch, pitch_std = compute_pitch(waveform, sr)
            energy = compute_energy(waveform)

            all_embeddings.append(np.array(emb))
            if pitch > 0:
                all_pitches.append(pitch)
            all_energies.append(energy)
            total_duration += duration

            print(f"  - OK: pitch={pitch:.1f}Hz")

            if wav_path != file_path and wav_path.exists():
                wav_path.unlink()

        except Exception as e:
            print(f"[WARN] Failed: {e}")
            continue

    if not all_embeddings:
        print("[ERROR] No valid segments processed")
        sys.exit(1)

    print(f"[INFO] Processed: {total_duration:.1f}s from {len(all_embeddings)} segments")

    combined_emb = np.mean(np.stack(all_embeddings), axis=0)
    combined_emb = combined_emb / (np.linalg.norm(combined_emb) + 1e-12)
    combined_emb = combined_emb.tolist()

    avg_pitch = np.mean(all_pitches) if all_pitches else 0.0
    pitch_std_new = np.std(all_pitches) if len(all_pitches) > 1 else 0.0
    avg_energy = np.mean(all_energies)
    new_duration = existing.get("total_speech_sec", 0) + total_duration

    new_pitch = avg_pitch if avg_pitch > 0 else existing.get("pitch_hz", 0)
    new_pitch_std = pitch_std_new if pitch_std_new > 0 else existing.get("pitch_std", 0)
    new_energy = (existing.get("energy_rms", 0) + avg_energy) / 2

    voiceprints[args.speaker] = {
        "pitch_hz": round(new_pitch, 1),
        "pitch_std": round(new_pitch_std, 1),
        "energy_rms": round(new_energy, 4),
        "total_speech_sec": round(new_duration, 1),
        "gender_hint": existing.get("gender_hint", "unknown"),
        "embedding": combined_emb,
    }

    save_voiceprints(voiceprints, voiceprints_path)

    print(f"[SUCCESS] Updated voiceprint for '{args.speaker}'")
    print(f"  - Total: {new_duration:.1f}s (was {existing.get('total_speech_sec', 0):.1f}s)")


def cmd_add(args):
    """Add/r
efine voiceprint with additional samples."""
    from server import state, ensure_embedding_model
    import onnxruntime as ort
    import soundfile as sf

    voiceprints_path = Path(args.voiceprints)
    if not voiceprints_path.exists():
        print(f"[ERROR] Voiceprints file not found: {args.voiceprints}")
        sys.exit(1)

    voiceprints = load_voiceprints(voiceprints_path)

    if args.speaker not in voiceprints:
        print(f"[ERROR] Speaker '{args.speaker}' not found")
        print(f"  Available: {list(voiceprints.keys())}")
        sys.exit(1)

    existing = voiceprints[args.speaker]
    print(f"[INFO] Existing voiceprint for '{args.speaker}':")
    print(f"  - Duration: {existing.get('total_speech_sec', 0):.1f}s")
    print(f"  - Pitch: {existing.get('pitch_hz', 0):.1f} Hz")

    settings = Settings()
    embedding_session = init_embedding_session(settings)

    all_embeddings = []
    all_pitches = []
    all_energies = []
    total_duration = 0.0

    i = 0
    segments = args.segments
    while i < len(segments):
        file_path = Path(segments[i])
        start_sec = parse_time(segments[i + 1])
        end_sec = parse_time(segments[i + 2])
        i += 3

        if not file_path.exists():
            print(f"[WARN] Skipping missing file: {file_path}")
            continue

        duration = end_sec - start_sec
        if duration < 1.5:
            print(f"[WARN] Skipping short segment ({duration:.1f}s)")
            continue

        print(f"[INFO] Processing: {file_path.name} {format_time(start_sec)}-{format_time(end_sec)}")

        try:
            wav_path = ensure_wav(file_path)
            waveform, sr = load_audio_segment(str(wav_path), start_sec, end_sec)

            emb = extract_embedding(waveform, sr, embedding_session)
            pitch, pitch_std = compute_pitch(waveform, sr)
            energy = compute_energy(waveform)

            all_embeddings.append(np.array(emb))
            if pitch > 0:
                all_pitches.append(pitch)
            all_energies.append(energy)
            total_duration += duration

            print(f"  - OK: pitch={pitch:.1f}Hz")

            if wav_path != file_path and wav_path.exists():
                wav_path.unlink()

        except Exception as e:
            print(f"[WARN] Failed: {e}")
            continue

    if not all_embeddings:
        print("[ERROR] No valid segments processed")
        sys.exit(1)

    print(f"[INFO] Processed: {total_duration:.1f}s from {len(all_embeddings)} segments")

    combined_emb = np.mean(np.stack(all_embeddings), axis=0)
    combined_emb = combined_emb / (np.linalg.norm(combined_emb) + 1e-12)
    combined_emb = combined_emb.tolist()

    avg_pitch = np.mean(all_pitches) if all_pitches else 0.0
    pitch_std_new = np.std(all_pitches) if len(all_pitches) > 1 else 0.0
    avg_energy = np.mean(all_energies)
    new_duration = existing.get("total_speech_sec", 0) + total_duration

    new_pitch = avg_pitch if avg_pitch > 0 else existing.get("pitch_hz", 0)
    new_pitch_std = pitch_std_new if pitch_std_new > 0 else existing.get("pitch_std", 0)
    new_energy = (existing.get("energy_rms", 0) + avg_energy) / 2

    voiceprints[args.speaker] = {
        "pitch_hz": round(new_pitch, 1),
        "pitch_std": round(new_pitch_std, 1),
        "energy_rms": round(new_energy, 4),
        "total_speech_sec": round(new_duration, 1),
        "gender_hint": existing.get("gender_hint", "unknown"),
        "embedding": combined_emb,
    }

    save_voiceprints(voiceprints, voiceprints_path)

    print(f"[SUCCESS] Updated voiceprint for '{args.speaker}'")
    print(f"  - Total: {new_duration:.1f}s (was {existing.get('total_speech_sec', 0):.1f}s)")


def cmd_mass_refine(args):
    """Refine voiceprints from a folder of speaker subfolders."""
    voiceprints_file = Path(args.voiceprints)
    if not voiceprints_file.exists():
        print(f"[ERROR] Voiceprints file not found: {voiceprints_file}")
        sys.exit(1)

    root_dir = Path(args.folder)
    if not root_dir.exists():
        print(f"[ERROR] Folder not found: {root_dir}")
        sys.exit(1)

    # Find all subdirectories (speaker folders)
    speaker_dirs = []
    for item in root_dir.iterdir():
        if item.is_dir():
            speaker_dirs.append(item)

    if not speaker_dirs:
        print(f"[ERROR] No speaker subfolders found in {root_dir}")
        sys.exit(1)

    print(f"[INFO] Found {len(speaker_dirs)} speaker folders")
    print(f"[INFO] Voiceprints file: {voiceprints_file}")
    
    if args.skip_existing:
        print(f"[INFO] Will skip speakers who already have voiceprints")
    else:
        print(f"[INFO] Will re-refine speakers who already have voiceprints (may double-count segments)")

    # Show preview
    print(f"\n[INFO] Speakers to process:")
    for sd in sorted(speaker_dirs):
        wavs = list(sd.glob("*.wav"))
        dur_str = f"{len(wavs)} segments" if wavs else "NO WAVS"
        print(f"  - {sd.name}: {dur_str}")

    if not args.skip_confirm:
        resp = input("\nProceed? [y/N]: ")
        if resp.lower() != "y":
            print("Cancelled")
            sys.exit(0)

    # Process each speaker folder
    success_count = 0
    error_count = 0
    
    for speaker_dir in sorted(speaker_dirs):
        speaker_name = speaker_dir.name
        wavs = list(speaker_dir.glob("*.wav"))
        
        if not wavs:
            print(f"[WARN] No .wav files in {speaker_name}, skipping")
            error_count += 1
            continue

        # Check if speaker exists in voiceprints
        voiceprints = load_voiceprints(voiceprints_file)
        if args.skip_existing and speaker_name in voiceprints:
            print(f"[SKIP] {speaker_name} already has voiceprint (use without --skip-existing to re-refine)")
            continue

        print(f"\n[INFO] Processing {speaker_name} ({len(wavs)} segments)...")
        
        try:
            result = refine_voiceprint_from_segments(
                voiceprints_file=voiceprints_file,
                speaker_name=speaker_name,
                segments_dir=speaker_dir,
                min_duration=args.min_duration,
            )
            print(f"[OK] {speaker_name}: total {result.get('total_speech_sec', 0):.1f}s")
            success_count += 1
        except Exception as e:
            print(f"[ERROR] Failed for {speaker_name}: {e}")
            error_count += 1

    print(f"\n[SUCCESS] Summary: {success_count} succeeded, {error_count} errors")


def main():
    parser = argparse.ArgumentParser(
        description="Voiceprint management utilities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # extract subcommand
    p_extract = subparsers.add_parser("extract", help="Extract segments for a speaker")
    p_extract.add_argument("--audio", required=True, help="Audio/video file")
    p_extract.add_argument("--voiceprints", help="Use existing voiceprints to identify speakers in audio")
    p_extract.add_argument("--diarize", help="Diarization JSON file (or use --start/--end)")
    p_extract.add_argument("--speaker", help="Speaker name/ID (required for --diarize and --start/--end)")
    p_extract.add_argument("--output", required=True, help="Output directory")
    p_extract.add_argument("--min-duration", type=float, default=1.5, help="Minimum segment duration")
    p_extract.add_argument("--start", help="Start time (if extracting single segment without diarization)")
    p_extract.add_argument("--end", help="End time (if extracting single segment without diarization)")
    p_extract.add_argument("--vad-threshold", type=float, default=0.5, help="VAD threshold (0.0-1.0)")
    p_extract.add_argument("--vad-min-speech", type=int, default=250, help="VAD min speech duration (ms)")
    p_extract.add_argument("--match-threshold", type=float, default=0.4, help="Voiceprint match threshold (cosine distance)")
    p_extract.add_argument("--single-speaker-threshold", type=float, default=0.8, help="Min ratio of windows matching dominant speaker (0.0-1.0)")
    p_extract.add_argument("--min-confidence", type=float, default=0.0, help="Minimum confidence threshold (0-1). Lower quality segments are skipped.")
    p_extract.set_defaults(func=cmd_extract)

    # reassign subcommand
    p_reassign = subparsers.add_parser("reassign", help="Reassign segments to a different speaker")
    p_reassign.add_argument("source", help="Source folder containing segments (e.g., segments/Nico Bruhl/)")
    p_reassign.add_argument("target", help="Target speaker name (existing or new)")
    p_reassign.add_argument("--output", help="Output directory (default: same as source parent)")
    p_reassign.set_defaults(func=cmd_reassign)

    # refine subcommand
    p_refine = subparsers.add_parser("refine", help="Refine voiceprint from segments")
    p_refine.add_argument("--voiceprints", required=True, help="Voiceprints JSON file")
    p_refine.add_argument("--speaker", required=True, help="Speaker name")
    p_refine.add_argument("--segments", required=True, help="Segments directory")
    p_refine.add_argument("--min-duration", type=float, default=1.5, help="Minimum segment duration")
    p_refine.set_defaults(func=cmd_refine)

    # mass_refine subcommand (NEW)
    p_mass = subparsers.add_parser("mass_refine", help="Refine multiple voiceprints from folder structure")
    p_mass.add_argument("folder", help="Root folder containing speaker subfolders (e.g., segments/)")
    p_mass.add_argument("--voiceprints", default="voiceprints.json", help="Voiceprints JSON file")
    p_mass.add_argument("--min-duration", type=float, default=1.5, help="Minimum segment duration")
    p_mass.add_argument("--skip-existing", action="store_true", help="Skip speakers who already have voiceprints")
    p_mass.add_argument("--skip-confirm", action="store_true", help="Skip confirmation prompt")
    p_mass.set_defaults(func=cmd_mass_refine)

    # create subcommand
    p_create = subparsers.add_parser("create", help="Create voiceprint from segment")
    p_create.add_argument("file", help="Audio/video file")
    p_create.add_argument("start", help="Start time (HH:MM:SS, MM:SS, or seconds)")
    p_create.add_argument("end", help="End time (HH:MM:SS, MM:SS, or seconds)")
    p_create.add_argument("name", help="Speaker name")
    p_create.add_argument("--output", default="voiceprints.json", help="Output file")
    p_create.add_argument("--overwrite", action="store_true", help="Overwrite existing")
    p_create.set_defaults(func=cmd_create)

    # add subcommand
    p_add = subparsers.add_parser("add", help="Add samples to existing voiceprint")
    p_add.add_argument("voiceprints", help="Voiceprints JSON file")
    p_add.add_argument("speaker", help="Speaker name")
    p_add.add_argument("segments", nargs="+", help="Pattern: file start end [file start end ...]")
    p_add.set_defaults(func=cmd_add)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()