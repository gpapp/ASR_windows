# Custom ONNX Diarization Pipeline

## Objective
Replace Pyannote with a custom, incredibly fast CPU-bound diarization pipeline using Silero VAD, a WeSpeaker ONNX embedding model, and scikit-learn Agglomerative Clustering. This design ensures maximum throughput by keeping inference purely ONNX-based and utilizing batching.

## 1. Architecture & Model Setup
- **Dependencies**: Remove pyannote dependencies. Add `scikit-learn` and `torchaudio`.
- **Model Files**:
  - Repo: `onnx-community/wespeaker-voxceleb-resnet34-LM`
  - Filename: `onnx/model.onnx`
- **State Initialization**: Load the ONNX embedding model inside `ModelState` using `ort.InferenceSession`. Remove `state.diarization_pipeline` completely. The session will be stored in `state.embedding_session`. Silero VAD is maintained.

## 2. Audio Chunking & Feature Extraction
- **VAD Processing**: Pass the audio through Silero VAD to obtain contiguous speech segments.
- **Sliding Windows**: Iterate over VAD segments. Split each segment into 1.5-second windows with a 0.75-second stride (50% overlap). If a segment is shorter than 1.5s, pad it or take it as a single chunk.
- **Filterbank Extraction**: For each window, compute 80-dimensional log-mel filterbanks using `torchaudio.compliance.kaldi.fbank(waveform, num_mel_bins=80, frame_length=25, frame_shift=10)`. Ensure the shape matches `[Time, 80]`.
- **Batching**: Stack the filterbanks into batches of size 32.

## 3. Embedding, Clustering, and Merging
- **Embedding Extraction**: Pass batches of shape `[B, T, 80]` to `state.embedding_session` (input name `input_features`). The output from `last_hidden_state` (shape `[B, 256]`) will be the embeddings.
- **L2 Normalization**: L2-normalize the generated embeddings to improve clustering quality.
- **Clustering**:
  - Use `sklearn.cluster.AgglomerativeClustering` with `metric="cosine"`, `linkage="average"`.
  - Use `distance_threshold=settings.diarization_threshold` (default e.g. 0.5) and `n_clusters=None`.
- **Timestamp Merging**:
  - Assign each 0.75-second chunk to the speaker of its center window.
  - Merge consecutive chunks with the same speaker label into contiguous segments.
- **Response Format**: Send NDJSON stream updates containing `{ "start": start, "end": end, "speaker": f"SPEAKER_{cluster_id:02d}" }`.

## 4. API Adjustments
- Remove all references to Pyannote hooks. Send custom NDJSON stream updates for VAD, Feature Extraction, Embedding, and Clustering phases manually within the threaded diarization function.