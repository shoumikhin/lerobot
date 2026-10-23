# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Turn spoken commands into tasks for `lerobot-rollout --interactive=true`, with Whisper in ExecuTorch.

It listens to a microphone, cuts each utterance at a pause, transcribes it on the board, and prints the
rollout's own commands: `/subtask <what was said>` then `/start`, or `/reset` for "stop". Pipe it into
the rollout:

    python examples/export/voice_task.py --model=outputs/whisper/model.pte \\
        --preprocessor=outputs/whisper/whisper_preprocessor.pte --tokenizer=outputs/whisper/tokenizer.json \\
        | lerobot-rollout --policy.path=outputs/export/pi05 --interactive=true ...

Export Whisper with Optimum ExecuTorch, for the GPU (it writes a `.ptd` weights file next to the program,
which is loaded with it):

    optimum-cli export executorch --model openai/whisper-tiny --task automatic-speech-recognition \\
        --recipe cuda --dtype bfloat16 --device cuda --output_dir outputs/whisper

and the preprocessor that turns audio into Whisper's input, in the same precision:

    python examples/export/voice_task.py --export_preprocessor=outputs/whisper/whisper_preprocessor.pte \\
        --bfloat16

Optimum ExecuTorch 1.1.0's CUDA export computes Whisper's cross-attention once per loaded program, so every
command after the first is heard as the first one; export with a version that recomputes it at the start of
each command (at decoder position 0). The XNNPACK export does not have this problem.

Pass `--wav` files instead of a microphone to try it without one.
"""

import argparse
import os
import subprocess
import sys
import time
import wave
from pathlib import Path

import numpy as np

os.environ["EXECUTORCH_PYBINDINGS_TENSOR_OUTPUT"] = "executorch"

SAMPLE_RATE = 16000
# Whisper's special tokens: start of transcript, English, transcribe, no timestamps, end of text.
PROMPT = [50258, 50259, 50359, 50363]
END_OF_TEXT = 50257
STOP_WORDS = {"stop", "stop.", "stop!"}


class Whisper:
    """A Whisper program with `encoder`, `text_decoder` and `sampler` methods, from Optimum ExecuTorch."""

    def __init__(self, model: Path, preprocessor: Path, tokenizer: Path, max_tokens: int = 64):
        from executorch.runtime import Runtime
        from tokenizers import Tokenizer

        runtime = Runtime.get()
        self._mel = runtime.load_program(preprocessor).load_method("forward")
        program = runtime.load_program(model, data_path=next(model.parent.glob("*.ptd"), None))
        self._encoder = program.load_method("encoder")
        self._decoder = program.load_method("text_decoder")
        self._sampler = program.load_method("sampler")
        self._tokenizer = Tokenizer.from_file(str(tokenizer))
        self._max_tokens = max_tokens

    def __call__(self, audio: np.ndarray) -> str:
        """The English text of 16 kHz mono `audio`, greedy decoding."""
        # Each method's result goes into the next as it is: it may be bfloat16, which NumPy cannot hold.
        # A CUDA program takes CPU inputs and copies them to the GPU itself.
        (mel,) = self._mel.execute([audio])
        (hidden,) = self._encoder.execute([mel])
        token, position, text = PROMPT[0], 0, []
        while position < self._max_tokens:
            (logits,) = self._decoder.execute([np.array([[token]]), hidden, np.array([position])])
            position += 1
            if position < len(PROMPT):
                token = PROMPT[position]
                continue
            (best,) = self._sampler.execute([logits])
            token = int(np.asarray(best).item())
            if token == END_OF_TEXT:
                break
            text.append(token)
        return self._tokenizer.decode(text, skip_special_tokens=True).strip()


def utterances(device: str, threshold: float, pause: float, longest: float):
    """Each stretch of speech from the microphone, cut at `pause` seconds of quiet, as float32 audio."""
    block = SAMPLE_RATE // 20
    recorder = subprocess.Popen(  # nosec B603 B607
        ["arecord", "-q", "-D", device, "-f", "S16_LE", "-r", str(SAMPLE_RATE), "-c", "1", "-t", "raw"],
        stdout=subprocess.PIPE,
    )
    speech, quiet = [], 0.0
    while chunk := recorder.stdout.read(2 * block):
        audio = np.frombuffer(chunk, np.int16).astype(np.float32) / 32768
        loud = np.sqrt(np.mean(audio**2)) > threshold
        if speech or loud:
            speech.append(audio)
            quiet = 0.0 if loud else quiet + len(audio) / SAMPLE_RATE
            if quiet >= pause or len(speech) * block >= longest * SAMPLE_RATE:
                yield np.concatenate(speech)
                speech, quiet = [], 0.0


def export_preprocessor(path: Path, bfloat16: bool) -> None:
    """Export ExecuTorch's Whisper preprocessor for up to 30 s of audio, its output in bfloat16 if asked.

    A bfloat16 Whisper takes bfloat16 features, which NumPy cannot pass in, so the preprocessor makes them.
    """
    import torch
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
    from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
    from executorch.extension.audio.mel_spectrogram import WhisperAudioProcessor

    class Preprocessor(WhisperAudioProcessor):
        def forward(self, waveform: torch.Tensor) -> torch.Tensor:
            features = super().forward(waveform)
            return features.to(torch.bfloat16) if bfloat16 else features

    module = Preprocessor(feature_size=80, max_audio_len=30, stack_output=True)
    audio = torch.zeros(SAMPLE_RATE)
    longest = torch.export.Dim("samples", max=30 * SAMPLE_RATE)
    with torch.no_grad():
        program = torch.export.export(module, (audio,), dynamic_shapes=({0: longest},), strict=True)
    # Its spectrogram mixes dtypes the edge dialect check rejects; ExecuTorch exports it the same way.
    config = EdgeCompileConfig(_check_ir_validity=False)
    lowered = to_edge_transform_and_lower(
        program, partitioner=[XnnpackPartitioner()], compile_config=config
    ).to_executorch()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(lowered.buffer)


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path)) as file:
        if (file.getframerate(), file.getnchannels(), file.getsampwidth()) != (SAMPLE_RATE, 1, 2):
            raise SystemExit(f"{path}: expected 16 kHz mono 16-bit audio.")
        return np.frombuffer(file.readframes(file.getnframes()), np.int16).astype(np.float32) / 32768


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", type=Path, help="Whisper .pte from Optimum ExecuTorch.")
    parser.add_argument("--preprocessor", type=Path, help="The preprocessor .pte.")
    parser.add_argument("--tokenizer", type=Path, help="The Whisper model's tokenizer.json.")
    parser.add_argument("--device", default="default", help="ALSA capture device, as `arecord -L` lists.")
    parser.add_argument("--threshold", type=float, default=0.02, help="Loudness that counts as speech.")
    parser.add_argument("--pause", type=float, default=0.6, help="Seconds of quiet that end a command.")
    parser.add_argument("--wav", type=Path, nargs="*", help="Transcribe these files instead of listening.")
    parser.add_argument("--export_preprocessor", type=Path, help="Instead, export the preprocessor here.")
    parser.add_argument("--bfloat16", action="store_true", help="For a bfloat16 Whisper.")
    args = parser.parse_args()
    if args.export_preprocessor:
        export_preprocessor(args.export_preprocessor, args.bfloat16)
        return
    if not (args.model and args.preprocessor and args.tokenizer):
        parser.error("--model, --preprocessor and --tokenizer are required.")

    whisper = Whisper(args.model, args.preprocessor, args.tokenizer)
    if args.wav:
        sources = (read_wav(path) for path in args.wav)
    else:
        sources = utterances(args.device, args.threshold, args.pause, longest=15)
    for audio in sources:
        start = time.perf_counter()
        text = whisper(audio)
        print(f"heard {text!r} in {(time.perf_counter() - start) * 1000:.0f} ms", file=sys.stderr, flush=True)
        if not text:
            continue
        if text.lower() in STOP_WORDS:
            print("/reset", flush=True)
        else:
            print(f"/subtask {text}", flush=True)
            print("/start", flush=True)


if __name__ == "__main__":
    main()
