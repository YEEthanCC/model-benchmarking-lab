import azure.cognitiveservices.speech as speechsdk
import json
import time

from core.config import Settings

settings = Settings()

class TranscriptDetail:
    def __init__(self, word: str, start_ms: int, duration_ms: int, end_ms: int):
        self.word = word
        self.start_ms = start_ms
        self.duration_ms = duration_ms
        self.end_ms = end_ms

class Transcript:
    def __init__(self, text: str, details: list[TranscriptDetail]):
        self.text = text
        self.details = details

    def addText(self, text: str):
        self.text+=text

    def addDetail(self, detail: TranscriptDetail):
        self.details.append(detail)

class AzureTranscriber:

    def __init__(self, key: str, region: str, language: str = "en-US"):
        """
        Args:
            key:      Azure Speech resource key (Key 1 from Azure Portal)
            region:   Azure region, e.g. "eastus", "westeurope"
            language: BCP-47 locale for recognition, e.g. "en-US", "pt-BR"
        """
        self.key      = key
        self.region   = region
        self.language = language
        self.text = ""
        self.words: list[TranscriptDetail] = []  # populated after transcribe()

    # ── Public API ─────────────────────────────────────────────────────────────

    def transcribe(self, audio_file: str) -> Transcript:
        """
        Transcribe an MP3 file and return word-level timestamps.

        Returns a list of dicts:
            { "word": str, "start_ms": int, "duration_ms": int, "end_ms": int }
        """
        self.words = []
        self.text = ""
        self._audio_file = audio_file

        speech_config = self._build_speech_config()
        audio_config  = self._build_audio_config(audio_file)
        recognizer    = speechsdk.SpeechRecognizer(
            speech_config=speech_config,
            audio_config=audio_config
        )

        self._run_recognition(recognizer)
        return Transcript(self.text, self.words)

    def save_json(self, path: str) -> None:
        """Save the word list to a JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.words, f, indent=2, ensure_ascii=False)
        print(f"💾 Saved to: {path}")

    def to_plain_text(self) -> str:
        """Return a simple space-joined transcript string."""
        return " ".join(w["word"] for w in self.words)

    # ── Private helpers ────────────────────────────────────────────────────────

    def _build_speech_config(self) -> speechsdk.SpeechConfig:
        cfg = speechsdk.SpeechConfig(subscription=self.key, region=self.region)
        cfg.request_word_level_timestamps()
        cfg.speech_recognition_language = self.language
        cfg.output_format = speechsdk.OutputFormat.Detailed
        return cfg

    def _build_audio_config(self, audio_file: str) -> speechsdk.audio.AudioConfig:
        audio_format = speechsdk.audio.AudioStreamFormat(
            compressed_stream_format=speechsdk.AudioStreamContainerFormat.MP3
        )
        stream = speechsdk.audio.PushAudioInputStream(stream_format=audio_format)
        with open(audio_file, "rb") as f:
            stream.write(f.read())
        stream.close()
        return speechsdk.audio.AudioConfig(stream=stream)

    def _run_recognition(self, recognizer: speechsdk.SpeechRecognizer) -> None:
        done = False

        def on_recognized(evt):
            result = evt.result
            if result.reason == speechsdk.ResultReason.RecognizedSpeech:
                detail = json.loads(result.json)
                print(f'\n📝 Segment: "{result.text}"')
                self.text+=f'\n📝 Segment: "{result.text}"'
                nbest = detail.get("NBest", [])
                if nbest:
                    for w in nbest[0].get("Words", []):
                        self.words.append(TranscriptDetail(w["Word"], w["Offset"] // 10000, w["Duration"] // 10000, (w["Offset"] + w["Duration"]) // 10000))
                        # entry = {
                        #     "word":        w["Word"],
                        #     "start_ms":    w["Offset"] // 10000,
                        #     "duration_ms": w["Duration"] // 10000,
                        #     "end_ms":      (w["Offset"] + w["Duration"]) // 10000,
                        # }
                        # self.words.append(entry)

        def on_canceled(evt):
            nonlocal done
            details = evt.cancellation_details
            if details.reason == speechsdk.CancellationReason.Error:
                print(f"❌ Error: {details.error_details}")
            done = True

        def on_stopped(evt):
            nonlocal done
            done = True

        recognizer.recognized.connect(on_recognized)
        recognizer.session_stopped.connect(on_stopped)
        recognizer.canceled.connect(on_canceled)

        print(f"🎙  Transcribing: {self._audio_file}\n{'─' * 50}")
        recognizer.start_continuous_recognition()
        while not done:
            time.sleep(0.5)
        recognizer.stop_continuous_recognition()

