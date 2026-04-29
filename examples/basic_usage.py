"""Basic usage examples for Modern ASR."""

from modern_asr import ASRPipeline, list_models

# 1. List all available models
print("=== Available Models ===")
for m in list_models():
    print(f"  {m['model_id']} -> {m['class']}")

# 2. Transcribe with SenseVoice (fast, supports emotion/events)
print("\n=== SenseVoice-Small ===")
pipe = ASRPipeline("sensevoice-small")
# result = pipe("examples/audio_zh.wav")
# print(result.text)

# 3. Switch to FireRedASR-LLM (highest Mandarin accuracy)
print("\n=== FireRedASR-LLM ===")
pipe.switch_model("fireredasr-llm")
# result = pipe("examples/audio_zh.wav", language="zh")
# print(result.text)

# 4. Use Whisper for multilingual
print("\n=== Whisper Large V3 ===")
pipe.switch_model("whisper-large-v3")
# result = pipe("examples/audio_multi.wav", language="auto")
# print(result.text)

# 5. Use MiMo for complex scenarios (dialects, noise, multi-speaker)
print("\n=== MiMo-V2.5-ASR ===")
pipe.switch_model("mimo-asr-v2.5")
# result = pipe("examples/meeting.wav")
# print(result.text)

print("\nDone! Uncomment the transcription lines and provide audio files to test.")
