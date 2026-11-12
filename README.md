# Nirzevis

Generate word-timed, Bilingual and enhanced Subtitle (Zirnevis) in various formats including .SRT, .LRC,  .TTML and .ASS

## Setup

- Install [Python 3.9+](https://www.python.org/) (no pip packages required!).
- Get latest [whisper.cpp build](https://github.com/ggml-org/whisper.cpp/releases) or build it from [source](https://github.com/ggml-org/whisper.cpp) and Download a compatible [model](https://huggingface.co/ggerganov/whisper.cpp) for it.
- Prepare an OpenAI-compatible LLM API for translation (optional)
- Install [ffmpeg](https://ffmpeg.org/) and put it in path of os if you want to generate Subtitle for videos (optional)

Edit `.env` beside the script: set `WHISPER_CLI`, `MODEL_PATH`, and if you want to translation, set your LLM API settings.
If `.env` is missing, copy `.env.example` to `.env`. Keep API keys private.
Match `DTW_MODEL` to your Whisper model. Blank `OPENAI_MODEL` uses `/models` detection.

## Run

```bash
python nirzevis.py "audio.mp3"
```

You can override any setting with an option:

```bash
python nirzevis.py "audio.mp3" --llm-model "my-model" --translation-batch-size 4
python nirzevis.py "audio.mp3" --output-dir ./results
python nirzevis.py "audio.mp3" --env-file ./custom.env
```

Resume translation using existing JSON (use the same output directory):

```bash
python nirzevis.py "audio.mp3" --skip-transcription
```

```bash
python nirzevis.py "audio.mp3" --skip-translation
python nirzevis.py "audio.mp3" --skip-transcription --refresh-translations
python nirzevis.py --help
```

## Translation language

Persian (`fa`) is the default. Set `TARGET_LANGUAGE` in `.env`:

```dotenv
TARGET_LANGUAGE=es
```

Or override it for one run:

```bash
python nirzevis.py "audio.mp3" --target-language es
```

Examples: `fa` Persian, `es` Spanish, `fr` French, `de` German,
`ar` Arabic, `pt-BR` Brazilian Portuguese.

Changing the target language invalidates the translation cache.
Use `--skip-transcription` to reuse existing Whisper JSON.
Translation quality depends on the selected LLM.

## Settings and output

Priority: command-line options > existing environment > `.env` > defaults.
`.env` uses literal `KEY=value` entries, optional quotes and comments; no variable expansion.
Paths are relative to your current working directory. `AUDIO_FILE` can replace the audio argument.
Each setting in `.env.example` has a command-line option listed by `--help`.
Boolean options also accept `--no-...` to override an enabled environment setting.

Outputs go beside the audio unless `--output-dir` is set:
`.srt`, `.json`, `.lrc`, `.ttml`, `.bilingual.lrc`, and `.translations.json`.
SRT remains untranslated. Persian translation is sentence-timed; English words are individually timed.
Player support for bilingual formats varies. Existing outputs with the same audio stem are overwritten.

Successful translations are saved after each request. Changed source text, endpoint or explicit model
invalidates the cache. When using model auto-detection, use `--refresh-translations` if the server's
model changes. Old-script caches are ignored. Do not run multiple jobs with the same output stem.

Exit codes: `0` success, `1` runtime error, `2` invalid arguments or incomplete translation, `130` interrupted.
Incomplete translations still produce lyric files; details go in `.failed_translations.json`.
Retry with `--skip-transcription`; use a smaller batch if the model omits segments.
