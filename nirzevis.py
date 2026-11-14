#!/usr/bin/env python3

import argparse
import shutil
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

# Defaults (overridable via environment / .env / CLI)
DEFAULT_WHISPER_CLI = Path('/home/hamid/Desktop/AI/whisper.cpp/whisper.cpp/build/bin/whisper-cli')
DEFAULT_MODEL_PATH = Path('/home/hamid/Desktop/AI/stt/models/ggml-medium.en.bin')
DEFAULT_LANGUAGE = 'en'
DEFAULT_DTW_MODEL = 'medium.en'
DEFAULT_BASE_URL = 'http://127.0.0.1:8000/v1'
DEFAULT_BATCH_SIZE = 8
DEFAULT_TARGET_LANGUAGE = 'fa'
NATIVE_AUDIO_EXTS = {'.flac', '.mp3', '.ogg', '.wav'}

# TTML namespaces
TT_NS = 'http://www.w3.org/ns/ttml'
TTM_NS = 'http://www.w3.org/ns/ttml#metadata'
ITUNES_NS = 'http://music.apple.com/lyric-ttml-internal'
ET.register_namespace('', TT_NS)
ET.register_namespace('ttm', TTM_NS)
ET.register_namespace('itunes', ITUNES_NS)
XML_NS = 'http://www.w3.org/XML/1998/namespace'

LANGUAGE_NAMES = {
    'fa': 'Persian', 'es': 'Spanish', 'fr': 'French', 'de': 'German',
    'ar': 'Arabic', 'pt-BR': 'Brazilian Portuguese',
}


def env_get(*names, default=None):
    """Return the first non-empty environment variable among names."""
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return default


def parse_bool(text):
    return str(text).strip().lower() in ('1', 'true', 'yes', 'y', 'on')


def resolve_path(value, default):
    if not value:
        return default
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def env_path(*names, default):
    return resolve_path(env_get(*names), default)


def load_env_file(env_path):
    """Load KEY=value entries from a .env file without overriding existing vars."""
    if not env_path or not env_path.is_file():
        return
    try:
        with env_path.open('r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#') or '=' not in line:
                    continue
                key, _, value = line.partition('=')
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
    except OSError as exc:
        print(f'Warning: could not read env file {env_path}: {exc}', file=sys.stderr)


def get_timestamp_ms(obj, field):
    offsets = obj.get('offsets', {})
    if field in offsets:
        return int(offsets[field])
    value = obj['timestamps'][field].replace(',', '.')
    hours, minutes, seconds = value.split(':')
    return round((int(hours) * 3600 + int(minutes) * 60 + float(seconds)) * 1000)


def format_lrc_time(ms):
    cs = (ms + 5) // 10
    minutes, remainder = divmod(cs, 6000)
    seconds, centiseconds = divmod(remainder, 100)
    return f'{minutes:02d}:{seconds:02d}.{centiseconds:02d}'


def format_ttml_time(ms):
    hours, remainder = divmod(ms, 3600000)
    minutes, remainder = divmod(remainder, 60000)
    seconds, milliseconds = divmod(remainder, 1000)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}'


def format_ass_time(ms):
    hours, remainder = divmod(ms, 3600000)
    minutes, remainder = divmod(remainder, 60000)
    seconds, milliseconds = divmod(remainder, 1000)
    centiseconds = milliseconds // 10
    return f'{hours}:{minutes:02d}:{seconds:02d}.{centiseconds:02d}'


def extract_words(tokens):
    """Combine Whisper BPE tokens; preserve DTW start/end timestamps."""
    words = []
    current_word = ''
    current_start = current_end = None

    def flush():
        nonlocal current_word, current_start, current_end
        if current_word.strip():
            words.append((current_start, max(current_start + 1, current_end), current_word.strip()))
        current_word = ''
        current_start = current_end = None

    for token in tokens:
        raw = token.get('text', '')
        if not raw or re.fullmatch(r'\[.*\]', raw.strip()):
            continue
        start = get_timestamp_ms(token, 'from')
        end = get_timestamp_ms(token, 'to')
        previous_end = 0
        for match in re.finditer(r'\S+', raw):
            piece = match.group()
            has_space = bool(raw[previous_end:match.start()])
            previous_end = match.end()
            punctuation_only = not any(ch.isalnum() for ch in piece)
            if not current_word:
                current_word, current_start, current_end = piece, start, end
            elif has_space and not punctuation_only:
                flush()
                current_word, current_start, current_end = piece, start, end
            else:
                current_word += piece
                current_end = max(current_end, end)
    flush()
    return words


def collect_segments(data):
    records = []
    for segment in data.get('transcription', []):
        words = extract_words(segment.get('tokens', []))
        if words:
            records.append({
                'words': words,
                'text': ' '.join(word for _, _, word in words),
                'translation': ''
            })
    return records


# OpenAI-compatible local LLM client; uses only Python standard library.
def http_json(url, payload=None, api_key=None, timeout=300):
    headers = {'Accept': 'application/json', 'User-Agent': 'Whisper-DTW-Bilingual/1.0'}
    if payload is not None:
        headers['Content-Type'] = 'application/json'
    if api_key:
        headers['Authorization'] = f'Bearer {api_key}'
    data = json.dumps(payload, ensure_ascii=False).encode('utf-8') if payload is not None else None
    request = urllib.request.Request(url, data=data, headers=headers, method='POST' if data is not None else 'GET')
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode('utf-8'))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode('utf-8', errors='replace')
        raise RuntimeError(f'LLM HTTP {exc.code}: {detail[:1500]}') from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f'Cannot connect to local LLM at {url}: {exc.reason}') from exc


def resolve_model(base_url, requested_model, api_key):
    if requested_model:
        return requested_model
    try:
        result = http_json(base_url.rstrip('/') + '/models', api_key=api_key, timeout=15)
        models = result.get('data', [])
        if models and models[0].get('id'):
            print(f"LLM model detected: {models[0]['id']}")
            return models[0]['id']
    except Exception as exc:
        print(f'Warning: model auto-detection failed: {exc}', file=sys.stderr)
    print("Warning: using model name 'local-model'; pass --llm-model if required by your server.", file=sys.stderr)
    return 'local-model'


def parse_model_json(content):
    """Parse model JSON, repairing common malformations.

    Strips markdown/code fences, extracts the outermost JSON structure, and
    repairs frequent LLM mistakes (single quotes, unquoted keys, trailing
    commas, bareword values, missing value quotes, stray text).
    """
    import ast

    content = content.strip()

    # Remove markdown / code fences anywhere in the text.
    content = re.sub(r'```(?:json)?\s*', '', content, flags=re.I)
    content = re.sub(r'~~~.*?~~~', '', content, flags=re.I | re.S)
    content = content.strip()

    def try_loads(text):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None

    result = try_loads(content)
    if result is not None:
        return result

    # Collect candidate substrings delimited by matching outermost { } or [ ].
    candidates = []

    def add_balanced(text, open_ch, close_ch):
        depth = 0
        start = None
        for i, ch in enumerate(text):
            if ch == open_ch:
                if depth == 0:
                    start = i
                depth += 1
            elif ch == close_ch:
                depth -= 1
                if depth == 0 and start is not None:
                    candidates.append(text[start:i + 1])
                    start = None

    add_balanced(content, '{', '}')
    add_balanced(content, '[', ']')

    # Also try the naive first-to-last span (covers cases with nested braces
    # inside string values that break the balanced scan).
    for open_ch, close_ch in (('{', '}'), ('[', ']')):
        s, e = content.find(open_ch), content.rfind(close_ch)
        if s >= 0 and e > s:
            candidates.append(content[s:e + 1])

    # Deduplicate while preserving order.
    seen = set()
    uniq = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            uniq.append(c)

    def repair_quotes(text):
        # Replace single-quoted strings with double-quoted ones (naive but
        # effective for LLM output); avoid touching apostrophes inside words.
        return re.sub(r"'(?!\\')([^']*)'(?!\\')", r'"\1"', text)

    def repair_keys(text):
        # Quote unquoted object keys: {id: 0} -> {"id": 0}
        return re.sub(r'({\s*|,\s*)([A-Za-z_][A-Za-z0-9_]*)\s*:', r'\1"\2":', text)

    def repair_trailing_commas(text):
        # Remove commas before closing braces/brackets.
        text = re.sub(r',\s*}', '}', text)
        text = re.sub(r',\s*]', ']', text)
        return text

    def repair_bareword_values(text):
        # "key":elif 49 -> "key":49  ;  "key":trueish -> "key":trueish (keep as-is, json will fail)
        text = re.sub(
            r'("id"\s*:\s*)[A-Za-z_]+\s*(-?\d+)',
            r'\1\2',
            text
        )
        return text

    repairers = [
        lambda t: t,
        repair_quotes,
        repair_keys,
        repair_trailing_commas,
        lambda t: repair_keys(repair_quotes(t)),
        lambda t: repair_trailing_commas(repair_keys(repair_quotes(t))),
        lambda t: repair_bareword_values(repair_trailing_commas(repair_keys(repair_quotes(t)))),
    ]

    for candidate in uniq:
        for repair in repairers:
            repaired = repair(candidate)
            result = try_loads(repaired)
            if result is not None:
                return result
            # Also try Python literal eval on repaired text (handles single
            # quotes, unquoted keys, etc. that json still rejects).
            try:
                evaluated = ast.literal_eval(repaired)
                if isinstance(evaluated, (dict, list)):
                    return evaluated
            except (ValueError, SyntaxError):
                pass

    raise RuntimeError(
        'Model returned malformed JSON after repair attempts:\n' +
        content[:3000]
    )


def extract_item_id(item):
    """Extract integer id from an item, accepting both 'id' and 'sans_id' keys."""
    if not isinstance(item, dict):
        return None
    for key in ('id', 'sans_id'):
        if key in item:
            try:
                return int(item[key])
            except (TypeError, ValueError):
                continue
    return None


def translate_batch(id_text_pairs, model, base_url, api_key, attempt=1, target_language='fa'):
    """
    Translate a list of (global_id, text) pairs.
    Returns (success_map, missing_ids) where:
      success_map: {global_id: translation_str}
      missing_ids: list of global_ids that were expected but not returned.
    Does NOT raise on missing ids; caller decides what to do.
    """
    items = [{'id': gid, 'text': text} for gid, text in id_text_pairs]
    expected_ids = {gid for gid, _ in id_text_pairs}

    target_name = LANGUAGE_NAMES.get(target_language, target_language)
    system_prompt = (
        f'You are a professional English-to-{target_name} translator. '
        f'Translate spoken English sentences into fluent, natural {target_name}.\n'
        'Rules:\n'
        '- Translate faithfully; do not summarize.\n'
        '- Preserve names, brands, technical terms and numbers accurately.\n'
        '- Do not include the original English or add explanations, notes, quotes or speaker labels.\n'
        f'- Return ONLY valid JSON in this exact shape: {{"translations":[{{"id":0,"translation":"Example {target_name} translation"}}]}}\n'
        '- Do NOT wrap the JSON in markdown code fences (no ``` or ~~~).\n'
        '- Do NOT output any text before or after the JSON.\n'
        '- Double-quote all keys and string values. No trailing commas.\n'
        '- Return exactly one translation for every supplied id; never change an id.\n'
        '- Always use the key "id" (never "sans_id" or any other name).'
    )
    user_prompt = 'Translate these segments into the target language. Keep every id unchanged:\n' + json.dumps(items, ensure_ascii=False, indent=2)
    if attempt > 1:
        user_prompt += (
            '\n\nIMPORTANT: Your previous response was invalid or incomplete. '
            'Return a syntactically valid JSON only. '
            'Every item must have an integer "id" copied exactly from the input '
            'and a non-empty translation string. Do not output any other text. '
            'Do not use keys like "sans_id".'
        )
    payload = {
        'model': model,
        'messages': [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ],
        'temperature': 0.2,
        'max_tokens': 8192,
        'stream': False
    }
    result = http_json(base_url.rstrip('/') + '/chat/completions', payload, api_key)
    try:
        message = result['choices'][0]['message']

        print('\nDEBUG: LLM message structure:')
        print(json.dumps(message, ensure_ascii=False, indent=2)[:5000])

        content = message.get('content')

        if isinstance(content, list):
            content = ''.join(
                item.get('text', '') if isinstance(item, dict) else str(item)
                for item in content
            )

        if not content:
            raise RuntimeError(
                'LLM returned empty content. Full response:\n' +
                json.dumps(result, ensure_ascii=False, indent=2)[:5000]
            )

    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(
            'Unexpected LLM response:\n' +
            json.dumps(result, ensure_ascii=False, indent=2)[:5000]
        ) from exc

    parsed = parse_model_json(content)

    # Support both JSON formats:
    # 1. {"translations": [...]}
    # 2. [...]

    if isinstance(parsed, list):
        translation_items = parsed
    elif isinstance(parsed, dict):
        translation_items = parsed.get('translations', [])
    else:
        raise RuntimeError(
            f'Unexpected JSON structure: {type(parsed).__name__}'
        )

    if not isinstance(translation_items, list):
        raise RuntimeError('Invalid translations structure.')

    mapped = {}

    for item in translation_items:
        if not isinstance(item, dict):
            continue

        item_id = extract_item_id(item)
        if item_id is None:
            continue

        translation = str(item.get('translation', '')).strip()
        if translation:
            if item_id in mapped:
                # Keep the first one
                continue
            mapped[item_id] = translation

    missing = sorted(expected_ids - set(mapped))
    return mapped, missing


def load_translations_cache(cache_path, target_language):
    """Load previously successful translations. Returns (translations_dict, cached_language)."""
    if not cache_path.is_file():
        return {}, None
    try:
        with cache_path.open('r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, dict) and isinstance(data.get('translations'), dict):
            lang = data.get('language')
            translations = {int(k): v for k, v in data['translations'].items() if v}
            if lang is not None and lang != target_language:
                print(
                    f'Cache language ({lang}) differs from target ({target_language}); ignoring cache.',
                    file=sys.stderr
                )
                return {}, lang
            return translations, lang
        # Old-script flat cache format is ignored.
        return {}, None
    except Exception as exc:
        print(f'Warning: could not load translation cache {cache_path}: {exc}', file=sys.stderr)
        return {}, None


def save_translations_cache(cache_path, translations_dict, target_language):
    """Persist the full map of successful translations (id -> text)."""
    serializable = {
        'language': target_language,
        'translations': {str(k): v for k, v in sorted(translations_dict.items())},
    }
    tmp_path = cache_path.with_suffix(cache_path.suffix + '.tmp')
    with tmp_path.open('w', encoding='utf-8') as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)
    tmp_path.replace(cache_path)


def save_failed_report(report_path, failed_items):
    """
    failed_items: list of dicts with keys id, text, error (optional)
    """
    with report_path.open('w', encoding='utf-8') as f:
        json.dump(failed_items, f, ensure_ascii=False, indent=2)
    print(f'Failed translation report written: {report_path}')


def translate_segments(records, base_url, requested_model=None, api_key=None,
                       batch_size=8, cache_path=None, failed_report_path=None,
                       target_language='fa', use_cache=True):
    """
    Translate all records with resume support and partial failure handling.

    - Successful translations are continuously written to cache_path (.translations.json)
    - On resume, already-translated segments are skipped.
    - If some ids remain missing after retries, they are recorded in failed_report_path
      and the process continues instead of aborting.
    Returns the list of permanently failed items.
    """
    if not records:
        return []

    model = resolve_model(base_url, requested_model, api_key)
    target_name = LANGUAGE_NAMES.get(target_language, target_language)
    print('\n' + '=' * 55 + f'\nLOCAL AI TRANSLATION ({target_name})\n' + '=' * 55)
    print(f'Endpoint : {base_url}\nModel    : {model}\nSegments : {len(records)}\nBatch    : {batch_size}')

    # Load previous successful translations for resume
    cache = {}
    if cache_path is not None and use_cache:
        cache, _ = load_translations_cache(cache_path, target_language)
        if cache:
            print(f'Resume: loaded {len(cache)} previously successful translations from {cache_path}')

    # Apply cached translations to records
    for idx, record in enumerate(records):
        if idx in cache and cache[idx]:
            record['translation'] = cache[idx]

    # Determine which indices still need translation
    pending_indices = [i for i, r in enumerate(records) if not r.get('translation')]
    if not pending_indices:
        print('All segments already translated (resume complete).')
        return []

    print(f'Pending segments to translate: {len(pending_indices)}')

    failed_items = []  # collect permanently failed ones
    max_attempts = 3

    # Process pending indices in batches
    i = 0
    while i < len(pending_indices):
        batch_indices = pending_indices[i:i + batch_size]

        print(f'\rTranslating ids {batch_indices[0]}-{batch_indices[-1]} '
              f'({i + 1}-{min(i + len(batch_indices), len(pending_indices))} / {len(pending_indices)})...',
              end='', flush=True)

        success_map = {}
        still_missing = list(batch_indices)
        last_error = None

        for attempt in range(1, max_attempts + 1):
            try:
                # Build (id, text) pairs for the currently missing ones only
                id_text_pairs = [(idx, records[idx]['text']) for idx in still_missing]
                mapped, missing = translate_batch(
                    id_text_pairs,
                    model, base_url, api_key,
                    attempt=attempt, target_language=target_language
                )
                # Merge newly obtained translations
                for gid, text in mapped.items():
                    if gid in still_missing:
                        success_map[gid] = text
                        records[gid]['translation'] = text

                still_missing = missing  # only the ones still absent

                if not still_missing:
                    break

                # Targeted retry of only the missing ids
                if attempt < max_attempts:
                    print(
                        f'\nPartial success on attempt {attempt}. '
                        f'Missing ids: {still_missing}. Retrying only those...',
                        file=sys.stderr
                    )
            except Exception as exc:
                last_error = exc
                print(
                    f'\nBatch starting at id {batch_indices[0]} failed '
                    f'(attempt {attempt}/{max_attempts}): {exc}',
                    file=sys.stderr
                )
                if attempt < max_attempts:
                    print('Retrying this batch...', file=sys.stderr)

        # After retries, apply whatever we got and record permanent failures
        for gid, text in success_map.items():
            records[gid]['translation'] = text
            cache[gid] = text

        if still_missing:
            print(
                f'\nWARNING: After {max_attempts} attempts, still missing ids: {still_missing}. '
                f'These will be recorded as failed and skipped.',
                file=sys.stderr
            )
            for gid in still_missing:
                failed_items.append({
                    'id': gid,
                    'text': records[gid]['text'],
                    'error': str(last_error) if last_error else 'Model omitted this id after retries'
                })
                # Leave translation empty so downstream knows it failed

        # Persist progress after every batch (even partial)
        if cache_path is not None:
            save_translations_cache(cache_path, cache, target_language)

        i += len(batch_indices)

    print(f'\n{target_name} translation phase finished.')

    if failed_items:
        if failed_report_path is not None:
            save_failed_report(failed_report_path, failed_items)
        print(f'{len(failed_items)} segment(s) could not be translated. See the failed report.')
    else:
        print('All requested segments translated successfully.')
    return failed_items


def generate_word_lrc(records, output_file):
    lines = []
    for record in records:
        for start, _end, word in record['words']:
            lines.append(f'[{format_lrc_time(start)}]{word}')
    if not lines:
        raise RuntimeError('No word timestamps found for LRC.')
    output_file.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'Word-by-word LRC: {output_file}')


def generate_bilingual_enhanced_lrc(records, output_file):
    """Enhanced LRC: timed English words; Persian translation on the next line."""
    lines = []
    for record in records:
        words = record['words']
        if not words:
            continue
        line_start = words[0][0]
        original = f'[{format_lrc_time(line_start)}]'
        for start, _end, word in words:
            original += f'<{format_lrc_time(start)}>{word} '
        lines.append(original.rstrip())
        if record['translation']:
            lines.append(f'[{format_lrc_time(line_start)}]{record["translation"].strip()}')
    if not lines:
        raise RuntimeError('No content found for bilingual Enhanced LRC.')
    output_file.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'Bilingual Enhanced LRC (Namida): {output_file}')


def generate_bilingual_ass(records, output_file):
    """Generate a word-by-word bilingual Advanced SubStation Alpha (.ass) subtitle file with karaoke tags."""
    lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        "Collisions: Normal",
        "PlayResX: 1920",
        "PlayResY: 1080",
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        "Style: English,Arial,48,&H00FFFFFF,&H0000FFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,2,2,2,10,10,60,1",
        "Style: Translation,Arial,42,&H0000FFFF,&H000000FF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"
    ]

    event_count = 0
    for record in records:
        words = record['words']
        if not words:
            continue
        
        start_ms = words[0][0]
        end_ms = words[-1][1]
        start_str = format_ass_time(start_ms)
        end_str = format_ass_time(end_ms)
        
        # Build word-by-word karaoke line using \kf (karaoke fill effect)
        karaoke_parts = []
        prev_end = start_ms
        for start, end, word in words:
            # Duration in centiseconds (10ms units) for the gap/word
            # To handle any gap before the word accurately:
            duration_cs = max(1, (end - start) // 10)
            karaoke_parts.append(f"{{\\kf{duration_cs}}}{word} ")
            prev_end = end

        original_karaoke_text = "".join(karaoke_parts)
        
        # Add English word-by-word karaoke line
        lines.append(f"Dialogue: 0,{start_str},{end_str},English,,0,0,0,,{original_karaoke_text}")
        event_count += 1

        # Add Translation line if available
        translation_text = record['translation'].strip()
        if translation_text:
            lines.append(f"Dialogue: 0,{start_str},{end_str},Translation,,0,0,0,,{translation_text}")
            event_count += 1

    if event_count == 0:
        raise RuntimeError('No content found for bilingual ASS subtitles.')

    output_file.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'Bilingual Word-by-Word ASS subtitles: {output_file} ({event_count} events)')


def generate_word_ttml(records, output_file, target_language='fa'):
    root = ET.Element(f'{{{TT_NS}}}tt', {
        f'{{{XML_NS}}}lang': 'en',
        f'{{{ITUNES_NS}}}timing': 'Word'
    })
    head = ET.SubElement(root, f'{{{TT_NS}}}head')
    metadata = ET.SubElement(head, f'{{{TT_NS}}}metadata')
    agent = ET.SubElement(metadata, f'{{{TTM_NS}}}agent', {
        'type': 'person', f'{{{XML_NS}}}id': 'v1'
    })
    ET.SubElement(agent, f'{{{TTM_NS}}}name', {'type': 'full'}).text = 'Speaker'
    body = ET.SubElement(root, f'{{{TT_NS}}}body')
    div = ET.SubElement(body, f'{{{TT_NS}}}div')

    word_count = 0
    for line_number, record in enumerate(records, start=1):
        words = record['words']
        if not words:
            continue
        paragraph = ET.SubElement(div, f'{{{TT_NS}}}p', {
            'begin': format_ttml_time(words[0][0]),
            'end': format_ttml_time(words[-1][1]),
            f'{{{ITUNES_NS}}}key': f'L{line_number}',
            f'{{{TTM_NS}}}agent': 'v1',
            f'{{{XML_NS}}}space': 'preserve'
        })
        # Keep the known-good LocalPlayer-compatible spacing convention.
        for index, (start, end, word) in enumerate(words):
            span = ET.SubElement(paragraph, f'{{{TT_NS}}}span', {
                'begin': format_ttml_time(start),
                'end': format_ttml_time(end)
            })
            span.text = word + (' ' if index < len(words) - 1 else '')
            word_count += 1
        # AMLL/Apple-style translation track; translation is sentence-synced.
        if record['translation'].strip():
            translated = ET.SubElement(paragraph, f'{{{TT_NS}}}span', {
                f'{{{TTM_NS}}}role': 'x-translation',
                f'{{{XML_NS}}}lang': target_language
            })
            translated.text = record['translation'].strip()

    if not word_count:
        raise RuntimeError('No word timestamps found for TTML.')
    # Do not pretty-print: preserve whitespace in timed lyric text.
    ET.ElementTree(root).write(output_file, encoding='utf-8', xml_declaration=True)
    print(f'Bilingual word-by-word TTML: {output_file} ({word_count} words)')


def run_whisper(audio_file, whisper_cli, model_path, language, dtw_model,
                output_dir, llm_base_url, llm_model, llm_api_key, batch_size,
                skip_transcription, skip_translation, target_language,
                refresh_translations):
    for path, label in ((whisper_cli, 'whisper-cli'), (model_path, 'Whisper model')):
        if not path.is_file():
            raise FileNotFoundError(f'{label} not found: {path}')

    if not skip_transcription and not audio_file.is_file():
        raise FileNotFoundError(f'Audio file not found: {audio_file}')

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = audio_file.stem
    prefix = output_dir / stem
    json_file = Path(str(prefix) + '.json')
    srt_file = Path(str(prefix) + '.srt')
    lrc_file = Path(str(prefix) + '.lrc')
    ttml_file = Path(str(prefix) + '.ttml')
    bilingual_lrc_file = Path(str(prefix) + '.bilingual.lrc')
    ass_file = Path(str(prefix) + '.ass')
    translations_cache = Path(str(prefix) + '.translations.json')
    failed_report = Path(str(prefix) + '.failed_translations.json')

    ran_whisper = False
    if not skip_transcription:
        # Check if the audio extension requires conversion via ffmpeg
        if audio_file.suffix.lower() not in NATIVE_AUDIO_EXTS:
            if not shutil.which('ffmpeg'):
                raise RuntimeError(
                    f"Unsupported audio extension '{audio_file.suffix}' and 'ffmpeg' is not available "
                    "in PATH to pipe conversion into whisper.cpp."
                )
            
            command = [
                str(whisper_cli), '-m', str(model_path), '-f', '-',
                '-l', language, '-dtw', dtw_model, '-nfa',
                '-osrt', '-ojf', '-of', str(prefix)
            ]
            print('\n' + '=' * 55 + '\nWHISPER TRANSCRIPTION (FFMPEG PIPED)\n' + '=' * 55)
            print(f'Audio : {audio_file} (Converted via ffmpeg -> stdin)')
            print(f'Model : {model_path}\nDTW   : {dtw_model}\nFlash Attention: Disabled')
            print('\nStarting ffmpeg & whisper.cpp...\n')

            ffmpeg_cmd = [
                'ffmpeg', '-nostdin', '-v', 'quiet', '-i', str(audio_file),
                '-ar', '16000', '-ac', '1', '-f', 'wav', '-'
            ]
            
            p_ffmpeg = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE)
            p_whisper = subprocess.run(command, stdin=p_ffmpeg.stdout, check=True)
            p_ffmpeg.stdout.close()
            p_ffmpeg.wait()
        else:
            command = [
                str(whisper_cli), '-m', str(model_path), '-f', str(audio_file),
                '-l', language, '-dtw', dtw_model, '-nfa',
                '-osrt', '-ojf', '-of', str(prefix)
            ]
            print('\n' + '=' * 55 + '\nWHISPER TRANSCRIPTION\n' + '=' * 55)
            print(f'Audio : {audio_file}\nModel : {model_path}\nDTW   : {dtw_model}\nFlash Attention: Disabled')
            print('\nStarting whisper.cpp...\n')
            subprocess.run(command, check=True)

        ran_whisper = True

        if not json_file.is_file():
            raise RuntimeError(f'JSON output not found: {json_file}')
        if not srt_file.is_file():
            raise RuntimeError(f'SRT output not found: {srt_file}')

        with json_file.open('r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        if not json_file.is_file():
            raise FileNotFoundError(f'--skip-transcription requires existing JSON: {json_file}')
        with json_file.open('r', encoding='utf-8') as f:
            data = json.load(f)

    records = collect_segments(data)
    if not records:
        raise RuntimeError('No timed transcription segments found.')
    print(f'Transcription segments: {len(records)}')

    failed_items = []
    if not skip_translation:
        answer = input(
            "\nDo you want to proceed with translating the generated text? [y/N]: "
        ).strip().lower()
    
        if answer in ('y', 'yes'):
            failed_items = translate_segments(
                records, llm_base_url, llm_model, llm_api_key, batch_size,
                cache_path=translations_cache, failed_report_path=failed_report,
                target_language=target_language, use_cache=not refresh_translations,
            )
        else:
            print("Translation skipped by user.")

    generate_word_lrc(records, lrc_file)
    generate_word_ttml(records, ttml_file, target_language)
    generate_bilingual_enhanced_lrc(records, bilingual_lrc_file)
    generate_bilingual_ass(records, ass_file)

    # Keep JSON if any conversion/translation failed; delete only on full success.
    if ran_whisper and not failed_items:
        json_file.unlink()

    exit_code = 2 if failed_items else 0

    print('\n' + '=' * 55 + '\nALL TASKS COMPLETED\n' + '=' * 55)
    print(f'SRT            : {srt_file}')
    print(f'Original LRC   : {lrc_file}')
    print(f'Bilingual TTML : {ttml_file}')
    print(f'Namida LRC     : {bilingual_lrc_file}')
    print(f'Bilingual ASS  : {ass_file}')
    print(f'Translations   : {translations_cache}')
    if failed_report.is_file():
        print(f'Failed report  : {failed_report}')
    return exit_code


def build_parser():
    parser = argparse.ArgumentParser(
        description='Generate SRT, word LRC, bilingual TTML, bilingual Enhanced LRC, and bilingual ASS using whisper.cpp DTW and a local LLM.'
    )
    parser.add_argument('audio', nargs='?', default=None, help='Input audio file path')
    parser.add_argument('--env-file', default='.env', help='Path to .env file (default: .env)')
    parser.add_argument('--llm-base-url', default=os.getenv('OPENAI_BASE_URL', DEFAULT_BASE_URL),
                        help='OpenAI-compatible API base URL (default: http://127.0.0.1:8000/v1)')
    parser.add_argument('--llm-model', default=os.getenv('OPENAI_MODEL'),
                        help='Model ID; auto-detected from /models when omitted')
    parser.add_argument('--llm-api-key', default=os.getenv('OPENAI_API_KEY', ''),
                        help='API key, if required by the local server')
    parser.add_argument('--translation-batch-size', type=int,
                        default=int(os.getenv('TRANSLATION_BATCH_SIZE', str(DEFAULT_BATCH_SIZE))),
                        help='Segments per translation request (default: 8)')
    parser.add_argument('--output-dir', default=os.getenv('OUTPUT_DIR', ''),
                        help='Output directory (default: beside the audio file)')
    parser.add_argument('--target-language', default=env_get('TARGET_LANGUAGE', default=DEFAULT_TARGET_LANGUAGE),
                        help='Translation target language code (default: fa)')
    parser.add_argument('--skip-transcription', action='store_true', dest='skip_transcription', default=None)
    parser.add_argument('--no-skip-transcription', action='store_false', dest='skip_transcription', default=None)
    parser.add_argument('--skip-translation', action='store_true', dest='skip_translation', default=None)
    parser.add_argument('--no-skip-translation', action='store_false', dest='skip_translation', default=None)
    parser.add_argument('--refresh-translations', action='store_true', dest='refresh_translations', default=None)
    parser.add_argument('--no-refresh-translations', action='store_false', dest='refresh_translations', default=None)
    return parser


def main():
    # First pass: locate the .env file so later defaults can reflect it.
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument('--env-file', default='.env')
    pre_args, _ = pre_parser.parse_known_args()
    load_env_file(Path(pre_args.env_file).expanduser())

    parser = build_parser()
    args = parser.parse_args()

    audio_arg = args.audio or env_get('AUDIO_FILE')
    if not audio_arg:
        parser.error('No audio file provided (positional argument or AUDIO_FILE env).')
    audio_file = Path(audio_arg).expanduser().resolve()

    whisper_cli = env_path('WHISPER_CLI', 'NIRZEVIS_WHISPER_CLI', default=DEFAULT_WHISPER_CLI)
    model_path = env_path('MODEL_PATH', 'NIRZEVIS_WHISPER_MODEL', default=DEFAULT_MODEL_PATH)
    language = env_get('NIRZEVIS_LANGUAGE', default=DEFAULT_LANGUAGE)
    dtw_model = env_get('NIRZEVIS_DTW_MODEL', default=DEFAULT_DTW_MODEL)

    if args.translation_batch_size < 1:
        parser.error('--translation-batch-size must be at least 1')

    output_dir = Path(args.output_dir) if args.output_dir else audio_file.parent
    target_language = args.target_language

    skip_transcription = args.skip_transcription if args.skip_transcription is not None else parse_bool(env_get('SKIP_TRANSCRIPTION', default='false'))
    skip_translation = args.skip_translation if args.skip_translation is not None else parse_bool(env_get('SKIP_TRANSLATION', default='false'))
    refresh_translations = args.refresh_translations if args.refresh_translations is not None else parse_bool(env_get('REFRESH_TRANSLATIONS', default='false'))

    try:
        exit_code = run_whisper(
            audio_file, whisper_cli, model_path, language, dtw_model,
            output_dir, args.llm_base_url, args.llm_model, args.llm_api_key,
            args.translation_batch_size, skip_transcription, skip_translation,
            target_language, refresh_translations,
        )
    except KeyboardInterrupt:
        print('\nInterrupted.', file=sys.stderr)
        sys.exit(130)
    except subprocess.CalledProcessError as exc:
        print(f'\nWhisper failed with exit code {exc.returncode}', file=sys.stderr)
        sys.exit(exc.returncode)
    except Exception as exc:
        print(f'\nERROR: {exc}', file=sys.stderr)
        sys.exit(1)
    sys.exit(exit_code)


if __name__ == '__main__':
    main()