"""Native DashScope SSE and compatible chat transport; never logs headers."""
import json
import time
import requests

DEFAULT_MODEL = 'qwen3.5-flash'
DEFAULT_URL = 'https://maas.qianwenaiapi.com/api/v1/services/aigc/multimodal-generation/generation'


def native_messages(messages):
    converted = []
    for message in messages:
        content = message['content']
        if isinstance(content, str):
            parts = [{'text': content}]
        else:
            parts = []
            for item in content:
                if item['type'] == 'text':
                    parts.append({'text': item['text']})
                elif item['type'] == 'image_url':
                    parts.append({'image': item['image_url']['url']})
        converted.append({'role': message['role'], 'content': parts})
    return converted


def extract_text(body):
    choices = body.get('output', body).get('choices', [])
    if not choices:
        return ''
    content = choices[0].get('message', {}).get('content', '')
    if isinstance(content, str):
        return content
    # Native thinking fields are deliberately not persisted as output evidence.
    return ''.join(part.get('text', '') for part in content)


def consume_native(lines, deadline=None):
    texts, usage = [], None
    for line in lines:
        if deadline is not None and time.monotonic() >= deadline:
            raise requests.Timeout('Native SSE exceeded total response deadline')
        if not line or not line.startswith('data:'):
            continue
        value = line[5:].strip()
        if not value or value == '[DONE]':
            continue
        body = json.loads(value)
        if body.get('code') and not body.get('output'):
            raise ValueError('Native API error code: '+str(body['code']))
        texts.append(extract_text(body))
        if body.get('usage'):
            usage = body['usage']
    return ''.join(texts), usage


def call_model(url, model, key, messages, timeout, protocol='auto', enable_thinking=True,
               thinking_budget=50, max_tokens=4096, temperature=None, response_format=None):
    deadline = time.monotonic() + timeout
    native = protocol == 'native' or (protocol == 'auto' and '/multimodal-generation/' in url)
    headers = {'Authorization': 'Bearer '+key, 'Content-Type': 'application/json'}
    if native:
        headers['X-DashScope-SSE'] = 'enable'
        parameters = {'enable_thinking': enable_thinking, 'incremental_output': True,
                      'max_tokens': max_tokens}
        if enable_thinking:
            parameters['thinking_budget'] = thinking_budget
        if temperature is not None:
            parameters['temperature'] = temperature
        if response_format is not None:
            parameters['response_format'] = response_format
        payload = {'model': model, 'input': {'messages': native_messages(messages)},
                   'parameters': parameters}
    else:
        payload = {'model': model, 'messages': messages, 'temperature': 0,
                   'max_tokens': max_tokens, 'stream': False, 'enable_thinking': enable_thinking}
        if enable_thinking:
            payload['thinking_budget'] = thinking_budget
        if response_format is not None:
            payload['response_format'] = response_format
    with requests.post(url, headers=headers, json=payload, timeout=(20,timeout), stream=native) as response:
        if response.status_code != 200:
            return {'status': response.status_code, 'text': None, 'usage': None}
        if native and 'event-stream' in response.headers.get('Content-Type',''):
            response.encoding = 'utf-8'
            # Small chunks let heartbeat-only streams reach the deadline check.
            text, usage = consume_native(response.iter_lines(chunk_size=1, decode_unicode=True), deadline)
        else:
            body = response.json()
            if body.get('code') and not body.get('output'):
                raise ValueError('Native API error code: '+str(body['code']))
            text, usage = extract_text(body), body.get('usage')
        if not text.strip():
            raise ValueError('Model returned no final answer text')
        return {'status': response.status_code, 'text': text, 'usage': usage}
