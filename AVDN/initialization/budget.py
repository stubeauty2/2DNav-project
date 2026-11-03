"""Conservative request reservations against a shared local CNY budget."""
import contextlib
import json
import os
import threading
import time
import uuid
from pathlib import Path

PRICES = {
    'qwen3.7-plus': {'input': 2.0, 'output': 8.0, 'max_input_tier': 256000,
                      'url': 'https://www.qianwenai.com/models/qwen3.7-plus'},
    'qwen3.8-max': {'input': 12.0, 'output': 36.0, 'max_input_tier': 983000,
                    'url': 'https://www.qianwenai.com/models/qwen3.8-max'},
    'qwen3.5-plus': {'input': 0.8, 'output': 4.8, 'max_input_tier': 128000,
                    'url': 'https://www.qianwenai.com/models/qwen3.5-plus'},
    'qwen3-vl-plus': {'input': 1.0, 'output': 10.0, 'max_input_tier': 32000,
                     'url': 'https://www.qianwenai.com/models/qwen3-vl-plus'},
    'qwen3.5-flash': {'input': 0.2, 'output': 2.0, 'max_input_tier': 128000,
                     'url': 'https://www.qianwenai.com/models/qwen3.5-flash'},
    'qwen3.8-flash': {'input': 0.8, 'output': 2.7, 'max_input_tier': 991000,
                     'url': 'https://www.qianwenai.com/models/qwen3.8-flash'}
}


class BudgetExceeded(RuntimeError):
    pass


class RequestBudget:
    def __init__(self, path, limit_cny, save_json):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.limit = None if limit_cny is None else float(limit_cny)
        self.save_json = save_json
        self.thread_lock = threading.Lock()
        if self.limit is not None and self.limit <= 0:
            raise ValueError('Budget must be positive')
        with self.locked():
            if self.path.exists():
                state = self.read()
                if state['limit_cny'] != self.limit:
                    raise ValueError('Budget ledger has a different limit; do not silently reset it')
            else:
                self.save_json(self.path, {'limit_cny': self.limit, 'charged_cny': 0.0,
                    'reservations': {}, 'settled_requests': 0, 'price_checked_on': '2026-10-14',
                    'basis': 'Published uncached rates; no discount assumed. Unsettled requests retain full reservations.'})

    @contextlib.contextmanager
    def locked(self):
        with self.thread_lock:
            with self.path.with_suffix('.lock').open('a+b') as handle:
                handle.seek(0, os.SEEK_END)
                if handle.tell() == 0:
                    handle.write(b'0'); handle.flush()
                handle.seek(0)
                if os.name == 'nt':
                    import msvcrt
                    for attempt in range(500):
                        try:
                            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                            break
                        except OSError:
                            if attempt == 499: raise
                            time.sleep(0.02)
                else:
                    import fcntl
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    handle.seek(0)
                    if os.name == 'nt':
                        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                    else:
                        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def read(self):
        return json.loads(self.path.read_text(encoding='utf-8'))

    def reserve(self, model, messages, max_tokens):
        if model not in PRICES:
            raise BudgetExceeded('No verified price configured for this model')
        price = PRICES[model]
        # UTF-8 bytes are a deliberately loose upper bound for text tokens.
        # Inputs here are 512x512 PNGs; reserve 1024 tokens/image vs observed 258.
        upper = 512
        for message in messages:
            content = message['content']
            if isinstance(content, str):
                upper += len(content.encode('utf-8'))
            else:
                for part in content:
                    if part.get('type') == 'text':
                        upper += len(part['text'].encode('utf-8'))
                    elif part.get('type') == 'image_url':
                        upper += 1024
        if upper > price['max_input_tier']:
            raise BudgetExceeded('Conservative input bound exceeds verified price tier; reduce input first')
        amount = (upper * price['input'] + max_tokens * price['output']) / 1e6
        ident = uuid.uuid4().hex
        with self.locked():
            state = self.read()
            committed = state['charged_cny'] + sum(x['cny'] for x in state['reservations'].values())
            if self.limit is not None and committed + amount > self.limit + 1e-10:
                raise BudgetExceeded('Round cost cap reached; no API request sent')
            state['reservations'][ident] = {'cny': amount, 'model': model, 'input_token_bound': upper}
            self.save_json(self.path, state)
        return ident

    def settle(self, ident, usage):
        with self.locked():
            state = self.read()
            reservation = state['reservations'].pop(ident)
            price = PRICES[reservation['model']]
            if usage and 'input_tokens' in usage and 'output_tokens' in usage:
                amount = (usage['input_tokens'] * price['input'] + usage['output_tokens'] * price['output']) / 1e6
            else:
                amount = reservation['cny']
            state['charged_cny'] += amount
            state['settled_requests'] += 1
            self.save_json(self.path, state)
