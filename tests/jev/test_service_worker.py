"""The inference worker must preserve upstream responses and keep health responsive."""
import asyncio
import importlib.util
from pathlib import Path
import threading
import time
import unittest

spec = importlib.util.spec_from_file_location('serve_jev', Path(__file__).with_name('serve_jev.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ServiceWorkerTest(unittest.IsolatedAsyncioTestCase):
    async def test_serial_inference_and_responsive_health(self):
        import httpx
        from fastapi import FastAPI, Request
        app = FastAPI()
        started = threading.Event()
        counters = {'active': 0, 'peak': 0}

        @app.get('/health')
        async def health():
            return {'status': 'ok'}

        @app.post('/v1/systemone')
        async def upstream(request: Request):
            body = await request.json()
            counters['active'] += 1
            counters['peak'] = max(counters['peak'], counters['active'])
            started.set()
            time.sleep(.3)
            counters['active'] -= 1
            return {'same_body': body, 'debug': request.query_params.get('debug')}

        module.serial_inference_worker(app)
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
                first = asyncio.create_task(client.post('/v1/systemone?debug=1', json={'value': 1}))
                await asyncio.to_thread(started.wait, 2)
                second = asyncio.create_task(client.post('/v1/systemone', json={'value': 2}))
                before = time.perf_counter()
                result = await client.get('/health')
                self.assertLess(time.perf_counter()-before, .15)
                self.assertEqual(result.json(), {'status': 'ok'})
                a, b = await asyncio.gather(first, second)
                self.assertEqual(a.json(), {'same_body': {'value': 1}, 'debug': '1'})
                self.assertEqual(b.json(), {'same_body': {'value': 2}, 'debug': None})
                self.assertEqual(counters['peak'], 1)
        finally:
            app.state.inference_executor.shutdown()


if __name__ == '__main__':
    unittest.main()
