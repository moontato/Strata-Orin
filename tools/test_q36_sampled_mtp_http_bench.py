"""HTTP harness checks only; no inference/performance claim."""
import contextlib
import io
import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

from tools import q36_sampled_mtp_http_bench as bench


class HttpBench(unittest.TestCase):
    def test_discovered_model_and_on_off_bodies(self):
        bodies = []
        auth = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def answer(self, body):
                raw = json.dumps(body).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self):
                auth.append((self.path, self.headers.get("Authorization")))
                self.answer({"data": [{"id": "qwen3.6-sampled-mtp-test"}]})

            def do_POST(self):
                auth.append((self.path, self.headers.get("Authorization")))
                bodies.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                self.answer({"usage": {"prompt_tokens": 20, "completion_tokens": 4},
                             "choices": [{"finish_reason": "length"}],
                             "timings": {"predicted_ms": 100, "predicted_per_second": 40,
                                         "draft_n": 2, "draft_n_accepted": 1}})

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            url = f"http://127.0.0.1:{server.server_port}"
            with mock.patch("sys.argv", ["bench", "--base-url", url, "--api-key", "test-key",
                                        "--warmup", "0", "--reps", "1"]), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(bench.main(), 0)
        finally:
            server.shutdown()
            thread.join()
            server.server_close()
        self.assertEqual(len(bodies), 2)
        self.assertEqual(auth[0][0], "/v1/models")
        self.assertTrue(all(key == "Bearer test-key" for _, key in auth))
        self.assertTrue(all(b["model"] == "qwen3.6-sampled-mtp-test" for b in bodies))
        self.assertEqual([b.pop("strata_tune")["sampled_mtp"] for b in bodies], [0, 1])
        self.assertEqual(bodies[0], bodies[1])

    def test_empty_model_list_fails_explicitly(self):
        with mock.patch.object(bench, "api_request", return_value={"data": []}):
            with self.assertRaisesRegex(ValueError, "no API model ID"):
                bench.discover_model("http://unused", "")


if __name__ == "__main__":
    unittest.main()
