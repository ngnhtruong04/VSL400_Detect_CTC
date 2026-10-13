"""Integration tests for Realtime WebSocket endpoints across Isolated and Continuous modes."""

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from backend.app.main import app


def test_continuous_websocket_flow():
    client = TestClient(app)

    with client.websocket_connect("/ws/recognition?mode=continuous") as ws:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        _, jpeg_buf = cv2.imencode(".jpg", frame)

        # Send frame
        ws.send_bytes(jpeg_buf.tobytes())
        msg = ws.receive_json()

        assert msg["type"] == "continuous_recognition"
        assert msg["mode"] == "continuous"
        assert "state" in msg
        assert "live_gloss" in msg
        assert "landmarks" in msg
        assert "model" in msg
        assert "continuous" in msg

        # Verify landmarks
        landmarks = msg["landmarks"]
        assert "has_left_hand" in landmarks
        assert "has_right_hand" in landmarks
        assert "in_signing_space" in landmarks

        # Send reset command
        ws.send_text('{"type": "reset"}')
        reset_ack = ws.receive_json()
        assert reset_ack["type"] == "reset_ack"
        assert reset_ack["status"] == "ok"


def test_isolated_websocket_flow():
    client = TestClient(app)

    with client.websocket_connect("/ws/recognition?mode=isolated") as ws:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        _, jpeg_buf = cv2.imencode(".jpg", frame)

        # Send frame
        ws.send_bytes(jpeg_buf.tobytes())
        msg = ws.receive_json()

        assert msg["type"] == "isolated_recognition"
        assert msg["mode"] == "isolated"
        assert "state" in msg
        assert "landmarks" in msg
        assert "model" in msg

        # Send start_record command
        ws.send_text('{"type": "start_record"}')
        rec_ack = ws.receive_json()
        assert rec_ack["type"] == "record_ack"
        assert rec_ack["status"] == "recording"

        # Send another frame while recording
        ws.send_bytes(jpeg_buf.tobytes())
        msg2 = ws.receive_json()
        assert msg2["state"] == "RECORDING"

        # Send reset command
        ws.send_text('{"type": "reset"}')
        reset_ack = ws.receive_json()
        assert reset_ack["type"] == "reset_ack"
        assert reset_ack["status"] == "ok"
