"""Realtime WebSocket endpoint supporting Isolated Mode (Mode A) and Continuous Mode (Mode B).

Uses a Producer-Consumer latest-frame-slot architecture:
- Receiver continuously drains incoming WebSocket messages so control commands
  (start_record, stop_record, reset, config) execute with zero delay.
- Video frames overwrite a single `latest_frame` slot so stale frames are dropped
  instead of piling up in a growing TCP/WebSocket backlog.
"""

import asyncio
import base64
import json
import logging
from typing import Optional
import cv2
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query

from backend.app.services.isolated_session import IsolatedRecognitionSession
from backend.app.services.continuous_session import ContinuousRecognitionSession

logger = logging.getLogger("vsl.realtime_api")
router = APIRouter(tags=["Realtime"])


def _decode_jpeg_bytes(raw_bytes: bytes) -> Optional[np.ndarray]:
    if not raw_bytes:
        return None
    buf = np.frombuffer(raw_bytes, dtype=np.uint8)
    frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if frame is None or frame.size == 0:
        return None
    return frame


async def _handle_recognition_socket(websocket: WebSocket, mode: str):
    """Handles low-latency bidirectional video stream for isolated or continuous mode."""
    await websocket.accept()

    mode_clean = mode.lower().strip()
    is_isolated = (mode_clean == "isolated")

    if is_isolated:
        session = IsolatedRecognitionSession()
        logger.info(f"Client connected to ISOLATED mode session: {session.session_id}")
    else:
        session = ContinuousRecognitionSession()
        logger.info(f"Client connected to CONTINUOUS mode session: {session.session_id}")

    # Single-slot latest-frame buffer (guarantees zero backlog / minimum latency)
    latest_frame_bytes: Optional[bytes] = None
    frame_event = asyncio.Event()
    send_lock = asyncio.Lock()
    stop_flag = False

    async def safe_send_json(payload: dict):
        async with send_lock:
            await websocket.send_json(payload)

    async def frame_worker():
        nonlocal latest_frame_bytes
        try:
            while not stop_flag:
                await frame_event.wait()
                frame_event.clear()

                raw_bytes = latest_frame_bytes
                latest_frame_bytes = None
                if raw_bytes is None:
                    continue

                frame_bgr = _decode_jpeg_bytes(raw_bytes)
                if frame_bgr is None:
                    continue

                rec_msg, _ = await session.process_frame(
                    frame_bgr, client_action=None, draw_landmarks=False
                )
                await safe_send_json(rec_msg.model_dump())
        except asyncio.CancelledError:
            return
        except WebSocketDisconnect:
            return
        except Exception as e:
            logger.error(f"Frame worker error in session {session.session_id}: {e}", exc_info=True)

    worker_task = asyncio.create_task(frame_worker())

    try:
        while True:
            message = await websocket.receive()

            if message.get("type") == "websocket.disconnect":
                logger.info(f"Client disconnected (Mode: {mode_clean}, Session: {session.session_id})")
                break

            if "bytes" in message and message["bytes"]:
                # Overwrite slot with newest frame (drops any unconsumed older frame)
                latest_frame_bytes = message["bytes"]
                frame_event.set()

            elif "text" in message and message["text"]:
                try:
                    payload = json.loads(message["text"])
                    msg_type = payload.get("type", "frame")

                    if msg_type == "frame":
                        b64_str = payload.get("data", "")
                        if "," in b64_str:
                            b64_str = b64_str.split(",", 1)[1]
                        latest_frame_bytes = base64.b64decode(b64_str)
                        frame_event.set()

                    elif msg_type == "reset":
                        latest_frame_bytes = None
                        session.reset()
                        await safe_send_json({"type": "reset_ack", "status": "ok", "mode": mode_clean})

                    elif msg_type == "start_record":
                        latest_frame_bytes = None
                        if is_isolated and hasattr(session, "start_recording"):
                            session.start_recording()
                            await safe_send_json({"type": "record_ack", "status": "recording"})

                    elif msg_type == "stop_record":
                        latest_frame_bytes = None
                        if is_isolated and hasattr(session, "finalize_and_infer_async"):
                            rec_msg = await session.finalize_and_infer_async()
                            await safe_send_json(rec_msg.model_dump())
                        elif is_isolated and hasattr(session, "finalize_and_infer"):
                            rec_msg = session.finalize_and_infer()
                            await safe_send_json(rec_msg.model_dump())
                        elif is_isolated and hasattr(session, "stop_recording"):
                            session.stop_recording()
                            await safe_send_json({"type": "record_ack", "status": "inferring"})

                    elif msg_type == "config":
                        sensitivity = payload.get("sensitivity")
                        if not is_isolated and sensitivity and hasattr(session, "update_sensitivity"):
                            session.update_sensitivity(sensitivity)
                        await safe_send_json({"type": "config_ack", "status": "ok"})

                    elif msg_type == "ping":
                        await safe_send_json({"type": "pong", "timestamp": payload.get("timestamp")})

                except Exception as e:
                    logger.warning(f"Error parsing incoming JSON: {e}")

    except WebSocketDisconnect:
        logger.info(f"Client disconnected from WebSocket (Mode: {mode_clean}, Session: {session.session_id})")
    except Exception as e:
        logger.error(f"WebSocket error in session {session.session_id}: {e}", exc_info=True)
    finally:
        stop_flag = True
        frame_event.set()
        if not worker_task.done():
            worker_task.cancel()
            try:
                await worker_task
            except asyncio.CancelledError:
                pass
        session.close()


@router.websocket("/ws/recognition")
async def recognition_websocket(
    websocket: WebSocket,
    mode: str = Query("continuous", description="Mode: 'isolated' (Mode A) or 'continuous' (Mode B)"),
):
    await _handle_recognition_socket(websocket, mode=mode)


@router.websocket("/ws/recognition/isolated")
async def isolated_recognition_websocket(websocket: WebSocket):
    await _handle_recognition_socket(websocket, mode="isolated")


@router.websocket("/ws/recognition/continuous")
async def continuous_recognition_websocket(websocket: WebSocket):
    await _handle_recognition_socket(websocket, mode="continuous")
