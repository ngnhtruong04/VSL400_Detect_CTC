"""Realtime WebSocket endpoint supporting both Isolated Mode (Mode A) and Continuous Mode (Mode B)."""

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


async def _handle_recognition_socket(websocket: WebSocket, mode: str):
    """Handles bidirectional video stream for either isolated or continuous mode."""
    await websocket.accept()

    mode_clean = mode.lower().strip()
    is_isolated = (mode_clean == "isolated")

    if is_isolated:
        session = IsolatedRecognitionSession()
        logger.info(f"Client connected to ISOLATED mode session: {session.session_id}")
    else:
        session = ContinuousRecognitionSession()
        logger.info(f"Client connected to CONTINUOUS mode session: {session.session_id}")

    try:
        while True:
            message = await websocket.receive()
            frame_bgr: Optional[np.ndarray] = None
            client_action: Optional[str] = None

            if "bytes" in message and message["bytes"]:
                buf = np.frombuffer(message["bytes"], dtype=np.uint8)
                frame_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)

            elif "text" in message and message["text"]:
                try:
                    payload = json.loads(message["text"])
                    msg_type = payload.get("type", "frame")

                    if msg_type == "frame":
                        b64_str = payload.get("data", "")
                        if "," in b64_str:
                            b64_str = b64_str.split(",", 1)[1]
                        img_bytes = base64.b64decode(b64_str)
                        buf = np.frombuffer(img_bytes, dtype=np.uint8)
                        frame_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)

                    elif msg_type == "reset":
                        session.reset()
                        await websocket.send_json({"type": "reset_ack", "status": "ok", "mode": mode_clean})
                        continue

                    elif msg_type == "start_record":
                        if is_isolated and hasattr(session, "start_recording"):
                            session.start_recording()
                            await websocket.send_json({"type": "record_ack", "status": "recording"})
                        continue

                    elif msg_type == "stop_record":
                        if is_isolated and hasattr(session, "finalize_and_infer"):
                            rec_msg = session.finalize_and_infer()
                            await websocket.send_json(rec_msg.model_dump())
                        elif is_isolated and hasattr(session, "stop_recording"):
                            session.stop_recording()
                            await websocket.send_json({"type": "record_ack", "status": "inferring"})
                        continue

                    elif msg_type == "ping":
                        await websocket.send_json({"type": "pong", "timestamp": payload.get("timestamp")})
                        continue

                    else:
                        continue

                except Exception as e:
                    logger.warning(f"Error parsing incoming JSON: {e}")
                    continue

            if frame_bgr is None or frame_bgr.size == 0:
                continue

            # Process frame using the respective mode session
            if is_isolated:
                rec_msg, _ = session.process_frame(
                    frame_bgr, client_action=client_action, draw_landmarks=False
                )
                await websocket.send_json(rec_msg.model_dump())
            else:
                rec_msg, _ = await session.process_frame(
                    frame_bgr, client_action=client_action, draw_landmarks=False
                )
                await websocket.send_json(rec_msg.model_dump())

    except WebSocketDisconnect:
        logger.info(f"Client disconnected from WebSocket (Mode: {mode_clean}, Session: {session.session_id})")
    except Exception as e:
        logger.error(f"WebSocket error in session {session.session_id}: {e}", exc_info=True)
    finally:
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
