import { useState, useEffect, useRef, useCallback } from 'react';
import {
  FSMStatus,
  MetricsData,
  RecognitionMode,
  LandmarksData,
  IsolatedPrediction,
  ModelDiagnostic,
  LandmarksDiagnostic,
  IsolatedRecognitionMessage,
  ContinuousRecognitionMessage,
  RecognitionMessage,
  UtteranceCompleteMessage,
  UtteranceHistoryItem,
} from '../types';

interface UseRecognitionSocketProps {
  serverBaseUrl?: string;
  mode: RecognitionMode;
  enabled?: boolean;
}

function arraysEqual<T>(a: T[], b: T[]): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i]) return false;
  }
  return true;
}

export function useRecognitionSocket({
  serverBaseUrl = `ws://${window.location.hostname || 'localhost'}:8000/ws/recognition`,
  mode = 'continuous',
  enabled = true,
}: UseRecognitionSocketProps) {
  const [isConnected, setIsConnected] = useState<boolean>(false);

  // Mode A (Isolated) states
  const [isolatedState, setIsolatedState] = useState<'IDLE' | 'RECORDING' | 'INFERRING' | 'RESULT'>('IDLE');
  const [isolatedPrediction, setIsolatedPrediction] = useState<IsolatedPrediction | null>(null);

  // Mode B (Continuous) states
  const [fsmState, setFsmState] = useState<FSMStatus>('IDLE');
  const [liveGloss, setLiveGloss] = useState<string[]>([]);
  const [stableTokens, setStableTokens] = useState<string[]>([]);
  const [finalGloss, setFinalGloss] = useState<string[]>([]);
  const [finalSentence, setFinalSentence] = useState<string>('');
  const [utteranceHistory, setUtteranceHistory] = useState<UtteranceHistoryItem[]>([]);
  const [endpointReason, setEndpointReason] = useState<string | null>(null);
  const prevFsmStateRef = useRef<FSMStatus>('IDLE');
  const lastCommitSigRef = useRef<string>('');

  // Shared diagnostics & tokens
  const [rawTokens, setRawTokens] = useState<number[]>([]);
  const [collapsedTokens, setCollapsedTokens] = useState<number[]>([]);
  const [landmarks, setLandmarks] = useState<LandmarksData | null>(null);
  const [modelDiagnostic, setModelDiagnostic] = useState<ModelDiagnostic | null>(null);
  const [landmarksDiagnostic, setLandmarksDiagnostic] = useState<LandmarksDiagnostic | null>(null);
  const [metrics, setMetrics] = useState<MetricsData>({
    fps: 0,
    mediapipe_ms: 0,
    preprocessing_ms: 0,
    inference_ms: 0,
    decode_ms: 0,
    total_ms: 0,
    motion_energy: 0,
    blank_ratio: 1.0,
    buffer_frames: 0,
  });

  // High-frequency landmark ref for zero-latency 60 FPS canvas rendering (bypasses React reconciler)
  const landmarksRef = useRef<LandmarksData | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const isEncodingRef = useRef<boolean>(false);
  // Strict in-flight flow control: prevents WebSocket frame backlog & hand-tracking lag
  const inFlightRef = useRef<boolean>(false);
  const lastSentAtRef = useRef<number>(0);

  // Guard against stale pre-record frames overwriting RECORDING right after clicking Start Record
  const pendingRecordAckUntilRef = useRef<number>(0);

  // Throttle heavy HUD/metrics React updates to ~10 Hz while landmarks draw at full 30-60 FPS
  const lastHudUpdateRef = useRef<number>(0);
  const lastHandFlagsRef = useRef<{ l: boolean; r: boolean; s: boolean }>({ l: false, r: false, s: false });

  // Connect WebSocket when enabled or mode changes
  useEffect(() => {
    if (!enabled) {
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
      landmarksRef.current = null;
      inFlightRef.current = false;
      isEncodingRef.current = false;
      setIsConnected(false);
      return;
    }

    let isUnmounted = false;
    let reconnectTimeout: ReturnType<typeof setTimeout> | null = null;

    const fullUrl = `${serverBaseUrl}?mode=${mode}`;

    const connect = () => {
      try {
        const ws = new WebSocket(fullUrl);
        ws.binaryType = 'arraybuffer';
        wsRef.current = ws;

        ws.onopen = () => {
          if (isUnmounted) {
            ws.close();
            return;
          }
          inFlightRef.current = false;
          isEncodingRef.current = false;
          setIsConnected(true);
        };

        ws.onclose = () => {
          if (isUnmounted) return;
          inFlightRef.current = false;
          isEncodingRef.current = false;
          setIsConnected(false);
          reconnectTimeout = setTimeout(connect, 1500);
        };

        ws.onerror = (err) => {
          console.warn('WebSocket error:', err);
          if (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING) {
            ws.close();
          }
        };

        ws.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            const nowMs = performance.now();

            if (data.type === 'record_ack') {
              if (data.status === 'recording') {
                pendingRecordAckUntilRef.current = 0;
                setIsolatedState('RECORDING');
              }
              return;
            }

            if (data.type === 'reset_ack' || data.type === 'config_ack' || data.type === 'pong') {
              return;
            }

            // Server finished processing a frame -> release in-flight gate immediately
            inFlightRef.current = false;

            const shouldUpdateHud = nowMs - lastHudUpdateRef.current >= 100;

            if (data.type === 'isolated_recognition') {
              const msg = data as IsolatedRecognitionMessage;

              // Ignore stale pre-record frames while waiting for backend start_record ack
              const awaitingRecordAck = pendingRecordAckUntilRef.current > nowMs;
              if (!awaitingRecordAck || msg.state === 'RECORDING') {
                if (msg.state === 'RECORDING') {
                  pendingRecordAckUntilRef.current = 0;
                }
                setIsolatedState((prev) => (prev === msg.state ? prev : msg.state));
              }

              if (msg.prediction) {
                setIsolatedPrediction(msg.prediction);
                if (msg.prediction.gloss) {
                  setLiveGloss((prev) =>
                    prev.length === 1 && prev[0] === msg.prediction!.gloss ? prev : [msg.prediction!.gloss]
                  );
                }
              }

              if (msg.landmarks) {
                const lmData: LandmarksData = {
                  left_hand: msg.landmarks.left_hand,
                  right_hand: msg.landmarks.right_hand,
                  pose: msg.landmarks.pose,
                  has_left_hand: msg.landmarks.has_left_hand,
                  has_right_hand: msg.landmarks.has_right_hand,
                  in_signing_space: msg.landmarks.in_signing_space,
                  left_valid_ratio: msg.landmarks.left_valid_ratio,
                  right_valid_ratio: msg.landmarks.right_valid_ratio,
                  overall_valid_ratio: msg.landmarks.overall_valid_ratio,
                };
                // Immediate zero-latency update for 60 FPS canvas renderer
                landmarksRef.current = lmData;

                const flagsChanged =
                  lastHandFlagsRef.current.l !== lmData.has_left_hand ||
                  lastHandFlagsRef.current.r !== lmData.has_right_hand ||
                  lastHandFlagsRef.current.s !== lmData.in_signing_space;

                if (flagsChanged || shouldUpdateHud) {
                  lastHandFlagsRef.current = {
                    l: lmData.has_left_hand,
                    r: lmData.has_right_hand,
                    s: lmData.in_signing_space,
                  };
                  setLandmarks(lmData);
                  setLandmarksDiagnostic(msg.landmarks);
                }
              }

              if (msg.model && (shouldUpdateHud || msg.state === 'RESULT' || msg.model.buffer_frames === 0)) {
                lastHudUpdateRef.current = nowMs;
                setModelDiagnostic(msg.model);
                setMetrics({
                  fps: msg.fps,
                  mediapipe_ms: msg.model.mediapipe_ms ?? 0,
                  preprocessing_ms: msg.model.preprocessing_ms ?? 0,
                  inference_ms: msg.model.forward_ms,
                  decode_ms: msg.model.decode_ms,
                  total_ms: msg.total_ms,
                  motion_energy: msg.motion_energy,
                  blank_ratio: msg.model.blank_ratio,
                  buffer_frames: msg.model.buffer_frames,
                });
              }
            } else if (data.type === 'continuous_recognition') {
              const msg = data as ContinuousRecognitionMessage;

              const prevState = prevFsmStateRef.current;
              prevFsmStateRef.current = msg.state;
              setFsmState((prev) => (prev === msg.state ? prev : msg.state));

              const nextLive = msg.continuous.live_tokens || [];
              setLiveGloss((prev) => (arraysEqual(prev, nextLive) ? prev : nextLive));

              const nextStable = msg.continuous.stable_tokens || [];
              setStableTokens((prev) => (arraysEqual(prev, nextStable) ? prev : nextStable));

              const isCommitEvent =
                msg.state === 'END_OF_UTTERANCE' &&
                (prevState !== 'END_OF_UTTERANCE' || msg.continuous.endpoint_reason === 'manual_commit');

              if (isCommitEvent && msg.continuous.final_tokens && msg.continuous.final_tokens.length > 0) {
                const committedTokens = msg.continuous.final_tokens;
                const committedSentence = msg.continuous.final_sentence || committedTokens.join(' ');
                const sig = `${Math.floor(msg.timestamp * 2)}|${committedTokens.join('|')}`;
                if (sig !== lastCommitSigRef.current) {
                  lastCommitSigRef.current = sig;
                  // Append committed glosses into continuous session timeline (cleared only on Reset)
                  setFinalGloss((prev) => [...prev, ...committedTokens]);
                  setFinalSentence(committedSentence);
                  setUtteranceHistory((prev) => [
                    {
                      id: `${Date.now()}-${prev.length}`,
                      gloss: committedTokens,
                      sentence: committedSentence,
                      timestamp: Date.now(),
                    },
                    ...prev.slice(0, 9),
                  ]);
                }
              } else if (msg.continuous.final_tokens && msg.continuous.final_tokens.length > 0) {
                const fallbackTokens = msg.continuous.final_tokens;
                setFinalGloss((prev) => (prev.length === 0 ? fallbackTokens : prev));
                if (msg.continuous.final_sentence) {
                  const fallbackSent = msg.continuous.final_sentence;
                  setFinalSentence((prev) => (prev ? prev : fallbackSent));
                }
              }
              const nextReason = msg.continuous.endpoint_reason || null;
              setEndpointReason((prev) => (prev === nextReason ? prev : nextReason));

              if (msg.landmarks) {
                const lmData: LandmarksData = {
                  left_hand: msg.landmarks.left_hand,
                  right_hand: msg.landmarks.right_hand,
                  pose: msg.landmarks.pose,
                  has_left_hand: msg.landmarks.has_left_hand,
                  has_right_hand: msg.landmarks.has_right_hand,
                  in_signing_space: msg.landmarks.in_signing_space,
                  left_valid_ratio: msg.landmarks.left_valid_ratio,
                  right_valid_ratio: msg.landmarks.right_valid_ratio,
                  overall_valid_ratio: msg.landmarks.overall_valid_ratio,
                };
                // Immediate zero-latency update for 60 FPS canvas renderer
                landmarksRef.current = lmData;

                const flagsChanged =
                  lastHandFlagsRef.current.l !== lmData.has_left_hand ||
                  lastHandFlagsRef.current.r !== lmData.has_right_hand ||
                  lastHandFlagsRef.current.s !== lmData.in_signing_space;

                if (flagsChanged || shouldUpdateHud) {
                  lastHandFlagsRef.current = {
                    l: lmData.has_left_hand,
                    r: lmData.has_right_hand,
                    s: lmData.in_signing_space,
                  };
                  setLandmarks(lmData);
                  setLandmarksDiagnostic(msg.landmarks);
                }
              }

              if (msg.model && shouldUpdateHud) {
                lastHudUpdateRef.current = nowMs;
                setRawTokens(msg.continuous.raw_token_ids || []);
                setCollapsedTokens(msg.continuous.collapsed_token_ids || []);
                setModelDiagnostic(msg.model);
                setMetrics({
                  fps: msg.fps,
                  mediapipe_ms: msg.model.mediapipe_ms ?? 0,
                  preprocessing_ms: msg.model.preprocessing_ms ?? 0,
                  inference_ms: msg.model.forward_ms,
                  decode_ms: msg.model.decode_ms,
                  total_ms: msg.total_ms,
                  motion_energy: msg.motion_energy,
                  blank_ratio: msg.model.blank_ratio,
                  buffer_frames: msg.continuous.utterance_frames,
                });
              }
            } else if (data.type === 'recognition') {
              const msg = data as RecognitionMessage;
              setFsmState(msg.state);
              setLiveGloss(msg.live_gloss || []);
              if (msg.raw_tokens) setRawTokens(msg.raw_tokens);
              if (msg.collapsed_tokens) setCollapsedTokens(msg.collapsed_tokens);
              if (msg.landmarks !== undefined) {
                landmarksRef.current = msg.landmarks;
                setLandmarks(msg.landmarks);
              }
              if (msg.metrics) setMetrics(msg.metrics);
            } else if (data.type === 'utterance_complete') {
              const msg = data as UtteranceCompleteMessage;
              setFsmState(msg.state);
              setFinalGloss(msg.gloss || []);
              setFinalSentence(msg.sentence || '');
              if (msg.metrics) setMetrics(msg.metrics);
            }
          } catch (e) {
            console.error('Failed to parse WebSocket message:', e);
          }
        };
      } catch (err) {
        reconnectTimeout = setTimeout(connect, 1500);
      }
    };

    connect();

    return () => {
      isUnmounted = true;
      if (reconnectTimeout) clearTimeout(reconnectTimeout);
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
      inFlightRef.current = false;
      isEncodingRef.current = false;
    };
  }, [serverBaseUrl, mode, enabled]);

  // Low-latency backpressure-aware binary frame sender (never builds a frame queue)
  const sendFrame = useCallback((canvas: HTMLCanvasElement) => {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;

    // Skip if still encoding or socket write buffer is backed up
    if (isEncodingRef.current || ws.bufferedAmount > 65536) {
      return;
    }

    // Strict 1-in-flight flow control with 350ms safety watchdog
    const now = performance.now();
    if (inFlightRef.current && now - lastSentAtRef.current < 350) {
      return;
    }

    isEncodingRef.current = true;
    inFlightRef.current = true;
    lastSentAtRef.current = now;

    canvas.toBlob(
      (blob) => {
        isEncodingRef.current = false;
        if (!blob || !ws || ws.readyState !== WebSocket.OPEN) {
          inFlightRef.current = false;
          return;
        }

        blob
          .arrayBuffer()
          .then((buf) => {
            if (ws && ws.readyState === WebSocket.OPEN) {
              ws.send(buf);
            } else {
              inFlightRef.current = false;
            }
          })
          .catch(() => {
            inFlightRef.current = false;
          });
      },
      'image/jpeg',
      0.6
    );
  }, []);

  // Mode A Manual Controls
  const startRecord = useCallback(() => {
    const ws = wsRef.current;
    pendingRecordAckUntilRef.current = performance.now() + 1000;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'start_record' }));
    }
    setIsolatedState('RECORDING');
    setIsolatedPrediction(null);
    setModelDiagnostic((prev) =>
      prev ? { ...prev, buffer_frames: 0, input_shape: '(1, 0, 294)' } : prev
    );
  }, []);

  const stopRecord = useCallback(() => {
    const ws = wsRef.current;
    pendingRecordAckUntilRef.current = 0;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'stop_record' }));
    }
    setIsolatedState('INFERRING');
  }, []);

  // Mode B Manual Commit (immediately finalize & translate active continuous utterance)
  const commitUtterance = useCallback(() => {
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'commit_utterance' }));
    }
  }, []);

  // Runtime config update (sensitivity + LLM provider)
  const sendConfig = useCallback(
    (config: {
      sensitivity?: 'low' | 'medium' | 'high';
      llm_enabled?: boolean;
      llm_provider?: 'none' | 'gemini' | 'ollama';
    }) => {
      const ws = wsRef.current;
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'config', ...config }));
      }
    },
    []
  );

  // Session reset
  const resetSession = useCallback(() => {
    const ws = wsRef.current;
    pendingRecordAckUntilRef.current = 0;
    inFlightRef.current = false;
    prevFsmStateRef.current = 'IDLE';
    lastCommitSigRef.current = '';
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'reset' }));
    }
    landmarksRef.current = null;
    setLiveGloss([]);
    setStableTokens([]);
    setFinalGloss([]);
    setFinalSentence('');
    setUtteranceHistory([]);
    setRawTokens([]);
    setCollapsedTokens([]);
    setLandmarks(null);
    setFsmState('IDLE');
    setIsolatedState('IDLE');
    setIsolatedPrediction(null);
    setEndpointReason(null);
  }, []);

  return {
    isConnected,
    mode,
    // Mode A state
    isolatedState,
    isolatedPrediction,
    startRecord,
    stopRecord,
    // Mode B state
    fsmState,
    liveGloss,
    stableTokens,
    finalGloss,
    finalSentence,
    utteranceHistory,
    endpointReason,
    commitUtterance,
    // Common telemetry
    rawTokens,
    collapsedTokens,
    landmarks,
    landmarksRef,
    landmarksDiagnostic,
    modelDiagnostic,
    metrics,
    sendFrame,
    sendConfig,
    resetSession,
  };
}
