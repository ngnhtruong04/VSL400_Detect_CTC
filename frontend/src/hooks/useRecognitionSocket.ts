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
} from '../types';

interface UseRecognitionSocketProps {
  serverBaseUrl?: string;
  mode: RecognitionMode;
  enabled?: boolean;
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
  const [endpointReason, setEndpointReason] = useState<string | null>(null);

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

  const wsRef = useRef<WebSocket | null>(null);
  const isSendingFrameRef = useRef<boolean>(false);

  // Connect WebSocket when enabled or mode changes
  useEffect(() => {
    if (!enabled) {
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
      setIsConnected(false);
      return;
    }

    let isUnmounted = false;
    let reconnectTimeout: any = null;

    const fullUrl = `${serverBaseUrl}?mode=${mode}`;

    const connect = () => {
      try {
        const ws = new WebSocket(fullUrl);
        ws.binaryType = 'arraybuffer';
        wsRef.current = ws;

        ws.onopen = () => {
          if (isUnmounted) return;
          setIsConnected(true);
        };

        ws.onclose = () => {
          if (isUnmounted) return;
          setIsConnected(false);
          reconnectTimeout = setTimeout(connect, 2000);
        };

        ws.onerror = (err) => {
          console.warn('WebSocket error:', err);
          ws.close();
        };

        ws.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);

            if (data.type === 'isolated_recognition') {
              const msg = data as IsolatedRecognitionMessage;
              setIsolatedState(msg.state);
              if (msg.prediction) {
                setIsolatedPrediction(msg.prediction);
                if (msg.prediction.gloss) {
                  setLiveGloss([msg.prediction.gloss]);
                }
              }
              if (msg.landmarks) {
                setLandmarksDiagnostic(msg.landmarks);
                setLandmarks({
                  left_hand: msg.landmarks.left_hand,
                  right_hand: msg.landmarks.right_hand,
                  pose: msg.landmarks.pose,
                  has_left_hand: msg.landmarks.has_left_hand,
                  has_right_hand: msg.landmarks.has_right_hand,
                  in_signing_space: msg.landmarks.in_signing_space,
                  left_valid_ratio: msg.landmarks.left_valid_ratio,
                  right_valid_ratio: msg.landmarks.right_valid_ratio,
                  overall_valid_ratio: msg.landmarks.overall_valid_ratio,
                });
              }
              if (msg.model) {
                setModelDiagnostic(msg.model);
                setMetrics((prev) => ({
                  ...prev,
                  fps: msg.fps,
                  total_ms: msg.total_ms,
                  inference_ms: msg.model.forward_ms,
                  decode_ms: msg.model.decode_ms,
                  motion_energy: msg.motion_energy,
                  blank_ratio: msg.model.blank_ratio,
                  buffer_frames: msg.model.buffer_frames,
                }));
              }
            } else if (data.type === 'continuous_recognition') {
              const msg = data as ContinuousRecognitionMessage;
              setFsmState(msg.state);
              setLiveGloss(msg.continuous.live_tokens || []);
              setStableTokens(msg.continuous.stable_tokens || []);
              if (msg.continuous.final_tokens && msg.continuous.final_tokens.length > 0) {
                setFinalGloss(msg.continuous.final_tokens);
              }
              if (msg.continuous.final_sentence) {
                setFinalSentence(msg.continuous.final_sentence);
              }
              setRawTokens(msg.continuous.raw_token_ids || []);
              setCollapsedTokens(msg.continuous.collapsed_token_ids || []);
              setEndpointReason(msg.continuous.endpoint_reason || null);

              if (msg.landmarks) {
                setLandmarksDiagnostic(msg.landmarks);
                setLandmarks({
                  left_hand: msg.landmarks.left_hand,
                  right_hand: msg.landmarks.right_hand,
                  pose: msg.landmarks.pose,
                  has_left_hand: msg.landmarks.has_left_hand,
                  has_right_hand: msg.landmarks.has_right_hand,
                  in_signing_space: msg.landmarks.in_signing_space,
                  left_valid_ratio: msg.landmarks.left_valid_ratio,
                  right_valid_ratio: msg.landmarks.right_valid_ratio,
                  overall_valid_ratio: msg.landmarks.overall_valid_ratio,
                });
              }

              if (msg.model) {
                setModelDiagnostic(msg.model);
                setMetrics((prev) => ({
                  ...prev,
                  fps: msg.fps,
                  total_ms: msg.total_ms,
                  inference_ms: msg.model.forward_ms,
                  decode_ms: msg.model.decode_ms,
                  motion_energy: msg.motion_energy,
                  blank_ratio: msg.model.blank_ratio,
                  buffer_frames: msg.continuous.utterance_frames,
                }));
              }
            } else if (data.type === 'recognition') {
              // Backward compatibility
              const msg = data as RecognitionMessage;
              setFsmState(msg.state);
              setLiveGloss(msg.live_gloss || []);
              if (msg.raw_tokens) setRawTokens(msg.raw_tokens);
              if (msg.collapsed_tokens) setCollapsedTokens(msg.collapsed_tokens);
              if (msg.landmarks !== undefined) setLandmarks(msg.landmarks);
              if (msg.metrics) setMetrics(msg.metrics);
            } else if (data.type === 'utterance_complete') {
              // Backward compatibility
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
        reconnectTimeout = setTimeout(connect, 2000);
      }
    };

    connect();

    return () => {
      isUnmounted = true;
      if (reconnectTimeout) clearTimeout(reconnectTimeout);
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [serverBaseUrl, mode, enabled]);

  // Fast binary frame sender
  const sendFrame = useCallback((canvas: HTMLCanvasElement) => {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;

    if (isSendingFrameRef.current) {
      return;
    }

    isSendingFrameRef.current = true;

    canvas.toBlob(
      (blob) => {
        isSendingFrameRef.current = false;
        if (!blob || !ws || ws.readyState !== WebSocket.OPEN) return;

        blob.arrayBuffer().then((buf) => {
          if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(buf);
          }
        });
      },
      'image/jpeg',
      0.75
    );
  }, []);

  // Mode A Manual Controls
  const startRecord = useCallback(() => {
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'start_record' }));
    }
    setIsolatedState('RECORDING');
    setIsolatedPrediction(null);
  }, []);

  const stopRecord = useCallback(() => {
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'stop_record' }));
    }
    setIsolatedState('INFERRING');
  }, []);

  // Session reset
  const resetSession = useCallback(() => {
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'reset' }));
    }
    setLiveGloss([]);
    setStableTokens([]);
    setFinalGloss([]);
    setFinalSentence('');
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
    endpointReason,
    // Common telemetry
    rawTokens,
    collapsedTokens,
    landmarks,
    landmarksDiagnostic,
    modelDiagnostic,
    metrics,
    sendFrame,
    resetSession,
  };
}
