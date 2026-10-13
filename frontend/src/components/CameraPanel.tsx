import React, { useRef, useEffect } from 'react';
import { Camera, AlertTriangle, Eye, EyeOff, Sparkles } from 'lucide-react';
import { FSMStatus, LandmarksData, NormalizedPoint } from '../types';

interface CameraPanelProps {
  videoRef: React.RefObject<HTMLVideoElement>;
  isStreaming: boolean;
  mirrorPreview: boolean;
  showFps: boolean;
  fps: number;
  fsmState: FSMStatus;
  landmarks: LandmarksData | null;
  showLandmarks: boolean;
  onToggleLandmarks: () => void;
  error: string | null;
  onStartCamera: () => void;
}

// MediaPipe Hand bone connection topology
const HAND_CONNECTIONS = [
  // Palm base
  [0, 1], [0, 5], [0, 17], [5, 9], [9, 13], [13, 17],
  // Thumb
  [1, 2], [2, 3], [3, 4],
  // Index
  [5, 6], [6, 7], [7, 8],
  // Middle
  [9, 10], [10, 11], [11, 12],
  // Ring
  [13, 14], [14, 15], [15, 16],
  // Pinky
  [17, 18], [18, 19], [19, 20],
];

export const CameraPanel: React.FC<CameraPanelProps> = ({
  videoRef,
  isStreaming,
  mirrorPreview,
  showFps,
  fps,
  fsmState,
  landmarks,
  showLandmarks,
  onToggleLandmarks,
  error,
  onStartCamera,
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  // Map technical FSM states to friendly user terms
  const stateLabels: Record<FSMStatus, { text: string; color: string; dot: string }> = {
    IDLE: { text: 'Sẵn sàng', color: 'bg-slate-100 text-slate-700 border-slate-300', dot: 'bg-slate-400' },
    STARTING: { text: 'Phát hiện ký hiệu', color: 'bg-blue-50 text-blue-700 border-blue-300', dot: 'bg-blue-500 animate-ping' },
    SIGNING: { text: 'Đang ký hiệu...', color: 'bg-sky-50 text-sky-700 border-sky-300 animate-glow', dot: 'bg-sky-500 animate-pulse' },
    POSSIBLE_END: { text: 'Đang chờ hoàn tất', color: 'bg-amber-50 text-amber-700 border-amber-300', dot: 'bg-amber-500' },
    END_OF_UTTERANCE: { text: 'Hoàn tất phát ngôn', color: 'bg-emerald-50 text-emerald-700 border-emerald-300', dot: 'bg-emerald-500' },
    COOLDOWN: { text: 'Đang làm mới', color: 'bg-slate-100 text-slate-600 border-slate-300', dot: 'bg-slate-400' },
  };

  const currentLabel = stateLabels[fsmState] || stateLabels.IDLE;

  // Render Skeleton on Canvas Overlay
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Sync canvas buffer dimensions with video element
    const video = videoRef.current;
    if (video && video.videoWidth > 0 && video.videoHeight > 0) {
      if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
      }
    }

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (!isStreaming || !showLandmarks || !landmarks) {
      return;
    }

    const w = canvas.width;
    const h = canvas.height;

    const drawHand = (joints: NormalizedPoint[], strokeColor: string, jointColor: string, tipColor: string) => {
      if (!joints || joints.length < 21) return;

      // Draw bones
      ctx.lineWidth = 2.5;
      ctx.strokeStyle = strokeColor;
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';

      for (const [startIdx, endIdx] of HAND_CONNECTIONS) {
        const p1 = joints[startIdx];
        const p2 = joints[endIdx];
        if (!p1 || !p2) continue;

        ctx.beginPath();
        ctx.moveTo(p1.x * w, p1.y * h);
        ctx.lineTo(p2.x * w, p2.y * h);
        ctx.stroke();
      }

      // Draw joints
      for (let i = 0; i < joints.length; i++) {
        const pt = joints[i];
        if (!pt) continue;

        const isTip = [4, 8, 12, 16, 20].includes(i);
        const radius = isTip ? 4.5 : 3.0;

        ctx.beginPath();
        ctx.arc(pt.x * w, pt.y * h, radius, 0, 2 * Math.PI);
        ctx.fillStyle = isTip ? tipColor : jointColor;
        ctx.fill();
        ctx.lineWidth = 1;
        ctx.strokeStyle = '#ffffff';
        ctx.stroke();
      }
    };

    // 1. Draw Pose shoulders and arms
    if (landmarks.pose && landmarks.pose.length > 16) {
      const p = landmarks.pose;
      const leftSh = p[11];
      const rightSh = p[12];
      const leftElbow = p[13];
      const rightElbow = p[14];
      const leftWrist = p[15];
      const rightWrist = p[16];

      ctx.lineWidth = 3;
      ctx.strokeStyle = 'rgba(129, 140, 248, 0.8)'; // Indigo/Slate
      ctx.lineCap = 'round';

      // Shoulder bar
      if (leftSh && rightSh) {
        ctx.beginPath();
        ctx.moveTo(leftSh.x * w, leftSh.y * h);
        ctx.lineTo(rightSh.x * w, rightSh.y * h);
        ctx.stroke();

        // Shoulder joint anchors
        ctx.beginPath();
        ctx.arc(leftSh.x * w, leftSh.y * h, 5, 0, 2 * Math.PI);
        ctx.arc(rightSh.x * w, rightSh.y * h, 5, 0, 2 * Math.PI);
        ctx.fillStyle = '#6366f1';
        ctx.fill();
      }

      // Left arm
      if (leftSh && leftElbow && leftWrist) {
        ctx.strokeStyle = 'rgba(16, 185, 129, 0.6)'; // Green
        ctx.beginPath();
        ctx.moveTo(leftSh.x * w, leftSh.y * h);
        ctx.lineTo(leftElbow.x * w, leftElbow.y * h);
        ctx.lineTo(leftWrist.x * w, leftWrist.y * h);
        ctx.stroke();
      }

      // Right arm
      if (rightSh && rightElbow && rightWrist) {
        ctx.strokeStyle = 'rgba(14, 165, 233, 0.6)'; // Sky
        ctx.beginPath();
        ctx.moveTo(rightSh.x * w, rightSh.y * h);
        ctx.lineTo(rightElbow.x * w, rightElbow.y * h);
        ctx.lineTo(rightWrist.x * w, rightWrist.y * h);
        ctx.stroke();
      }
    }

    // 2. Draw Left Hand (Emerald palette)
    if (landmarks.left_hand) {
      drawHand(
        landmarks.left_hand,
        'rgba(16, 185, 129, 0.85)',
        '#10b981',
        '#34d399'
      );
    }

    // 3. Draw Right Hand (Sky/Cyan dominant palette)
    if (landmarks.right_hand) {
      drawHand(
        landmarks.right_hand,
        'rgba(14, 165, 233, 0.85)',
        '#0284c7',
        '#38bdf8'
      );
    }
  }, [landmarks, isStreaming, showLandmarks]);

  return (
    <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-4 sm:p-5 flex flex-col h-full">
      {/* Panel Header */}
      <div className="flex items-center justify-between mb-3 flex-wrap gap-2">
        <div className="flex items-center space-x-2">
          <Camera className="w-5 h-5 text-sky-600" />
          <h2 className="text-base font-bold text-slate-800">Khung hình Camera</h2>
        </div>

        {/* Action Controls & Badges */}
        <div className="flex items-center space-x-2">
          {/* Landmark Toggle Button */}
          <button
            type="button"
            onClick={onToggleLandmarks}
            className={`flex items-center space-x-1.5 px-3 py-1 rounded-full text-xs font-semibold border transition-all ${
              showLandmarks
                ? 'bg-sky-50 text-sky-700 border-sky-300 shadow-sm'
                : 'bg-slate-100 text-slate-500 border-slate-200 hover:bg-slate-200'
            }`}
            title="Bật/Tắt hiển thị khung xương khớp tay và tư thế"
          >
            {showLandmarks ? <Eye className="w-3.5 h-3.5 text-sky-600" /> : <EyeOff className="w-3.5 h-3.5" />}
            <span>Khung xương: {showLandmarks ? 'BẬT' : 'TẮT'}</span>
          </button>

          {/* FSM State Badge */}
          <div className={`flex items-center space-x-2 px-3 py-1 rounded-full text-xs font-semibold border ${currentLabel.color}`}>
            <span className={`w-2 h-2 rounded-full ${currentLabel.dot}`} />
            <span>{currentLabel.text}</span>
          </div>

          {/* FPS Badge */}
          {showFps && isStreaming && (
            <div className="text-xs font-mono font-medium px-2 py-1 rounded-lg bg-slate-100 text-slate-600 border border-slate-200">
              {fps.toFixed(1)} FPS
            </div>
          )}
        </div>
      </div>

      {/* Video Viewport Container */}
      <div className="relative flex-1 min-h-[360px] sm:min-h-[420px] bg-slate-900 rounded-xl overflow-hidden flex items-center justify-center border border-slate-800 shadow-inner">
        {error ? (
          <div className="text-center p-6 max-w-sm">
            <div className="w-12 h-12 rounded-full bg-rose-500/20 text-rose-400 flex items-center justify-center mx-auto mb-3">
              <AlertTriangle className="w-6 h-6" />
            </div>
            <p className="text-sm text-slate-300 font-medium mb-4">{error}</p>
            <button
              onClick={onStartCamera}
              className="px-4 py-2 text-xs font-semibold rounded-lg bg-sky-600 text-white hover:bg-sky-500 transition-colors shadow-sm"
            >
              Thử lại camera
            </button>
          </div>
        ) : !isStreaming ? (
          <div className="text-center p-6">
            <div className="w-16 h-16 rounded-2xl bg-slate-800/80 border border-slate-700 flex items-center justify-center mx-auto mb-4 text-slate-400">
              <Camera className="w-8 h-8" />
            </div>
            <p className="text-sm text-slate-300 font-semibold mb-1">Camera chưa được kích hoạt</p>
            <p className="text-xs text-slate-400 max-w-xs mx-auto mb-4">
              Nhấn "Bắt đầu nhận diện" để cấp quyền và mở máy ảnh của bạn.
            </p>
            <button
              onClick={onStartCamera}
              className="px-5 py-2.5 text-xs font-bold rounded-xl bg-sky-600 text-white hover:bg-sky-500 transition-all shadow-md shadow-sky-600/30 active:scale-95"
            >
              Bật Camera
            </button>
          </div>
        ) : null}

        {/* Realtime Video Stream */}
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className={`w-full h-full object-cover ${mirrorPreview ? '-scale-x-100' : ''} ${
            !isStreaming ? 'hidden' : ''
          }`}
        />

        {/* Realtime Landmark Skeleton Canvas Overlay */}
        <canvas
          ref={canvasRef}
          width={640}
          height={360}
          className={`absolute inset-0 w-full h-full object-cover pointer-events-none ${
            mirrorPreview ? '-scale-x-100' : ''
          } ${!isStreaming ? 'hidden' : ''}`}
        />

        {/* Realtime HUD Indicator Pills on top-left of video */}
        {isStreaming && (
          <div className="absolute top-3 left-3 flex items-center space-x-2 pointer-events-none flex-wrap gap-1">
            {/* Right Hand Detection Badge */}
            <div
              className={`flex items-center space-x-1.5 px-2.5 py-1 rounded-lg backdrop-blur-md text-[11px] font-semibold border shadow-sm transition-all ${
                landmarks?.has_right_hand
                  ? 'bg-sky-950/70 border-sky-400/40 text-sky-300'
                  : 'bg-slate-950/60 border-white/10 text-slate-400'
              }`}
            >
              <span
                className={`w-2 h-2 rounded-full ${
                  landmarks?.has_right_hand ? 'bg-sky-400 animate-pulse' : 'bg-slate-500'
                }`}
              />
              <span>Tay phải</span>
            </div>

            {/* Left Hand Detection Badge */}
            <div
              className={`flex items-center space-x-1.5 px-2.5 py-1 rounded-lg backdrop-blur-md text-[11px] font-semibold border shadow-sm transition-all ${
                landmarks?.has_left_hand
                  ? 'bg-emerald-950/70 border-emerald-400/40 text-emerald-300'
                  : 'bg-slate-950/60 border-white/10 text-slate-400'
              }`}
            >
              <span
                className={`w-2 h-2 rounded-full ${
                  landmarks?.has_left_hand ? 'bg-emerald-400 animate-pulse' : 'bg-slate-500'
                }`}
              />
              <span>Tay trái</span>
            </div>

            {/* Signing Space Badge */}
            <div
              className={`flex items-center space-x-1.5 px-2.5 py-1 rounded-lg backdrop-blur-md text-[11px] font-semibold border shadow-sm transition-all ${
                landmarks?.in_signing_space
                  ? 'bg-indigo-950/70 border-indigo-400/40 text-indigo-300'
                  : 'bg-slate-950/60 border-white/10 text-slate-500'
              }`}
            >
              <Sparkles className={`w-3 h-3 ${landmarks?.in_signing_space ? 'text-indigo-300' : 'text-slate-500'}`} />
              <span>Vùng ký hiệu</span>
            </div>
          </div>
        )}

        {/* Guide overlay on bottom of camera */}
        {isStreaming && (
          <div className="absolute bottom-3 inset-x-3 text-center pointer-events-none">
            <span className="inline-block px-3 py-1.5 rounded-lg bg-slate-950/75 backdrop-blur-md text-[11px] text-slate-200 font-medium border border-white/10 shadow-sm">
              💡 Giữ phần thân trên và hai bàn tay trong khung hình để mô hình bắt chuẩn nhất
            </span>
          </div>
        )}
      </div>
    </div>
  );
};
