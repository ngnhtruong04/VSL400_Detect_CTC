import React from 'react';
import { X, Code2, Gauge, Layers, Activity, Target, CheckCircle2 } from 'lucide-react';
import { MetricsData, FSMStatus, RecognitionMode, ModelDiagnostic, LandmarksDiagnostic, IsolatedPrediction } from '../types';

interface DevMetricsModalProps {
  isOpen: boolean;
  onClose: () => void;
  mode: RecognitionMode;
  metrics: MetricsData;
  fsmState: FSMStatus;
  rawTokens: number[];
  collapsedTokens: number[];
  liveGloss: string[];
  stableTokens?: string[];
  endpointReason?: string | null;
  modelDiagnostic?: ModelDiagnostic | null;
  landmarksDiagnostic?: LandmarksDiagnostic | null;
  isolatedPrediction?: IsolatedPrediction | null;
}

export const DevMetricsModal: React.FC<DevMetricsModalProps> = ({
  isOpen,
  onClose,
  mode,
  metrics,
  fsmState,
  rawTokens,
  collapsedTokens,
  liveGloss,
  stableTokens = [],
  endpointReason,
  modelDiagnostic,
  landmarksDiagnostic,
  isolatedPrediction,
}) => {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/60 backdrop-blur-sm animate-fade-in">
      <div className="bg-white rounded-3xl shadow-xl border border-slate-200 w-full max-w-2xl overflow-hidden flex flex-col max-h-[90vh]">
        {/* Header */}
        <div className="px-6 py-4 border-b border-slate-100 flex items-center justify-between bg-slate-50/50">
          <div className="flex items-center space-x-2">
            <Code2 className="w-5 h-5 text-sky-600" />
            <div>
              <h3 className="text-base font-bold text-slate-800">Chế độ Nhà phát triển (Developer Diagnostics)</h3>
              <p className="text-[11px] text-slate-400">
                Active Mode: <b className="text-sky-600 uppercase font-mono">{mode}</b>
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-xl text-slate-400 hover:text-slate-600 hover:bg-slate-100 transition-colors cursor-pointer"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content */}
        <div className="p-6 overflow-y-auto space-y-5 font-mono text-xs">
          {/* 1. Model Metadata */}
          <div className="bg-slate-50 rounded-xl p-4 border border-slate-200">
            <h4 className="font-bold text-slate-700 mb-2 flex items-center gap-1.5 font-sans">
              <Layers className="w-4 h-4 text-indigo-500" />
              Locked Architecture & Checkpoint
            </h4>
            <div className="grid grid-cols-2 gap-2 text-[11px] text-slate-600">
              <div><span className="text-slate-400">Model:</span> HandsTCNTransformerCTC</div>
              <div><span className="text-slate-400">Checkpoint:</span> best_seed_20260827.pt</div>
              <div><span className="text-slate-400">Trainable Params:</span> 5,442,961 (Exact)</div>
              <div><span className="text-slate-400">Input Contract:</span> (B, T, 294)</div>
              <div><span className="text-slate-400">Glosses / Classes:</span> 400 Glosses / 401 CTC</div>
              <div><span className="text-slate-400">CTC Blank ID:</span> 400</div>
            </div>
          </div>

          {/* 2. Latency Breakdown */}
          <div className="bg-slate-50 rounded-xl p-4 border border-slate-200">
            <h4 className="font-bold text-slate-700 mb-2 flex items-center gap-1.5 font-sans">
              <Gauge className="w-4 h-4 text-emerald-500" />
              Độ Trễ Thành Phần (Pipeline Latency)
            </h4>
            <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 text-center">
              <div className="p-2.5 rounded-lg bg-white border border-slate-200 shadow-sm">
                <div className="text-[10px] text-slate-400 uppercase font-sans">MediaPipe</div>
                <div className="text-sm font-bold text-slate-800">{metrics.mediapipe_ms} ms</div>
              </div>
              <div className="p-2.5 rounded-lg bg-white border border-slate-200 shadow-sm">
                <div className="text-[10px] text-slate-400 uppercase font-sans">Preprocessing</div>
                <div className="text-sm font-bold text-slate-800">{metrics.preprocessing_ms} ms</div>
              </div>
              <div className="p-2.5 rounded-lg bg-white border border-slate-200 shadow-sm">
                <div className="text-[10px] text-slate-400 uppercase font-sans">Model Forward</div>
                <div className="text-sm font-bold text-slate-800">{metrics.inference_ms} ms</div>
              </div>
              <div className="p-2.5 rounded-lg bg-white border border-slate-200 shadow-sm">
                <div className="text-[10px] text-slate-400 uppercase font-sans">CTC Decode</div>
                <div className="text-sm font-bold text-slate-800">{metrics.decode_ms} ms</div>
              </div>
              <div className="p-2.5 rounded-lg bg-white border border-slate-200 shadow-sm">
                <div className="text-[10px] text-slate-400 uppercase font-sans">Total Pipeline</div>
                <div className="text-sm font-bold text-sky-600">{metrics.total_ms} ms</div>
              </div>
              <div className="p-2.5 rounded-lg bg-white border border-slate-200 shadow-sm">
                <div className="text-[10px] text-slate-400 uppercase font-sans">Camera Rate</div>
                <div className="text-sm font-bold text-slate-800">{metrics.fps.toFixed(1)} FPS</div>
              </div>
            </div>
          </div>

          {/* 3. Mode-Specific Telemetry */}
          {mode === 'isolated' ? (
            <div className="bg-slate-50 rounded-xl p-4 border border-slate-200">
              <h4 className="font-bold text-slate-700 mb-2 flex items-center gap-1.5 font-sans">
                <Target className="w-4 h-4 text-sky-500" />
                Mode A: Isolated Diagnostics
              </h4>
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 text-slate-600 text-[11px]">
                <div><span className="text-slate-400">Buffer Frames:</span> {modelDiagnostic?.buffer_frames || 0}</div>
                <div><span className="text-slate-400">Input Shape:</span> {modelDiagnostic?.input_shape || '(1, 0, 294)'}</div>
                <div><span className="text-slate-400">Blank Ratio:</span> {((modelDiagnostic?.blank_ratio || 0) * 100).toFixed(1)}%</div>
                <div><span className="text-slate-400">NaN Count:</span> {modelDiagnostic?.nan_count || 0}</div>
                <div><span className="text-slate-400">Inf Count:</span> {modelDiagnostic?.inf_count || 0}</div>
                <div><span className="text-slate-400">Predicted Gloss:</span> <b className="text-sky-600">{isolatedPrediction?.gloss || 'None'}</b></div>
              </div>
            </div>
          ) : (
            <div className="bg-slate-50 rounded-xl p-4 border border-slate-200">
              <h4 className="font-bold text-slate-700 mb-2 flex items-center gap-1.5 font-sans">
                <Activity className="w-4 h-4 text-indigo-500" />
                Mode B: Continuous FSM & Utterance Diagnostics
              </h4>
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 text-slate-600 text-[11px]">
                <div><span className="text-slate-400">FSM State:</span> <b className="text-indigo-600 font-sans">{fsmState}</b></div>
                <div><span className="text-slate-400">Utterance Frames:</span> {modelDiagnostic?.buffer_frames || 0}</div>
                <div><span className="text-slate-400">Motion Energy:</span> {metrics.motion_energy.toFixed(4)}</div>
                <div><span className="text-slate-400">Blank Ratio:</span> {((modelDiagnostic?.blank_ratio || 0) * 100).toFixed(1)}%</div>
                <div><span className="text-slate-400">Endpoint Reason:</span> {endpointReason || 'None'}</div>
                <div><span className="text-slate-400">Stable Tokens:</span> {stableTokens.join(', ') || 'None'}</div>
              </div>
            </div>
          )}

          {/* 4. Landmarks Parity & Validity */}
          <div className="bg-slate-50 rounded-xl p-4 border border-slate-200">
            <h4 className="font-bold text-slate-700 mb-2 flex items-center gap-1.5 font-sans">
              <CheckCircle2 className="w-4 h-4 text-teal-500" />
              Landmarks & Normalization Parity
            </h4>
            <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 text-slate-600 text-[11px]">
              <div><span className="text-slate-400">Left Hand Valid:</span> {Math.round((landmarksDiagnostic?.left_valid_ratio || 0) * 100)}%</div>
              <div><span className="text-slate-400">Right Hand Valid:</span> {Math.round((landmarksDiagnostic?.right_valid_ratio || 0) * 100)}%</div>
              <div><span className="text-slate-400">Overall Valid:</span> {Math.round((landmarksDiagnostic?.overall_valid_ratio || 0) * 100)}%</div>
              <div><span className="text-slate-400">Has Left Hand:</span> {landmarksDiagnostic?.has_left_hand ? 'True' : 'False'}</div>
              <div><span className="text-slate-400">Has Right Hand:</span> {landmarksDiagnostic?.has_right_hand ? 'True' : 'False'}</div>
              <div><span className="text-slate-400">In Signing Space:</span> {landmarksDiagnostic?.in_signing_space ? 'True' : 'False'}</div>
            </div>
          </div>

          {/* 5. CTC Tokens */}
          <div className="space-y-2">
            <div>
              <span className="text-slate-400 block mb-1">Raw Token IDs:</span>
              <div className="p-2 rounded-lg bg-slate-900 text-slate-300 text-[10px] break-all max-h-16 overflow-y-auto">
                {rawTokens.length > 0 ? rawTokens.join(', ') : 'None'}
              </div>
            </div>
            <div>
              <span className="text-slate-400 block mb-1">Collapsed Token IDs:</span>
              <div className="p-2 rounded-lg bg-slate-900 text-emerald-400 text-[10px] break-all">
                {collapsedTokens.length > 0 ? collapsedTokens.join(', ') : 'None'}
              </div>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="px-6 py-4 border-t border-slate-100 bg-slate-50/50 flex justify-end">
          <button
            onClick={onClose}
            className="px-5 py-2 text-xs font-bold rounded-xl bg-slate-900 text-white hover:bg-slate-800 transition-colors shadow-sm font-sans cursor-pointer"
          >
            Đóng
          </button>
        </div>
      </div>
    </div>
  );
};
