import React from 'react';
import { Sparkles, ArrowRight, Layers, CheckCircle2, Activity, Info, Send } from 'lucide-react';
import { FSMStatus, ModelDiagnostic, LandmarksDiagnostic } from '../types';

interface RecognitionPanelProps {
  liveGloss: string[];
  stableTokens?: string[];
  finalGloss: string[];
  fsmState?: FSMStatus;
  endpointReason?: string | null;
  modelDiagnostic?: ModelDiagnostic | null;
  landmarksDiagnostic?: LandmarksDiagnostic | null;
  isStreaming?: boolean;
  onCommitUtterance?: () => void;
}

export const RecognitionPanel: React.FC<RecognitionPanelProps> = ({
  liveGloss,
  stableTokens = [],
  finalGloss,
  fsmState = 'IDLE',
  endpointReason,
  modelDiagnostic,
  landmarksDiagnostic,
  isStreaming = false,
  onCommitUtterance,
}) => {
  const canCommit = isStreaming && (liveGloss.length > 0 || (modelDiagnostic?.buffer_frames || 0) >= 15);

  return (
    <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-4 sm:p-5 flex flex-col space-y-4">
      {/* 1. FSM Lifecycle Context Bar */}
      <div className="flex items-center justify-between pb-3 border-b border-slate-100 flex-wrap gap-2">
        <div className="flex items-center space-x-2">
          <Layers className="w-5 h-5 text-indigo-600" />
          <div>
            <h3 className="text-sm font-bold text-slate-800">Kiểm tra Continuous (Mode B)</h3>
            <p className="text-[11px] text-slate-500">Chuỗi ký hiệu liên tục và bộ đệm phát ngôn</p>
          </div>
        </div>

        <div className="flex items-center gap-2 flex-wrap">
          {onCommitUtterance && (
            <button
              type="button"
              disabled={!canCommit}
              onClick={onCommitUtterance}
              title="Chốt chuỗi đang ký và dịch ngay lập tức"
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-bold transition-all ${
                canCommit
                  ? 'bg-indigo-600 hover:bg-indigo-500 text-white shadow-sm cursor-pointer active:scale-95'
                  : 'bg-slate-100 text-slate-400 cursor-not-allowed'
              }`}
            >
              <Send className="w-3.5 h-3.5" />
              <span>Chốt & Dịch</span>
            </button>
          )}

          {endpointReason && (
            <span className="text-[11px] font-mono text-indigo-700 bg-indigo-50 border border-indigo-200 px-2 py-0.5 rounded-full flex items-center gap-1">
              <Info className="w-3 h-3 text-indigo-500" />
              Lý do chốt: {endpointReason}
            </span>
          )}
        </div>
      </div>

      {/* 2. Live Recognition Chips (Active utterance) */}
      <div>
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center space-x-2">
            <Sparkles className="w-4 h-4 text-sky-500 animate-pulse" />
            <h4 className="text-xs font-bold uppercase tracking-wider text-slate-500">
              Từ đang ký hiệu (Live Utterance Tokens)
            </h4>
          </div>
          {fsmState === 'SIGNING' && (
            <span className="text-[11px] font-semibold text-sky-600 animate-pulse flex items-center gap-1">
              <span className="w-1.5 h-1.5 rounded-full bg-sky-500" />
              Đang tích lũy phát ngôn...
            </span>
          )}
        </div>

        <div className="min-h-[52px] p-2.5 rounded-xl bg-slate-50 border border-slate-200 flex flex-wrap items-center gap-2">
          {liveGloss.length > 0 ? (
            liveGloss.map((gloss, idx) => {
              const isStable = stableTokens.includes(gloss);
              return (
                <span
                  key={`${gloss}-${idx}`}
                  className={`inline-flex items-center px-3 py-1 rounded-lg text-sm font-bold shadow-sm transition-all animate-fade-in ${
                    isStable
                      ? 'bg-emerald-600 text-white shadow-emerald-600/20'
                      : 'bg-sky-500 text-white shadow-sky-500/20'
                  }`}
                  title={isStable ? 'Từ này đã ổn định qua các bước giải mã' : 'Từ đang phát sinh trong câu'}
                >
                  {isStable && <CheckCircle2 className="w-3.5 h-3.5 mr-1 text-emerald-200" />}
                  {gloss}
                </span>
              );
            })
          ) : (
            <span className="text-xs text-slate-400 italic font-normal">
              Chưa phát hiện ký hiệu trong phát ngôn...
            </span>
          )}
        </div>
      </div>

      {/* 3. Final Gloss Sequence Timeline */}
      <div>
        <div className="flex items-center space-x-2 mb-2">
          <CheckCircle2 className="w-4 h-4 text-indigo-500" />
          <h4 className="text-xs font-bold uppercase tracking-wider text-slate-500">
            Chuỗi ký hiệu hoàn chỉnh (Final Gloss Sequence)
          </h4>
        </div>

        <div className="min-h-[64px] p-3 rounded-xl bg-slate-50 border border-slate-200 flex flex-wrap items-center gap-2">
          {finalGloss.length > 0 ? (
            finalGloss.map((gloss, idx) => (
              <React.Fragment key={`${gloss}-${idx}`}>
                <span className="inline-flex items-center px-3 py-1.5 rounded-lg text-sm font-bold bg-white text-indigo-950 border border-indigo-300 shadow-sm">
                  {gloss}
                </span>
                {idx < finalGloss.length - 1 && (
                  <ArrowRight className="w-3.5 h-3.5 text-slate-400 shrink-0" />
                )}
              </React.Fragment>
            ))
          ) : (
            <span className="text-xs text-slate-400 italic font-normal">
              Chuỗi ký hiệu hoàn tất sẽ xuất hiện ở đây khi kết thúc câu...
            </span>
          )}
        </div>
      </div>

      {/* 4. Realtime Continuous Diagnostics */}
      <div className="bg-slate-50 rounded-xl p-3 border border-slate-200">
        <div className="flex items-center space-x-1.5 mb-2 text-[11px] font-bold text-slate-600">
          <Activity className="w-3.5 h-3.5 text-slate-500" />
          <span>Bộ đệm Phát ngôn & Giám sát FSM</span>
        </div>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-[11px] font-mono">
          <div className="bg-white p-2 rounded-lg border border-slate-200">
            <span className="text-slate-400 block text-[10px]">Utterance Buffer</span>
            <span className="font-bold text-slate-800">{modelDiagnostic?.buffer_frames || 0} f</span>
          </div>
          <div className="bg-white p-2 rounded-lg border border-slate-200">
            <span className="text-slate-400 block text-[10px]">Blank Ratio</span>
            <span className="font-bold text-slate-800">
              {((modelDiagnostic?.blank_ratio || 0) * 100).toFixed(1)}%
            </span>
          </div>
          <div className="bg-white p-2 rounded-lg border border-slate-200">
            <span className="text-slate-400 block text-[10px]">Forward Latency</span>
            <span className="font-bold text-slate-800">{modelDiagnostic?.forward_ms || 0} ms</span>
          </div>
          <div className="bg-white p-2 rounded-lg border border-slate-200">
            <span className="text-slate-400 block text-[10px]">FSM State</span>
            <span className="font-bold text-indigo-600">{fsmState}</span>
          </div>
        </div>

        {landmarksDiagnostic && (
          <div className="mt-2 pt-2 border-t border-slate-200/80 flex items-center justify-between text-[11px] text-slate-500">
            <span>Tay Trái: <b>{Math.round((landmarksDiagnostic.left_valid_ratio || 0) * 100)}%</b></span>
            <span>Tay Phải: <b>{Math.round((landmarksDiagnostic.right_valid_ratio || 0) * 100)}%</b></span>
            <span>Vùng Ký: <b className={landmarksDiagnostic.in_signing_space ? 'text-emerald-600' : 'text-slate-400'}>{landmarksDiagnostic.in_signing_space ? 'Hợp lệ' : 'Ngoài vùng'}</b></span>
          </div>
        )}
      </div>
    </div>
  );
};
