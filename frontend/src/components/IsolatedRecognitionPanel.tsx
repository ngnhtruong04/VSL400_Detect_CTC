import React from 'react';
import { Target, Play, Square, RotateCcw, Award, BarChart3, Clock, AlertCircle } from 'lucide-react';
import { IsolatedPrediction, ModelDiagnostic, LandmarksDiagnostic } from '../types';

interface IsolatedRecognitionPanelProps {
  state: 'IDLE' | 'RECORDING' | 'INFERRING' | 'RESULT';
  prediction: IsolatedPrediction | null;
  modelDiagnostic: ModelDiagnostic | null;
  landmarksDiagnostic: LandmarksDiagnostic | null;
  onStartRecord: () => void;
  onStopRecord: () => void;
  onReset: () => void;
  isStreaming: boolean;
}

export const IsolatedRecognitionPanel: React.FC<IsolatedRecognitionPanelProps> = ({
  state,
  prediction,
  modelDiagnostic,
  landmarksDiagnostic,
  onStartRecord,
  onStopRecord,
  onReset,
  isStreaming,
}) => {
  const isRecording = state === 'RECORDING';
  const isInferring = state === 'INFERRING';

  return (
    <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-4 sm:p-5 flex flex-col space-y-4">
      {/* 1. Header & State Badge */}
      <div className="flex items-center justify-between flex-wrap gap-2 pb-3 border-b border-slate-100">
        <div className="flex items-center space-x-2">
          <Target className="w-5 h-5 text-sky-600" />
          <div>
            <h3 className="text-sm font-bold text-slate-800">Kiểm tra Gloss Đơn lẻ (Mode A)</h3>
            <p className="text-[11px] text-slate-500">Thu nhận 1 cử chỉ và tính toán Top-5 xác suất CTC</p>
          </div>
        </div>

        {/* State Pill */}
        <div
          className={`flex items-center space-x-1.5 px-3 py-1 rounded-full text-xs font-bold border ${
            isRecording
              ? 'bg-rose-50 text-rose-700 border-rose-300 animate-pulse'
              : isInferring
              ? 'bg-amber-50 text-amber-700 border-amber-300'
              : state === 'RESULT'
              ? 'bg-emerald-50 text-emerald-700 border-emerald-300'
              : 'bg-slate-100 text-slate-600 border-slate-200'
          }`}
        >
          <span
            className={`w-2 h-2 rounded-full ${
              isRecording
                ? 'bg-rose-500 animate-ping'
                : isInferring
                ? 'bg-amber-500 animate-spin'
                : state === 'RESULT'
                ? 'bg-emerald-500'
                : 'bg-slate-400'
            }`}
          />
          <span>
            {isRecording
              ? `Đang ghi cử chỉ (${modelDiagnostic?.buffer_frames || 0} frames)`
              : isInferring
              ? 'Đang tính toán...'
              : state === 'RESULT'
              ? 'Hoàn tất phân tích'
              : 'Sẵn sàng'}
          </span>
        </div>
      </div>

      {/* 2. Recording Controls */}
      <div className="flex items-center space-x-2">
        {!isRecording ? (
          <button
            type="button"
            disabled={!isStreaming || isInferring}
            onClick={onStartRecord}
            className={`flex-1 flex items-center justify-center space-x-2 py-3 px-4 rounded-xl font-bold text-sm shadow-md transition-all ${
              !isStreaming || isInferring
                ? 'bg-slate-200 text-slate-400 cursor-not-allowed shadow-none'
                : 'bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 text-white shadow-emerald-600/25 active:scale-98 cursor-pointer'
            }`}
          >
            <Play className="w-4 h-4 fill-white" />
            <span>Bắt đầu ghi (Record Gloss)</span>
          </button>
        ) : (
          <button
            type="button"
            onClick={onStopRecord}
            className="flex-1 flex items-center justify-center space-x-2 py-3 px-4 rounded-xl bg-gradient-to-r from-rose-600 to-red-600 hover:from-rose-500 hover:to-red-500 text-white font-bold text-sm shadow-md shadow-rose-600/25 active:scale-98 transition-all cursor-pointer"
          >
            <Square className="w-4 h-4 fill-white" />
            <span>Dừng & Nhận diện (Stop & Infer)</span>
          </button>
        )}

        <button
          type="button"
          onClick={onReset}
          title="Xóa kết quả và làm mới"
          className="p-3 rounded-xl border border-slate-200 hover:bg-slate-100 text-slate-600 transition-colors active:scale-95"
        >
          <RotateCcw className="w-4 h-4" />
        </button>
      </div>

      {/* Guide text */}
      <div className="text-[11px] text-slate-500 bg-slate-50 p-2.5 rounded-xl border border-slate-200/80">
        💡 <b>Cách dùng:</b> Nhấn <i>"Bắt đầu ghi"</i> (hoặc tự động khi tay đưa lên vùng ký hiệu), thực hiện đúng 1 từ ký hiệu trong 1–3 giây, sau đó nhấn <i>"Dừng & Nhận diện"</i> hoặc hạ tay xuống.
      </div>

      {/* 3. Primary Prediction Result */}
      <div className="p-4 rounded-2xl bg-gradient-to-br from-sky-50 to-blue-50/50 border border-sky-200">
        <div className="flex items-center justify-between mb-1">
          <span className="text-[11px] font-bold uppercase tracking-wider text-sky-800 flex items-center gap-1.5">
            <Award className="w-4 h-4 text-sky-600" />
            Predicted Gloss (Dự đoán hàng đầu)
          </span>
          {prediction && prediction.label_id >= 0 && (
            <span className="text-[11px] font-mono text-sky-700 bg-sky-200/60 px-2 py-0.5 rounded">
              ID: {prediction.label_id}
            </span>
          )}
        </div>

        <div className="min-h-[50px] flex items-center">
          {prediction && prediction.gloss ? (
            <div className="text-2xl sm:text-3xl font-black text-sky-950 tracking-wide uppercase">
              {prediction.gloss}
            </div>
          ) : isRecording ? (
            <div className="text-sm font-medium text-rose-600 animate-pulse">
              Đang ghi cử chỉ... hãy thực hiện từ ký hiệu của bạn
            </div>
          ) : isInferring ? (
            <div className="text-sm font-medium text-amber-600 animate-pulse">
              Đang chạy inference qua HandsTCNTransformerCTC...
            </div>
          ) : (
            <span className="text-xs text-slate-400 italic">
              Chưa có kết quả. Nhấn "Bắt đầu ghi" để thử từ đầu tiên.
            </span>
          )}
        </div>
      </div>

      {/* 4. Top-5 Ranking Table */}
      <div>
        <div className="flex items-center space-x-2 mb-2">
          <BarChart3 className="w-4 h-4 text-indigo-500" />
          <h4 className="text-xs font-bold uppercase tracking-wider text-slate-700">
            Top-5 Ứng viên Dự đoán (CTC Hypotheses)
          </h4>
        </div>

        <div className="overflow-hidden border border-slate-200 rounded-xl">
          <table className="w-full text-left text-xs border-collapse">
            <thead>
              <tr className="bg-slate-100/80 text-slate-600 border-b border-slate-200 text-[11px]">
                <th className="py-2 px-3 font-bold w-10">#</th>
                <th className="py-2 px-3 font-bold">Từ Ký Hiệu (Gloss)</th>
                <th className="py-2 px-3 font-bold w-14">ID</th>
                <th className="py-2 px-3 font-bold text-right">Độ tin cậy</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 bg-white">
              {prediction && prediction.top5 && prediction.top5.length > 0 ? (
                prediction.top5.map((c) => (
                  <tr
                    key={`${c.rank}-${c.label_id}`}
                    className={c.rank === 1 ? 'bg-sky-50/50 font-bold' : 'hover:bg-slate-50/60'}
                  >
                    <td className="py-2 px-3 text-slate-500">
                      {c.rank === 1 ? (
                        <span className="w-5 h-5 rounded-full bg-sky-600 text-white inline-flex items-center justify-center text-[10px]">
                          1
                        </span>
                      ) : (
                        c.rank
                      )}
                    </td>
                    <td className="py-2 px-3 text-slate-900 font-medium">
                      {c.gloss}
                    </td>
                    <td className="py-2 px-3 text-slate-500 font-mono text-[11px]">
                      {c.label_id}
                    </td>
                    <td className="py-2 px-3 text-right">
                      <div className="flex items-center justify-end space-x-2">
                        <div className="w-16 sm:w-24 bg-slate-100 rounded-full h-2 overflow-hidden">
                          <div
                            className={`h-2 rounded-full transition-all ${
                              c.rank === 1 ? 'bg-sky-500' : 'bg-indigo-400'
                            }`}
                            style={{ width: `${Math.min(100, Math.max(2, c.diagnostic_confidence))}%` }}
                          />
                        </div>
                        <span className="font-mono text-[11px] text-slate-700 w-10">
                          {c.diagnostic_confidence.toFixed(1)}%
                        </span>
                      </div>
                    </td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td colSpan={4} className="py-4 px-3 text-center text-slate-400 italic text-xs">
                    Chưa có ứng viên. Thực hiện ký hiệu để xem bảng xếp hạng.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* 5. Diagnostic Metrics Bar */}
      <div className="bg-slate-50 rounded-xl p-3 border border-slate-200">
        <div className="flex items-center space-x-1.5 mb-2 text-[11px] font-bold text-slate-600">
          <Clock className="w-3.5 h-3.5 text-slate-500" />
          <span>Thông số Kỹ thuật Đoạn Ký (Diagnostics)</span>
        </div>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-[11px] font-mono">
          <div className="bg-white p-2 rounded-lg border border-slate-200">
            <span className="text-slate-400 block text-[10px]">Buffer Frames</span>
            <span className="font-bold text-slate-800">{modelDiagnostic?.buffer_frames || 0} f</span>
          </div>
          <div className="bg-white p-2 rounded-lg border border-slate-200">
            <span className="text-slate-400 block text-[10px]">Input Shape</span>
            <span className="font-bold text-slate-800">{modelDiagnostic?.input_shape || '(1, 0, 294)'}</span>
          </div>
          <div className="bg-white p-2 rounded-lg border border-slate-200">
            <span className="text-slate-400 block text-[10px]">Blank Ratio</span>
            <span className="font-bold text-slate-800">
              {((modelDiagnostic?.blank_ratio || 0) * 100).toFixed(1)}%
            </span>
          </div>
          <div className="bg-white p-2 rounded-lg border border-slate-200">
            <span className="text-slate-400 block text-[10px]">Forward / Decode</span>
            <span className="font-bold text-slate-800">
              {modelDiagnostic?.forward_ms || 0} / {modelDiagnostic?.decode_ms || 0} ms
            </span>
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

