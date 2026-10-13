import React from 'react';
import { Target, Layers } from 'lucide-react';
import { RecognitionMode } from '../types';

interface ModeSwitcherProps {
  mode: RecognitionMode;
  onChangeMode: (mode: RecognitionMode) => void;
  disabled?: boolean;
}

export const ModeSwitcher: React.FC<ModeSwitcherProps> = ({
  mode,
  onChangeMode,
  disabled = false,
}) => {
  return (
    <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-2 flex flex-col sm:flex-row gap-2 mb-4">
      {/* Mode A Tab */}
      <button
        type="button"
        disabled={disabled}
        onClick={() => onChangeMode('isolated')}
        className={`flex-1 flex items-center justify-start sm:justify-center space-x-3 p-3 rounded-xl transition-all text-left ${
          mode === 'isolated'
            ? 'bg-gradient-to-r from-sky-500/15 to-blue-500/10 border-2 border-sky-500 text-sky-950 shadow-sm'
            : 'border border-transparent hover:bg-slate-50 text-slate-600 hover:text-slate-900'
        } ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}`}
      >
        <div
          className={`w-9 h-9 rounded-lg flex items-center justify-center shrink-0 ${
            mode === 'isolated'
              ? 'bg-sky-600 text-white shadow-sm'
              : 'bg-slate-100 text-slate-500'
          }`}
        >
          <Target className="w-5 h-5" />
        </div>
        <div>
          <div className="flex items-center gap-2">
            <span className="font-bold text-sm">CHẾ ĐỘ A: KIỂM TRA GLOSS</span>
            <span
              className={`text-[10px] font-bold px-2 py-0.5 rounded-full ${
                mode === 'isolated'
                  ? 'bg-sky-600 text-white'
                  : 'bg-slate-200 text-slate-600'
              }`}
            >
              Đơn Lẻ
            </span>
          </div>
          <p className="text-[11px] text-slate-500 leading-tight mt-0.5">
            Ghi đoạn cử chỉ ngắn 1 từ, xếp hạng Top-5, độ tin cậy tương đối
          </p>
        </div>
      </button>

      {/* Mode B Tab */}
      <button
        type="button"
        disabled={disabled}
        onClick={() => onChangeMode('continuous')}
        className={`flex-1 flex items-center justify-start sm:justify-center space-x-3 p-3 rounded-xl transition-all text-left ${
          mode === 'continuous'
            ? 'bg-gradient-to-r from-indigo-500/15 to-purple-500/10 border-2 border-indigo-500 text-indigo-950 shadow-sm'
            : 'border border-transparent hover:bg-slate-50 text-slate-600 hover:text-slate-900'
        } ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}`}
      >
        <div
          className={`w-9 h-9 rounded-lg flex items-center justify-center shrink-0 ${
            mode === 'continuous'
              ? 'bg-indigo-600 text-white shadow-sm'
              : 'bg-slate-100 text-slate-500'
          }`}
        >
          <Layers className="w-5 h-5" />
        </div>
        <div>
          <div className="flex items-center gap-2">
            <span className="font-bold text-sm">CHẾ ĐỘ B: KIỂM TRA CONTINUOUS</span>
            <span
              className={`text-[10px] font-bold px-2 py-0.5 rounded-full ${
                mode === 'continuous'
                  ? 'bg-indigo-600 text-white'
                  : 'bg-slate-200 text-slate-600'
              }`}
            >
              Chuỗi Ký Hiệu
            </span>
          </div>
          <p className="text-[11px] text-slate-500 leading-tight mt-0.5">
            Nhận diện câu liên tục, FSM tách phát ngôn, dịch tiếng Việt tự nhiên
          </p>
        </div>
      </button>
    </div>
  );
};

