import React from 'react';
import { Play, Square, RotateCcw, Sliders } from 'lucide-react';

interface ControlBarProps {
  isStreaming: boolean;
  onStart: () => void;
  onStop: () => void;
  onClear: () => void;
  onOpenSettings: () => void;
}

export const ControlBar: React.FC<ControlBarProps> = ({
  isStreaming,
  onStart,
  onStop,
  onClear,
  onOpenSettings,
}) => {
  return (
    <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-3 sm:p-4 flex flex-wrap items-center justify-between gap-3">
      {/* Primary Action Button */}
      <div className="flex items-center space-x-2">
        {!isStreaming ? (
          <button
            onClick={onStart}
            className="flex items-center space-x-2 px-5 py-2.5 rounded-xl bg-gradient-to-r from-sky-600 to-cyan-600 hover:from-sky-500 hover:to-cyan-500 text-white font-bold text-sm shadow-md shadow-sky-600/25 active:scale-95 transition-all"
          >
            <Play className="w-4 h-4 fill-white" />
            <span>Bắt đầu nhận diện</span>
          </button>
        ) : (
          <button
            onClick={onStop}
            className="flex items-center space-x-2 px-5 py-2.5 rounded-xl bg-rose-600 hover:bg-rose-500 text-white font-bold text-sm shadow-md shadow-rose-600/25 active:scale-95 transition-all"
          >
            <Square className="w-4 h-4 fill-white" />
            <span>Dừng nhận diện</span>
          </button>
        )}

        {/* Clear / Reset Button */}
        <button
          onClick={onClear}
          title="Xóa kết quả và làm mới chuỗi"
          className="flex items-center space-x-1.5 px-4 py-2.5 rounded-xl border border-slate-200 hover:bg-slate-100 text-slate-700 font-semibold text-sm transition-colors active:scale-95"
        >
          <RotateCcw className="w-4 h-4 text-slate-500" />
          <span>Xóa kết quả</span>
        </button>
      </div>

      {/* Settings Button */}
      <button
        onClick={onOpenSettings}
        className="flex items-center space-x-1.5 px-4 py-2.5 rounded-xl border border-slate-200 hover:bg-slate-100 text-slate-700 font-semibold text-sm transition-colors"
      >
        <Sliders className="w-4 h-4 text-slate-500" />
        <span>Cài đặt</span>
      </button>
    </div>
  );
};

