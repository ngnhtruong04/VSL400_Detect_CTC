import React from 'react';
import { Activity, Settings, HelpCircle, Code2, CheckCircle2, AlertCircle } from 'lucide-react';
import { SystemStatus } from '../types';

interface HeaderProps {
  systemStatus: SystemStatus;
  isDevMode: boolean;
  onToggleDevMode: () => void;
  onOpenSettings: () => void;
  onOpenHelp: () => void;
}

export const Header: React.FC<HeaderProps> = ({
  systemStatus,
  isDevMode,
  onToggleDevMode,
  onOpenSettings,
  onOpenHelp,
}) => {
  return (
    <header className="bg-white border-b border-slate-200 sticky top-0 z-30 shadow-sm">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        {/* Title & Brand */}
        <div className="flex items-center space-x-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-sky-600 to-cyan-500 flex items-center justify-center text-white shadow-md shadow-sky-500/20">
            <Activity className="w-6 h-6" />
          </div>
          <div>
            <h1 className="text-lg sm:text-xl font-bold text-slate-900 tracking-tight flex items-center gap-2">
              VSL Translator
              <span className="text-xs font-semibold px-2 py-0.5 rounded-full bg-sky-100 text-sky-700 border border-sky-200">
                TCN-Transformer
              </span>
            </h1>
            <p className="text-xs text-slate-500 hidden sm:block">
              Hệ Thống Nhận Diện Ngôn Ngữ Ký Hiệu Việt Nam Realtime
            </p>
          </div>
        </div>

        {/* Status Badge & Actions */}
        <div className="flex items-center space-x-2 sm:space-x-3">
          {/* Status Badge */}
          <div
            className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-full text-xs font-semibold border transition-all ${
              systemStatus === 'ready'
                ? 'bg-emerald-50 text-emerald-700 border-emerald-200'
                : systemStatus === 'loading'
                ? 'bg-amber-50 text-amber-700 border-amber-200 animate-pulse'
                : 'bg-rose-50 text-rose-700 border-rose-200'
            }`}
          >
            {systemStatus === 'ready' ? (
              <CheckCircle2 className="w-4 h-4 text-emerald-600" />
            ) : (
              <AlertCircle className="w-4 h-4" />
            )}
            <span>
              {systemStatus === 'ready'
                ? 'Model Sẵn sàng'
                : systemStatus === 'loading'
                ? 'Đang kết nối...'
                : 'Mất kết nối'}
            </span>
          </div>

          {/* Dev Mode Button */}
          <button
            onClick={onToggleDevMode}
            title="Chế độ nhà phát triển"
            className={`p-2 rounded-xl border transition-all ${
              isDevMode
                ? 'bg-sky-50 text-sky-600 border-sky-300 shadow-sm'
                : 'text-slate-600 border-slate-200 hover:bg-slate-100'
            }`}
          >
            <Code2 className="w-5 h-5" />
          </button>

          {/* Settings Button */}
          <button
            onClick={onOpenSettings}
            title="Cài đặt hệ thống"
            className="p-2 rounded-xl text-slate-600 border border-slate-200 hover:bg-slate-100 transition-colors"
          >
            <Settings className="w-5 h-5" />
          </button>

          {/* Help Button */}
          <button
            onClick={onOpenHelp}
            title="Hướng dẫn sử dụng"
            className="p-2 rounded-xl text-slate-600 border border-slate-200 hover:bg-slate-100 transition-colors"
          >
            <HelpCircle className="w-5 h-5" />
          </button>
        </div>
      </div>
    </header>
  );
};

