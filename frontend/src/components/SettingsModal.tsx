import React from 'react';
import { X, Sliders, Camera, Cpu, Sparkles } from 'lucide-react';
import { AppSettings } from '../types';

interface SettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
  settings: AppSettings;
  onUpdateSettings: (newSettings: Partial<AppSettings>) => void;
  devices: MediaDeviceInfo[];
  selectedDeviceId: string;
  onSelectDevice: (id: string) => void;
}

export const SettingsModal: React.FC<SettingsModalProps> = ({
  isOpen,
  onClose,
  settings,
  onUpdateSettings,
  devices,
  selectedDeviceId,
  onSelectDevice,
}) => {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/60 backdrop-blur-sm animate-fade-in">
      <div className="bg-white rounded-3xl shadow-xl border border-slate-200 w-full max-w-lg overflow-hidden flex flex-col max-h-[90vh]">
        {/* Modal Header */}
        <div className="px-6 py-4 border-b border-slate-100 flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <Sliders className="w-5 h-5 text-sky-600" />
            <h3 className="text-base font-bold text-slate-800">Cài đặt Hệ thống</h3>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-xl text-slate-400 hover:text-slate-600 hover:bg-slate-100 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Modal Body */}
        <div className="p-6 overflow-y-auto space-y-6">
          {/* 1. Camera Settings */}
          <div className="space-y-3">
            <h4 className="text-xs font-bold uppercase tracking-wider text-slate-500 flex items-center gap-2">
              <Camera className="w-4 h-4 text-slate-400" />
              Thiết bị Camera
            </h4>

            <div>
              <label className="text-xs font-medium text-slate-600 mb-1 block">Chọn Camera</label>
              <select
                value={selectedDeviceId}
                onChange={(e) => onSelectDevice(e.target.value)}
                className="w-full text-xs font-medium bg-slate-50 border border-slate-200 rounded-xl px-3 py-2.5 text-slate-700 focus:outline-none focus:ring-2 focus:ring-sky-500/20 focus:border-sky-500"
              >
                {devices.length > 0 ? (
                  devices.map((d, idx) => (
                    <option key={d.deviceId || idx} value={d.deviceId}>
                      {d.label || `Camera ${idx + 1}`}
                    </option>
                  ))
                ) : (
                  <option value="">Không tìm thấy camera</option>
                )}
              </select>
            </div>

            <div className="flex items-center justify-between pt-1">
              <div>
                <span className="text-xs font-semibold text-slate-700 block">Lật ngược hình ảnh (Mirror)</span>
                <span className="text-[11px] text-slate-400">Hiển thị như gương soi để dễ thao tác ký hiệu</span>
              </div>
              <input
                type="checkbox"
                checked={settings.mirrorPreview}
                onChange={(e) => onUpdateSettings({ mirrorPreview: e.target.checked })}
                className="w-4 h-4 rounded text-sky-600 focus:ring-sky-500 border-slate-300"
              />
            </div>

            <div className="flex items-center justify-between pt-1">
              <div>
                <span className="text-xs font-semibold text-slate-700 block">Hiển thị chỉ số FPS</span>
                <span className="text-[11px] text-slate-400">Hiển thị tốc độ khung hình góc camera</span>
              </div>
              <input
                type="checkbox"
                checked={settings.showFps}
                onChange={(e) => onUpdateSettings({ showFps: e.target.checked })}
                className="w-4 h-4 rounded text-sky-600 focus:ring-sky-500 border-slate-300"
              />
            </div>

            <div className="flex items-center justify-between pt-1">
              <div>
                <span className="text-xs font-semibold text-slate-700 block">Khung xương khớp tay & vai (Landmarks)</span>
                <span className="text-[11px] text-slate-400">Vẽ trực tiếp bộ xương keypoint lên người ký</span>
              </div>
              <input
                type="checkbox"
                checked={settings.showLandmarks}
                onChange={(e) => onUpdateSettings({ showLandmarks: e.target.checked })}
                className="w-4 h-4 rounded text-sky-600 focus:ring-sky-500 border-slate-300"
              />
            </div>
          </div>

          <hr className="border-slate-100" />

          {/* 2. Endpoint Recognition Sensitivity */}
          <div className="space-y-3">
            <h4 className="text-xs font-bold uppercase tracking-wider text-slate-500 flex items-center gap-2">
              <Cpu className="w-4 h-4 text-slate-400" />
              Độ nhạy kết thúc câu (Endpoint FSM)
            </h4>

            <div className="grid grid-cols-3 gap-2">
              {(['low', 'medium', 'high'] as const).map((lvl) => (
                <button
                  key={lvl}
                  onClick={() => onUpdateSettings({ sensitivity: lvl })}
                  className={`py-2 px-3 rounded-xl text-xs font-bold border transition-all ${
                    settings.sensitivity === lvl
                      ? 'bg-sky-50 text-sky-700 border-sky-300 shadow-sm'
                      : 'bg-slate-50 text-slate-600 border-slate-200 hover:bg-slate-100'
                  }`}
                >
                  {lvl === 'low' ? 'Chậm (Kỹ)' : lvl === 'medium' ? 'Vừa (Chuẩn)' : 'Nhanh'}
                </button>
              ))}
            </div>
          </div>

          <hr className="border-slate-100" />

          {/* 3. Language Layer / LLM */}
          <div className="space-y-3">
            <h4 className="text-xs font-bold uppercase tracking-wider text-slate-500 flex items-center gap-2">
              <Sparkles className="w-4 h-4 text-slate-400" />
              Dịch thuật Tiếng Việt Tự Nhiên (LLM)
            </h4>

            <div className="flex items-center justify-between">
              <div>
                <span className="text-xs font-semibold text-slate-700 block">Kích hoạt LLM Post-Processing</span>
                <span className="text-[11px] text-slate-400">Chuyển chuỗi Gloss thành câu văn tự nhiên hoàn chỉnh</span>
              </div>
              <input
                type="checkbox"
                checked={settings.llmEnabled}
                onChange={(e) => onUpdateSettings({ llmEnabled: e.target.checked })}
                className="w-4 h-4 rounded text-sky-600 focus:ring-sky-500 border-slate-300"
              />
            </div>

            {settings.llmEnabled && (
              <div>
                <label className="text-xs font-medium text-slate-600 mb-1 block">Nhà cung cấp LLM</label>
                <select
                  value={settings.llmProvider}
                  onChange={(e) => onUpdateSettings({ llmProvider: e.target.value as any })}
                  className="w-full text-xs font-medium bg-slate-50 border border-slate-200 rounded-xl px-3 py-2.5 text-slate-700 focus:outline-none focus:ring-2 focus:ring-sky-500/20 focus:border-sky-500"
                >
                  <option value="none">Quy tắc chuẩn (Rule-based Smoothing)</option>
                  <option value="gemini">Google Gemini API</option>
                  <option value="ollama">Local Ollama</option>
                </select>
              </div>
            )}
          </div>
        </div>

        {/* Modal Footer */}
        <div className="px-6 py-4 border-t border-slate-100 bg-slate-50/50 flex justify-end">
          <button
            onClick={onClose}
            className="px-5 py-2 text-xs font-bold rounded-xl bg-slate-900 text-white hover:bg-slate-800 transition-colors shadow-sm"
          >
            Đóng
          </button>
        </div>
      </div>
    </div>
  );
};

