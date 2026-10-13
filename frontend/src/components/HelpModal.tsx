import React from 'react';
import { X, HelpCircle, CheckCircle2 } from 'lucide-react';

interface HelpModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export const HelpModal: React.FC<HelpModalProps> = ({ isOpen, onClose }) => {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/60 backdrop-blur-sm animate-fade-in">
      <div className="bg-white rounded-3xl shadow-xl border border-slate-200 w-full max-w-md overflow-hidden flex flex-col max-h-[90vh]">
        {/* Header */}
        <div className="px-6 py-4 border-b border-slate-100 flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <HelpCircle className="w-5 h-5 text-sky-600" />
            <h3 className="text-base font-bold text-slate-800">Hướng dẫn Sử dụng</h3>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-xl text-slate-400 hover:text-slate-600 hover:bg-slate-100 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content */}
        <div className="p-6 space-y-4">
          <div className="flex items-start space-x-3">
            <div className="w-7 h-7 rounded-full bg-sky-100 text-sky-700 font-bold text-xs flex items-center justify-center shrink-0 mt-0.5">
              1
            </div>
            <div>
              <h4 className="text-sm font-bold text-slate-800">Đứng trước máy ảnh</h4>
              <p className="text-xs text-slate-500 mt-0.5">
                Ngồi hoặc đứng thẳng đối diện với webcam trong môi trường đủ ánh sáng.
              </p>
            </div>
          </div>

          <div className="flex items-start space-x-3">
            <div className="w-7 h-7 rounded-full bg-sky-100 text-sky-700 font-bold text-xs flex items-center justify-center shrink-0 mt-0.5">
              2
            </div>
            <div>
              <h4 className="text-sm font-bold text-slate-800">Giữ hai tay và phần thân trên</h4>
              <p className="text-xs text-slate-500 mt-0.5">
                Đảm bảo hai vai và hai bàn tay luôn nằm trọn vẹn bên trong khung hình camera.
              </p>
            </div>
          </div>

          <div className="flex items-start space-x-3">
            <div className="w-7 h-7 rounded-full bg-sky-100 text-sky-700 font-bold text-xs flex items-center justify-center shrink-0 mt-0.5">
              3
            </div>
            <div>
              <h4 className="text-sm font-bold text-slate-800">Bắt đầu ký hiệu</h4>
              <p className="text-xs text-slate-500 mt-0.5">
                Nhấn nút "Bắt đầu nhận diện" và thực hiện các động tác ngôn ngữ ký hiệu một cách tự nhiên.
              </p>
            </div>
          </div>

          <div className="flex items-start space-x-3">
            <div className="w-7 h-7 rounded-full bg-emerald-100 text-emerald-700 font-bold text-xs flex items-center justify-center shrink-0 mt-0.5">
              ✓
            </div>
            <div>
              <h4 className="text-sm font-bold text-slate-800">Kết thúc câu</h4>
              <p className="text-xs text-slate-500 mt-0.5">
                Khi thực hiện xong câu, hạ hai tay hoặc giữ yên tay trong 1 giây để hệ thống tự động chốt kết quả và tạo câu tiếng Việt hoàn chỉnh.
              </p>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="px-6 py-4 border-t border-slate-100 bg-slate-50/50 flex justify-end">
          <button
            onClick={onClose}
            className="px-5 py-2 text-xs font-bold rounded-xl bg-sky-600 text-white hover:bg-sky-500 transition-colors shadow-sm"
          >
            Đã hiểu
          </button>
        </div>
      </div>
    </div>
  );
};

