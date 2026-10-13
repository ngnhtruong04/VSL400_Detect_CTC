import React, { useState } from 'react';
import { MessageSquare, Copy, Check, Volume2 } from 'lucide-react';

interface SentenceCardProps {
  sentence: string;
}

export const SentenceCard: React.FC<SentenceCardProps> = ({ sentence }) => {
  const [copied, setCopied] = useState<boolean>(false);
  const [isSpeaking, setIsSpeaking] = useState<boolean>(false);

  const handleCopy = async () => {
    if (!sentence) return;
    try {
      await navigator.clipboard.writeText(sentence);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.warn('Failed to copy text:', err);
    }
  };

  const handleSpeak = () => {
    if (!sentence || !('speechSynthesis' in window)) return;

    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(sentence);
    utterance.lang = 'vi-VN';
    utterance.rate = 0.95;

    utterance.onstart = () => setIsSpeaking(true);
    utterance.onend = () => setIsSpeaking(false);
    utterance.onerror = () => setIsSpeaking(false);

    window.speechSynthesis.speak(utterance);
  };

  return (
    <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-4 sm:p-5 flex flex-col justify-between space-y-3">
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <MessageSquare className="w-4 h-4 text-emerald-600" />
          <h3 className="text-xs font-bold uppercase tracking-wider text-slate-500">
            Tiếng Việt Tự Nhiên
          </h3>
        </div>

        {sentence && (
          <div className="flex items-center space-x-1.5">
            {/* TTS Button */}
            <button
              onClick={handleSpeak}
              disabled={isSpeaking}
              title="Phát âm câu"
              className={`p-1.5 rounded-lg border text-slate-600 transition-colors ${
                isSpeaking
                  ? 'bg-emerald-50 text-emerald-600 border-emerald-300'
                  : 'border-slate-200 hover:bg-slate-100'
              }`}
            >
              <Volume2 className="w-4 h-4" />
            </button>

            {/* Copy Button */}
            <button
              onClick={handleCopy}
              title="Sao chép câu"
              className="p-1.5 rounded-lg border border-slate-200 hover:bg-slate-100 text-slate-600 transition-colors"
            >
              {copied ? (
                <Check className="w-4 h-4 text-emerald-600" />
              ) : (
                <Copy className="w-4 h-4" />
              )}
            </button>
          </div>
        )}
      </div>

      {/* Sentence Display Area */}
      <div className="min-h-[72px] p-3.5 rounded-xl bg-gradient-to-br from-emerald-50/50 to-teal-50/30 border border-emerald-100 flex items-center">
        {sentence ? (
          <p className="text-base sm:text-lg font-semibold text-slate-800 leading-relaxed">
            "{sentence}"
          </p>
        ) : (
          <span className="text-xs text-slate-400 italic">
            Kết quả câu dịch tự nhiên sẽ xuất hiện tại đây sau khi hoàn tất chuỗi ký hiệu...
          </span>
        )}
      </div>
    </div>
  );
};

