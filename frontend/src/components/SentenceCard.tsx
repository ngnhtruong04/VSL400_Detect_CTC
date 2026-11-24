import React, { useState } from 'react';
import { MessageSquare, Copy, Check, Volume2, History, AlertTriangle } from 'lucide-react';
import { UtteranceHistoryItem } from '../types';

interface SentenceCardProps {
  sentence: string;
  liveGloss?: string[];
  history?: UtteranceHistoryItem[];
  translationStatus?: string;
  translationError?: string | null;
}

export const SentenceCard: React.FC<SentenceCardProps> = ({
  sentence,
  liveGloss = [],
  history = [],
  translationStatus = 'rule_based',
  translationError = null,
}) => {
  const [copied, setCopied] = useState<boolean>(false);
  const [isSpeaking, setIsSpeaking] = useState<boolean>(false);

  // Authoritative unified natural Vietnamese sentence for the entire accumulated Final Gloss sequence
  const displaySentence = sentence || (liveGloss.length > 0 ? liveGloss.join(' ') + '...' : '');
  const fullTranscript = sentence;

  const handleCopy = async () => {
    const textToCopy = fullTranscript || sentence;
    if (!textToCopy) return;
    try {
      await navigator.clipboard.writeText(textToCopy);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.warn('Failed to copy text:', err);
    }
  };

  const handleSpeak = () => {
    const textToSpeak = fullTranscript || sentence;
    if (!textToSpeak || !('speechSynthesis' in window)) return;

    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(textToSpeak);
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
          {sentence && (
            <span
              className={`text-[10px] font-semibold px-2 py-0.5 rounded-full border ${
                translationStatus === 'gemini'
                  ? 'bg-sky-50 text-sky-700 border-sky-200'
                  : translationStatus === 'ollama'
                  ? 'bg-emerald-50 text-emerald-700 border-emerald-200'
                  : translationStatus === 'fallback_error'
                  ? 'bg-amber-50 text-amber-700 border-amber-200'
                  : 'bg-slate-100 text-slate-600 border-slate-200'
              }`}
            >
              {translationStatus === 'gemini'
                ? 'Gemini AI'
                : translationStatus === 'ollama'
                ? 'Ollama cục bộ'
                : translationStatus === 'fallback_error'
                ? 'Dự phòng (lỗi API)'
                : 'Quy tắc ngữ pháp'}
            </span>
          )}
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
              title="Sao chép toàn bộ câu"
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

      {/* Fallback / LLM Connection Warning Banner */}
      {translationStatus === 'fallback_error' && translationError && (
        <div className="flex items-start gap-2 px-3 py-2 rounded-xl bg-amber-50 border border-amber-200 text-amber-800 text-xs">
          <AlertTriangle className="w-4 h-4 text-amber-600 shrink-0 mt-0.5" />
          <span>
            <b>Cảnh báo dịch thuật:</b> {translationError}. Hệ thống đang giữ nguyên chuỗi Gloss cục bộ.
          </span>
        </div>
      )}

      {/* Sentence Display Area */}
      <div className="min-h-[72px] p-3.5 rounded-xl bg-gradient-to-br from-emerald-50/50 to-teal-50/30 border border-emerald-100 flex flex-col justify-center space-y-1">
        {sentence ? (
          <p className="text-base sm:text-lg font-semibold text-slate-800 leading-relaxed">
            "{fullTranscript}"
          </p>
        ) : liveGloss.length > 0 ? (
          <p className="text-sm sm:text-base font-medium text-sky-700 italic animate-pulse">
            Đang nhận diện: {displaySentence}
          </p>
        ) : (
          <span className="text-xs text-slate-400 italic">
            Kết quả câu dịch tự nhiên sẽ xuất hiện tại đây sau khi hoàn tất chuỗi ký hiệu...
          </span>
        )}
      </div>

      {/* Utterance History List (when multiple continuous segments have been committed) */}
      {history.length > 1 && (
        <div className="pt-2 border-t border-slate-100 space-y-1.5">
          <div className="flex items-center gap-1.5 text-[11px] font-bold uppercase tracking-wider text-slate-400">
            <History className="w-3.5 h-3.5" />
            <span>Lịch sử phát ngôn ({history.length})</span>
          </div>
          <div className="max-h-28 overflow-y-auto space-y-1 pr-1">
            {history.map((item, idx) => (
              <div
                key={item.id}
                className="text-xs px-2.5 py-1.5 rounded-lg bg-slate-50 border border-slate-200/70 flex items-center justify-between gap-2"
              >
                <span className="text-slate-700 font-medium truncate">
                  <span className="text-slate-400 font-mono mr-1.5">#{history.length - idx}</span>
                  {item.sentence}
                </span>
                <span className="text-[10px] font-mono text-indigo-500 shrink-0">
                  [{item.gloss.join(' → ')}]
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};
