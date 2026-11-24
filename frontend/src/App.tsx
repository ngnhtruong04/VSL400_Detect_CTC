import React, { useState, useEffect, useRef } from 'react';
import { Header } from './components/Header';
import { ModeSwitcher } from './components/ModeSwitcher';
import { CameraPanel } from './components/CameraPanel';
import { IsolatedRecognitionPanel } from './components/IsolatedRecognitionPanel';
import { RecognitionPanel } from './components/RecognitionPanel';
import { SentenceCard } from './components/SentenceCard';
import { ControlBar } from './components/ControlBar';
import { SettingsModal } from './components/SettingsModal';
import { DevMetricsModal } from './components/DevMetricsModal';
import { HelpModal } from './components/HelpModal';
import { useWebcam } from './hooks/useWebcam';
import { useRecognitionSocket } from './hooks/useRecognitionSocket';
import { AppSettings, SystemStatus, RecognitionMode } from './types';

export const App: React.FC = () => {
  // Mode selection state: Mode A ('isolated') or Mode B ('continuous')
  const [mode, setMode] = useState<RecognitionMode>('isolated');

  // Modal states
  const [isDevMode, setIsDevMode] = useState<boolean>(false);
  const [isSettingsOpen, setIsSettingsOpen] = useState<boolean>(false);
  const [isHelpOpen, setIsHelpOpen] = useState<boolean>(false);

  // Settings
  const [settings, setSettings] = useState<AppSettings>({
    mirrorPreview: true,
    showLandmarks: true,
    drawHands: true,
    drawPose: true,
    showFps: true,
    sensitivity: 'medium',
    llmEnabled: false,
    llmProvider: 'none',
  });

  const {
    videoRef,
    isStreaming,
    devices,
    selectedDeviceId,
    setSelectedDeviceId,
    startStream,
    stopStream,
    error: cameraError,
  } = useWebcam();

  const {
    isConnected,
    isolatedState,
    isolatedPrediction,
    startRecord,
    stopRecord,
    fsmState,
    liveGloss,
    stableTokens,
    finalGloss,
    finalSentence,
    utteranceHistory,
    endpointReason,
    commitUtterance,
    rawTokens,
    collapsedTokens,
    landmarks,
    landmarksRef,
    landmarksDiagnostic,
    modelDiagnostic,
    metrics,
    sendFrame,
    sendConfig,
    resetSession,
  } = useRecognitionSocket({
    mode,
    enabled: isStreaming,
  });

  // Sync sensitivity and LLM settings to backend when connected or changed
  useEffect(() => {
    if (isConnected) {
      sendConfig({
        sensitivity: settings.sensitivity,
        llm_enabled: settings.llmEnabled,
        llm_provider: settings.llmProvider,
      });
    }
  }, [isConnected, settings.sensitivity, settings.llmEnabled, settings.llmProvider, sendConfig]);

  const systemStatus: SystemStatus = isConnected
    ? 'ready'
    : isStreaming
    ? 'loading'
    : 'disconnected';

  // Offscreen canvas for frame extraction
  const captureCanvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    captureCanvasRef.current = document.createElement('canvas');
  }, []);

  // Frame streaming loop
  useEffect(() => {
    if (!isStreaming) return;

    let animId: number;
    let lastTime = performance.now();
    const targetInterval = 1000 / 30; // Locked 30 FPS capture target

    const loop = (currentTime: number) => {
      const delta = currentTime - lastTime;

      if (delta >= targetInterval && videoRef.current && videoRef.current.readyState >= 2) {
        lastTime = currentTime - Math.min(delta % targetInterval, targetInterval);

        const video = videoRef.current;
        const canvas = captureCanvasRef.current;

        if (canvas && video.videoWidth > 0 && video.videoHeight > 0) {
          // Cap at 640px width (native 640x480 on 4:3, 640x360 on 16:9) to preserve hand landmark precision
          // and prevent MediaPipe palm re-detection slowdowns caused by aggressive 384px downscaling.
          const targetW = Math.min(640, video.videoWidth);
          const targetH = Math.round((targetW * video.videoHeight) / video.videoWidth);
          if (canvas.width !== targetW || canvas.height !== targetH) {
            canvas.width = targetW;
            canvas.height = targetH;
          }

          const ctx = canvas.getContext('2d', { alpha: false });
          if (ctx) {
            // Section 26 Mirror Rule: always send unmirrored frame for ML inference
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            sendFrame(canvas);
          }
        }
      }

      animId = requestAnimationFrame(loop);
    };

    animId = requestAnimationFrame(loop);

    return () => {
      cancelAnimationFrame(animId);
    };
  }, [isStreaming, sendFrame]);

  const handleStart = () => {
    startStream();
  };

  const handleStop = () => {
    stopStream();
  };

  const handleClear = () => {
    resetSession();
  };

  const handleUpdateSettings = (newSet: Partial<AppSettings>) => {
    setSettings((prev) => ({ ...prev, ...newSet }));
  };

  // Convert isolated state to CameraPanel FSM status for visual consistency
  const cameraFsmState =
    mode === 'isolated'
      ? isolatedState === 'RECORDING'
        ? 'SIGNING'
        : isolatedState === 'INFERRING'
        ? 'POSSIBLE_END'
        : isolatedState === 'RESULT'
        ? 'END_OF_UTTERANCE'
        : 'IDLE'
      : fsmState;

  return (
    <div className="min-h-screen bg-slate-50 flex flex-col">
      {/* 1. Header */}
      <Header
        systemStatus={systemStatus}
        isDevMode={isDevMode}
        onToggleDevMode={() => setIsDevMode(!isDevMode)}
        onOpenSettings={() => setIsSettingsOpen(true)}
        onOpenHelp={() => setIsHelpOpen(true)}
      />

      {/* 2. Main Content Container */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-4 sm:py-6">
        {/* Mode Switcher Tabs */}
        <ModeSwitcher
          mode={mode}
          onChangeMode={(newMode) => {
            if (newMode !== mode) {
              resetSession();
              setMode(newMode);
            }
          }}
        />

        {/* Grid Layout */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 items-start">
          {/* Left Column: Camera Viewport + Controls (7 cols on desktop) */}
          <div className="lg:col-span-7 flex flex-col space-y-4">
            <CameraPanel
              videoRef={videoRef}
              isStreaming={isStreaming}
              mirrorPreview={settings.mirrorPreview}
              showFps={settings.showFps}
              fps={metrics.fps}
              fsmState={cameraFsmState}
              landmarks={landmarks}
              landmarksRef={landmarksRef}
              showLandmarks={settings.showLandmarks}
              drawHands={settings.drawHands}
              drawPose={settings.drawPose}
              onToggleLandmarks={() => setSettings((prev) => ({ ...prev, showLandmarks: !prev.showLandmarks }))}
              error={cameraError}
              onStartCamera={handleStart}
            />

            <ControlBar
              isStreaming={isStreaming}
              onStart={handleStart}
              onStop={handleStop}
              onClear={handleClear}
              onOpenSettings={() => setIsSettingsOpen(true)}
            />
          </div>

          {/* Right Column: Mode-Specific Recognition Panel (5 cols on desktop) */}
          <div className="lg:col-span-5 flex flex-col space-y-4">
            {mode === 'isolated' ? (
              <IsolatedRecognitionPanel
                state={isolatedState}
                prediction={isolatedPrediction}
                modelDiagnostic={modelDiagnostic}
                landmarksDiagnostic={landmarksDiagnostic}
                onStartRecord={startRecord}
                onStopRecord={stopRecord}
                onReset={handleClear}
                isStreaming={isStreaming}
              />
            ) : (
              <>
                <RecognitionPanel
                  liveGloss={liveGloss}
                  stableTokens={stableTokens}
                  finalGloss={finalGloss}
                  fsmState={fsmState}
                  endpointReason={endpointReason}
                  modelDiagnostic={modelDiagnostic}
                  landmarksDiagnostic={landmarksDiagnostic}
                  isStreaming={isStreaming}
                  onCommitUtterance={commitUtterance}
                />

                <SentenceCard
                  sentence={finalSentence}
                  liveGloss={liveGloss}
                  history={utteranceHistory}
                />
              </>
            )}
          </div>
        </div>
      </main>

      {/* Modals */}
      <SettingsModal
        isOpen={isSettingsOpen}
        onClose={() => setIsSettingsOpen(false)}
        settings={settings}
        onUpdateSettings={handleUpdateSettings}
        devices={devices}
        selectedDeviceId={selectedDeviceId}
        onSelectDevice={setSelectedDeviceId}
      />

      <DevMetricsModal
        isOpen={isDevMode}
        onClose={() => setIsDevMode(false)}
        mode={mode}
        metrics={metrics}
        fsmState={cameraFsmState}
        rawTokens={rawTokens}
        collapsedTokens={collapsedTokens}
        liveGloss={liveGloss}
        stableTokens={stableTokens}
        endpointReason={endpointReason}
        modelDiagnostic={modelDiagnostic}
        landmarksDiagnostic={landmarksDiagnostic}
        isolatedPrediction={isolatedPrediction}
      />

      <HelpModal
        isOpen={isHelpOpen}
        onClose={() => setIsHelpOpen(false)}
      />
    </div>
  );
};

export default App;
