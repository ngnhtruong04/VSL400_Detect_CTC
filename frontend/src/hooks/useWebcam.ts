import { useState, useEffect, useRef, useCallback } from 'react';

export function useWebcam() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [isStreaming, setIsStreaming] = useState<boolean>(false);
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState<string>('');
  const [error, setError] = useState<string | null>(null);

  const refreshDevices = useCallback(async () => {
    try {
      const devList = await navigator.mediaDevices.enumerateDevices();
      const videoDevs = devList.filter((d) => d.kind === 'videoinput');
      setDevices(videoDevs);
      setSelectedDeviceId((prev) => prev || (videoDevs.length > 0 ? videoDevs[0].deviceId : ''));
    } catch (err: any) {
      console.warn('Unable to enumerate camera devices:', err);
    }
  }, []);

  // Enumerate video devices once on mount
  useEffect(() => {
    refreshDevices();
  }, [refreshDevices]);

  const startStream = useCallback(async (overrideDeviceId?: string) => {
    setError(null);
    try {
      if (videoRef.current && videoRef.current.srcObject) {
        const oldStream = videoRef.current.srcObject as MediaStream;
        oldStream.getTracks().forEach((t) => t.stop());
      }

      const targetDeviceId = overrideDeviceId !== undefined ? overrideDeviceId : selectedDeviceId;

      // 640x480 @ 30fps keeps laptop webcams locked at true 30 FPS (720p often drops to 15 FPS in indoor light)
      const constraints: MediaStreamConstraints = {
        video: {
          deviceId: targetDeviceId ? { exact: targetDeviceId } : undefined,
          width: { ideal: 640 },
          height: { ideal: 480 },
          frameRate: { ideal: 30, min: 24 },
        },
        audio: false,
      };

      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      // Re-enumerate after permission granted so camera labels populate
      refreshDevices();

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.onloadedmetadata = () => {
          videoRef.current?.play();
          setIsStreaming(true);
        };
      }
    } catch (err: any) {
      console.error('Camera access failed:', err);
      if (err.name === 'NotAllowedError') {
        setError('Vui lòng cấp quyền truy cập máy ảnh để nhận diện ký hiệu.');
      } else if (err.name === 'NotFoundError') {
        setError('Không tìm thấy camera trên thiết bị.');
      } else {
        setError(`Lỗi mở camera: ${err.message || err.name}`);
      }
      setIsStreaming(false);
    }
  }, [selectedDeviceId, refreshDevices]);

  const stopStream = useCallback(() => {
    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach((track) => track.stop());
      videoRef.current.srcObject = null;
    }
    setIsStreaming(false);
  }, []);

  const handleSelectDevice = useCallback(
    (deviceId: string) => {
      setSelectedDeviceId(deviceId);
      if (isStreaming) {
        startStream(deviceId);
      }
    },
    [isStreaming, startStream]
  );

  return {
    videoRef,
    isStreaming,
    devices,
    selectedDeviceId,
    setSelectedDeviceId: handleSelectDevice,
    startStream,
    stopStream,
    error,
  };
}
