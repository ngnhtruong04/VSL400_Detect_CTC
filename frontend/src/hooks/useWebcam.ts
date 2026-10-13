import { useState, useEffect, useRef, useCallback } from 'react';

export function useWebcam() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [isStreaming, setIsStreaming] = useState<boolean>(false);
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState<string>('');
  const [error, setError] = useState<string | null>(null);

  // Enumerate video devices
  useEffect(() => {
    const getDevices = async () => {
      try {
        const devList = await navigator.mediaDevices.enumerateDevices();
        const videoDevs = devList.filter((d) => d.kind === 'videoinput');
        setDevices(videoDevs);
        if (videoDevs.length > 0 && !selectedDeviceId) {
          setSelectedDeviceId(videoDevs[0].deviceId);
        }
      } catch (err: any) {
        console.warn('Unable to enumerate camera devices:', err);
      }
    };

    getDevices();
  }, [selectedDeviceId]);

  const startStream = useCallback(async () => {
    setError(null);
    try {
      if (videoRef.current && videoRef.current.srcObject) {
        const oldStream = videoRef.current.srcObject as MediaStream;
        oldStream.getTracks().forEach((t) => t.stop());
      }

      const constraints: MediaStreamConstraints = {
        video: {
          deviceId: selectedDeviceId ? { exact: selectedDeviceId } : undefined,
          width: { ideal: 1280 },
          height: { ideal: 720 },
          frameRate: { ideal: 30 },
        },
        audio: false,
      };

      const stream = await navigator.mediaDevices.getUserMedia(constraints);
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
  }, [selectedDeviceId]);

  const stopStream = useCallback(() => {
    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach((track) => track.stop());
      videoRef.current.srcObject = null;
    }
    setIsStreaming(false);
  }, []);

  return {
    videoRef,
    isStreaming,
    devices,
    selectedDeviceId,
    setSelectedDeviceId,
    startStream,
    stopStream,
    error,
  };
}

