// frontend/src/hooks/useUiSounds.jsx
import { useEffect, useRef } from "react";

export function useUiSounds() {
  const hover = useRef(null);
  const start = useRef(null);
  const done = useRef(null);

  useEffect(() => {
    hover.current = new Audio("/sounds/hover.mp3");
    start.current = new Audio("/sounds/start.mp3");
    done.current = new Audio("/sounds/done.mp3");
  }, []);

  return {
    playHover: () => hover.current?.play().catch(() => {}),
    playStart: () => start.current?.play().catch(() => {}),
    playDone: () => done.current?.play().catch(() => {}),
  };
}
