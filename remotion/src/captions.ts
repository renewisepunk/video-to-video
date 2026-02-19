import { useState, useEffect, useCallback } from "react";
import { staticFile, cancelRender, delayRender, continueRender } from "remotion";
import type { Caption } from "@remotion/captions";

export function useCaptions(filename: string): Caption[] | null {
  const [captions, setCaptions] = useState<Caption[] | null>(null);
  const [handle] = useState(() => delayRender("Loading captions JSON"));

  const fetchCaptions = useCallback(async () => {
    try {
      const response = await fetch(staticFile(filename));
      const data: Caption[] = await response.json();
      setCaptions(data);
      continueRender(handle);
    } catch (e) {
      cancelRender(e);
    }
  }, [filename, handle]);

  useEffect(() => {
    fetchCaptions();
  }, [fetchCaptions]);

  return captions;
}
