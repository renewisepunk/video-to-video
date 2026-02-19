import { AbsoluteFill, Sequence, useVideoConfig } from "remotion";
import { Video } from "@remotion/media";
import { createTikTokStyleCaptions } from "@remotion/captions";
import { useMemo } from "react";
import { staticFile } from "remotion";
import { useCaptions } from "./captions";
import { CaptionPage } from "./CaptionPage";

const SWITCH_CAPTIONS_EVERY_MS = 1200;

export const CaptionOverlay: React.FC = () => {
  const { fps } = useVideoConfig();
  const captions = useCaptions("captions.json");

  const pages = useMemo(() => {
    if (!captions) return [];
    const { pages } = createTikTokStyleCaptions({
      captions,
      combineTokensWithinMilliseconds: SWITCH_CAPTIONS_EVERY_MS,
    });
    return pages;
  }, [captions]);

  if (!captions) {
    return null;
  }

  return (
    <AbsoluteFill>
      <Video src={staticFile("video.mp4")} />
      <AbsoluteFill>
        {pages.map((page, index) => {
          const nextPage = pages[index + 1] ?? null;
          const startFrame = Math.round((page.startMs / 1000) * fps);
          const endFrame = Math.min(
            nextPage ? Math.round((nextPage.startMs / 1000) * fps) : Infinity,
            startFrame + Math.round((SWITCH_CAPTIONS_EVERY_MS / 1000) * fps),
          );
          const durationInFrames = endFrame - startFrame;

          if (durationInFrames <= 0) {
            return null;
          }

          return (
            <Sequence
              key={index}
              from={startFrame}
              durationInFrames={durationInFrames}
            >
              <CaptionPage page={page} />
            </Sequence>
          );
        })}
      </AbsoluteFill>
    </AbsoluteFill>
  );
};
