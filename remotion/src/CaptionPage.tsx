import { AbsoluteFill, useCurrentFrame, useVideoConfig, spring } from "remotion";
import type { TikTokPage } from "@remotion/captions";
import { useMemo } from "react";
import { loadFont } from "@remotion/google-fonts/Montserrat";

const { fontFamily } = loadFont("normal", {
  weights: ["800"],
  subsets: ["latin", "latin-ext"],
});

const ACTIVE_COLOR = "#FFD700";
const INACTIVE_COLOR = "#FFFFFF";

// All sizing is relative to a 1080px-wide reference.
// This way a 704px video gets proportionally smaller text,
// and a 1440px video gets proportionally larger.
const REFERENCE_WIDTH = 1080;

export const CaptionPage: React.FC<{ page: TikTokPage }> = ({ page }) => {
  const frame = useCurrentFrame();
  const { fps, width } = useVideoConfig();

  // Scale factor: 1.0 at 1080px wide, ~0.65 at 704px, ~1.33 at 1440px
  const s = width / REFERENCE_WIDTH;

  const sizes = useMemo(
    () => ({
      fontSize: Math.round(42 * s),
      borderRadius: Math.round(14 * s),
      paddingV: Math.round(10 * s),
      paddingH: Math.round(20 * s),
      glowRadius: Math.round(10 * s),
      shadowY: Math.round(2 * s),
      shadowBlur: Math.round(4 * s),
    }),
    [s],
  );

  const currentTimeMs = (frame / fps) * 1000;
  const absoluteTimeMs = page.startMs + currentTimeMs;

  return (
    <AbsoluteFill
      style={{
        justifyContent: "flex-end",
        alignItems: "center",
        paddingBottom: "32%",
      }}
    >
      <div
        style={{
          backgroundColor: "rgba(0, 0, 0, 0.6)",
          borderRadius: sizes.borderRadius,
          padding: `${sizes.paddingV}px ${sizes.paddingH}px`,
          maxWidth: "85%",
          display: "flex",
          flexWrap: "wrap",
          justifyContent: "center",
        }}
      >
        <div
          style={{
            fontSize: sizes.fontSize,
            fontWeight: 800,
            fontFamily,
            whiteSpace: "pre-wrap",
            textAlign: "center",
            lineHeight: 1.3,
          }}
        >
          {page.tokens.map((token, i) => {
            const isActive =
              token.fromMs <= absoluteTimeMs && token.toMs > absoluteTimeMs;

            const tokenStartFrame = Math.round(
              ((token.fromMs - page.startMs) / 1000) * fps,
            );
            const scaleVal = isActive
              ? spring({
                  frame: frame - tokenStartFrame,
                  fps,
                  config: { damping: 15, stiffness: 200 },
                  durationInFrames: 8,
                })
              : 1;

            const activeScale = 1 + 0.08 * scaleVal;

            return (
              <span
                key={`${token.fromMs}-${i}`}
                style={{
                  color: isActive ? ACTIVE_COLOR : INACTIVE_COLOR,
                  display: "inline-block",
                  transform: isActive ? `scale(${activeScale})` : undefined,
                  textShadow: isActive
                    ? `0 0 ${sizes.glowRadius}px rgba(255, 215, 0, 0.4)`
                    : `0 ${sizes.shadowY}px ${sizes.shadowBlur}px rgba(0, 0, 0, 0.8)`,
                  transition: "color 0.05s",
                }}
              >
                {token.text}
              </span>
            );
          })}
        </div>
      </div>
    </AbsoluteFill>
  );
};
