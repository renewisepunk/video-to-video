import { Composition, registerRoot } from "remotion";
import { CaptionOverlay } from "./CaptionOverlay";

const RemotionRoot: React.FC = () => {
  return (
    <Composition
      id="CaptionOverlay"
      component={CaptionOverlay}
      durationInFrames={30 * 60 * 10}
      fps={30}
      width={1080}
      height={1920}
    />
  );
};

registerRoot(RemotionRoot);
