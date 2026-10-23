import {interpolate as remotionInterpolate} from 'remotion';

/**
 * Remotion's interpolate, tolerant of repeated keyframe times. A cut that skips beats gives them
 * zero duration, so b.at(i) === b.at(i + 1) and the input range is no longer strictly increasing
 * (Remotion throws). Equal times are nudged apart by a tiny epsilon: the value jumps at that
 * instant, which is what a skipped beat means. Strictly increasing inputs are passed through as is.
 */
type Options = {
  easing?: (t: number) => number;
  extrapolateLeft?: 'clamp' | 'extend' | 'identity' | 'wrap';
  extrapolateRight?: 'clamp' | 'extend' | 'identity' | 'wrap';
};

export const interpolate = (input: number, inputRange: readonly number[], outputRange: readonly number[], options?: Options): number => {
  let range = inputRange;
  for (let i = 1; i < inputRange.length; i++) {
    if (inputRange[i] <= inputRange[i - 1]) {
      const fixed = [...inputRange];
      for (let k = 1; k < fixed.length; k++) if (fixed[k] <= fixed[k - 1]) fixed[k] = fixed[k - 1] + 1e-6;
      range = fixed;
      break;
    }
  }
  return remotionInterpolate(input, range, outputRange, options);
};
