import type { GuidanceState } from "./types";

export type GuidanceConfig = {
  maxMotionDelta: number;
  stabilityWindowMs: number;
  updateIntervalMs: number;
  staleAfterMs?: number;
  gravityFilterAlpha?: number;
};

export type GravityVector = readonly [number, number, number];

export type GuidanceAccumulator = {
  previousGravity: GravityVector | null;
  stableSince: number | null;
};

export const EMPTY_GUIDANCE_ACCUMULATOR: GuidanceAccumulator = {
  previousGravity: null,
  stableSince: null,
};

export function sensorStaleAfterMs(config: GuidanceConfig): number {
  return config.staleAfterMs ?? Math.max(500, config.updateIntervalMs * 3);
}

export function selectGuidanceSensorSource(
  deviceMotionAvailable: boolean,
  accelerometerAvailable: boolean,
): "device-motion" | "accelerometer" | null {
  if (deviceMotionAvailable) return "device-motion";
  if (accelerometerAvailable) return "accelerometer";
  return null;
}

export function lowPassGravity(
  previous: GravityVector | null,
  current: GravityVector,
  alpha: number,
): GravityVector {
  const weight = Math.min(1, Math.max(0.01, alpha));
  if (previous === null) return current;
  return [
    previous[0] + weight * (current[0] - previous[0]),
    previous[1] + weight * (current[1] - previous[1]),
    previous[2] + weight * (current[2] - previous[2]),
  ];
}

export function evaluateGuidanceReading(
  gravity: GravityVector,
  now: number,
  config: GuidanceConfig,
  accumulator: GuidanceAccumulator,
): { guidance: GuidanceState; accumulator: GuidanceAccumulator } {
  const previous = accumulator.previousGravity;
  const motionDelta = previous
    ? Math.hypot(
        gravity[0] - previous[0],
        gravity[1] - previous[1],
        gravity[2] - previous[2],
      )
    : Number.POSITIVE_INFINITY;
  const stable = motionDelta <= config.maxMotionDelta;
  const stableSince = stable
    ? accumulator.stableSince ?? now
    : null;
  const ready = stableSince !== null && now - stableSince >= config.stabilityWindowMs;

  return {
    guidance: {
      available: true,
      sampledAt: now,
      // Kept in the metadata shape for backward compatibility. Orientation is
      // deliberately not evaluated: a shelf can be photographed in portrait
      // or landscape as long as the camera is steady and faces the products.
      tiltDegrees: null,
      motionDelta,
      upright: true,
      stable,
      ready,
    },
    accumulator: { previousGravity: gravity, stableSince },
  };
}

export function describeGuidanceBlock(guidance: GuidanceState): {
  status: GuidanceState["status"];
  blockReason: string | null;
  reason?: string;
} {
  if (!guidance.stable) {
    return { status: "moving", blockReason: "motion_exceeds_limit", reason: "Di chuyển chậm lại" };
  }
  if (!guidance.ready) {
    return { status: "stabilizing", blockReason: "stability_window_incomplete", reason: "Giữ yên thêm một chút…" };
  }
  return { status: "ready", blockReason: null };
}

export function isCaptureGuidanceReady(
  guidance: GuidanceState,
  now: number,
  staleAfter: number,
): boolean {
  return guidance.available
    && guidance.ready
    && guidance.stable
    && guidance.sampledAt !== null
    && now - guidance.sampledAt <= staleAfter;
}
