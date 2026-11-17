import assert from "node:assert/strict";
import test from "node:test";

import {
  EMPTY_GUIDANCE_ACCUMULATOR,
  evaluateGuidanceReading,
  isCaptureGuidanceReady,
  lowPassGravity,
  selectGuidanceSensorSource,
  sensorStaleAfterMs,
} from "../src/captureGuidance.ts";

const config = {
  maxMotionDelta: 0.55,
  stabilityWindowMs: 350,
  updateIntervalMs: 100,
};

test("selects DeviceMotion first and falls back to Accelerometer", () => {
  assert.equal(selectGuidanceSensorSource(true, true), "device-motion");
  assert.equal(selectGuidanceSensorSource(false, true), "accelerometer");
  assert.equal(selectGuidanceSensorSource(false, false), null);
});

test("low-pass gravity filtering preserves units", () => {
  assert.deepEqual(lowPassGravity(null, [0, -9.8, 0], 0.2), [0, -9.8, 0]);
  assert.deepEqual(lowPassGravity([0, -9.8, 0], [1, -8.8, 1], 0.2), [0.2, -9.600000000000001, 0.2]);
});

test("applies only the configured motion tolerance", () => {
  const atMotionLimit = evaluateGuidanceReading(
    [0, -9.80665, 0.55], 100, config, { previousGravity: [0, -9.80665, 0], stableSince: 0 },
  );
  const aboveMotionLimit = evaluateGuidanceReading(
    [0, -9.80665, 0.551], 100, config, { previousGravity: [0, -9.80665, 0], stableSince: 0 },
  );
  assert.equal(atMotionLimit.guidance.stable, true);
  assert.equal(aboveMotionLimit.guidance.stable, false);
});

test("requires an uninterrupted stable window before becoming ready", () => {
  let accumulator = { ...EMPTY_GUIDANCE_ACCUMULATOR };
  let result = evaluateGuidanceReading([0, -9.80665, 0], 0, config, accumulator);
  assert.equal(result.guidance.stable, false);
  assert.equal(result.guidance.ready, false);

  accumulator = result.accumulator;
  result = evaluateGuidanceReading([0, -9.80665, 0], 100, config, accumulator);
  assert.equal(result.guidance.stable, true);
  assert.equal(result.guidance.ready, false);

  accumulator = result.accumulator;
  result = evaluateGuidanceReading([0, -9.80665, 0], 449, config, accumulator);
  assert.equal(result.guidance.ready, false);

  accumulator = result.accumulator;
  result = evaluateGuidanceReading([0, -9.80665, 0], 450, config, accumulator);
  assert.equal(result.guidance.ready, true);
});

test("phone orientation does not block capture guidance, but movement resets stability", () => {
  let accumulator = { previousGravity: [0, -9.80665, 0], stableSince: 0 };
  const rotated = evaluateGuidanceReading([0, 0, 9.80665], 900, config, {
    previousGravity: [0, 0, 9.80665],
    stableSince: 0,
  });
  assert.equal(rotated.guidance.upright, true);
  assert.equal(rotated.guidance.tiltDegrees, null);
  assert.equal(rotated.guidance.ready, true);

  accumulator = rotated.accumulator;
  const moved = evaluateGuidanceReading([0, -9.80665, 0], 1000, config, accumulator);
  assert.equal(moved.guidance.stable, false);
  assert.equal(moved.guidance.ready, false);
});

test("motion-ready state still detects unavailable, unstable, or stale readings", () => {
  const ready = {
    available: true,
    sampledAt: 1000,
    tiltDegrees: 0,
    motionDelta: 0,
    upright: true,
    stable: true,
    ready: true,
  };
  const staleAfter = sensorStaleAfterMs(config);

  assert.equal(staleAfter, 500);
  assert.equal(isCaptureGuidanceReady(ready, 1500, staleAfter), true);
  assert.equal(isCaptureGuidanceReady(ready, 1501, staleAfter), false);
  assert.equal(isCaptureGuidanceReady({ ...ready, available: false }, 1000, staleAfter), false);
  assert.equal(isCaptureGuidanceReady({ ...ready, stable: false }, 1000, staleAfter), false);
  assert.equal(isCaptureGuidanceReady({ ...ready, upright: false }, 1000, staleAfter), true);
});
