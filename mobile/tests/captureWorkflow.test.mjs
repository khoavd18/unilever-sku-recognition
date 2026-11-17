import assert from "node:assert/strict";
import test from "node:test";

import {
  captureActionState,
  captureInstruction,
} from "../src/captureWorkflow.ts";

test("recognition flow exposes one primary action per step", () => {
  assert.deepEqual(captureActionState("recognition", 0, true), {
    action: "capture",
    title: "Chụp ảnh 1/6",
    disabled: false,
  });
  assert.equal(captureActionState("recognition", 1, false).disabled, true);
  assert.deepEqual(captureActionState("recognition", 2, false), {
    action: "process_recognition",
    title: "Ghép và nhận diện · 2 ảnh",
    disabled: false,
  });
  assert.equal(captureActionState("recognition", 6, false).disabled, false);
  assert.equal(captureActionState("recognition", 7, false).disabled, true);
});

test("stitch flow keeps 2-6 image processing and optional extra capture", () => {
  assert.equal(captureActionState("stitch", 0, true).action, "capture");
  assert.equal(captureActionState("stitch", 2, false).action, "process_stitch");
  assert.equal(captureActionState("stitch", 6, false).disabled, false);
  assert.equal(captureActionState("stitch", 7, false).disabled, true);
});

test("instructions explain first frame and overlap direction", () => {
  assert.match(captureInstruction("recognition", 0), /Căn mặt kệ vào khung/);
  assert.match(captureInstruction("recognition", 1, "right"), /sang phải/);
  assert.match(captureInstruction("recognition", 1, "left"), /sang trái/);
  assert.match(captureInstruction("recognition", 2, "right"), /nhận diện ngay/);
  assert.match(captureInstruction("recognition", 6, "right"), /Đã đủ 6 ảnh/);
});
