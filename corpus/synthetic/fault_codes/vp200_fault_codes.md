---
doc_id: FC-VP200-003
title: VP-200 Fault Code Reference
revision: D
equipment_model: VP-200
doc_type: fault_codes
---

## Imaging and Optics Faults
Fault V-101 (Image Blur) indicates the captured image failed the sharpness check.
Likely causes are a contaminated lens, a defocused camera, or a loose camera
mount. Recommended action: clean the lens per SOP and re-run the focus check in
Calibration Mode. Fault V-102 (No Image) indicates the camera returned no frame,
commonly from a disconnected camera cable or a camera power fault.

## Stage and Motion Faults
Fault V-201 (Stage Home Fail) indicates the stage did not reach its home position.
Likely causes are a blocked home sensor or debris on the rails. Recommended
action: inspect and clean the rails and home sensor, then re-run homing. Fault
V-202 (Stage Timeout) indicates the stage did not reach a target position in time,
often due to an obstruction or belt wear.

## Lighting Faults
Fault V-301 (Lighting Channel Fail) indicates one or more LED channels did not
illuminate during self-test. Likely causes are a failed LED bank or a loose
lighting connector. Recommended action: check the lighting connectors and replace
the LED bank if a channel remains dark. Note that lighting driver service involves
high voltage and stored energy and must be escalated to a qualified engineer.

## Calibration and Recipe Faults
Fault V-401 (Calibration Drift) indicates alignment or lighting calibration is out
of tolerance. Recommended action: re-run calibration in Calibration Mode before
suspecting hardware. Fault V-402 (Recipe Mismatch) indicates the active inspection
recipe does not match the device type loaded; confirm and load the correct recipe.
