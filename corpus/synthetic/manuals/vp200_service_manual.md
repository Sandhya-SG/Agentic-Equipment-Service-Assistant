---
doc_id: MAN-VP200-001
title: VP-200 Vision Inspection Station — Service Manual
revision: B
equipment_model: VP-200
doc_type: manual
---

## Overview
The VP-200 is an automated optical inspection station used to detect surface
defects, lead coplanarity issues, and marking errors on semiconductor devices.
It combines a motorized stage, a multi-axis camera head, programmable lighting,
and an image-processing pipeline. This manual covers service of the optics, the
stage, the lighting subsystem, and the image controller. Confirm the station is
in Service Mode before any physical maintenance.

## Station Operating Modes
The VP-200 has three modes. Inspection Mode runs automated defect detection at
production rate. Service Mode disables stage motion and unlocks the optics
enclosure for maintenance. Calibration Mode runs alignment and lighting
calibration routines and reports status without inspecting production devices.
Entering Service Mode parks the camera head and disables the stage servo.

## Optics and Camera Head
The camera head captures images for defect analysis. Blurred or inconsistent
images are usually caused by a contaminated lens, a defocused camera, or a loose
mount. To service the optics, enter Service Mode, then inspect the lens surface
for dust or residue. Clean the lens only with the approved optical wipe and
solvent. Confirm focus using the calibration target in Calibration Mode. Do not
touch the lens surface with bare hands.

## Motorized Stage
The stage positions the device under the camera. Positioning errors show up as
misframed images or inspection timeouts. Check the stage belt tension and confirm
the stage home sensor triggers during a homing routine in Calibration Mode. A
stage that fails to home usually indicates a blocked home sensor or debris on the
rails. Clean the rails with a lint-free cloth; do not lubricate unless the manual
specifies it.

## Lighting Subsystem
The programmable LED array illuminates devices for imaging. Uneven or dim
lighting causes false defect calls. Verify each lighting channel activates during
a lighting self-test. A channel that fails to illuminate may indicate a failed
LED bank or a loose lighting connector. Replace the LED bank as a unit; individual
LEDs are not field-replaceable.

## Image Controller
The image controller runs the defect-detection pipeline. Systematic
misclassification across many devices often indicates a calibration drift or an
out-of-date inspection recipe rather than a hardware fault. Re-run lighting and
alignment calibration before suspecting hardware. Confirm the active recipe
matches the device type under inspection.
