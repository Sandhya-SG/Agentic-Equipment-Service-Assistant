---
doc_id: SOP-XR500-025
title: XR-500 Gantry Drive Belt Inspection and Tensioning
revision: B
equipment_model: XR-500
doc_type: sop
---

## Purpose and Scope
This SOP covers inspecting and tensioning the gantry drive belt on the XR-500
handler. Use it when the station reports Fault E-201 (Home Fail) or E-202
(Position Error), or when the gantry makes unusual noise or misses positions.

## Prerequisites
Confirm the handler is in Maintenance Mode and the gantry servo drives are
de-energized (automatic on entering Maintenance Mode). Never manually force the
gantry while drives are energized. Have the belt tension gauge available.

## Procedure
Enter Maintenance Mode. Open the gantry access panel. Visually inspect the drive
belt for cracks, fraying, or missing teeth; a damaged belt must be replaced.
Measure belt tension with the gauge and compare against the specified range. If
tension is low, adjust the tensioner per the marked adjustment points in small
increments, re-measuring after each. Do not over-tension, as this stresses the
bearings.

## Verification
Run a homing self-test in Diagnostic Mode and confirm the gantry homes reliably
and reports no position error across three cycles. If position errors persist with
correct belt tension, the home sensor or motion controller may require further
diagnosis.
