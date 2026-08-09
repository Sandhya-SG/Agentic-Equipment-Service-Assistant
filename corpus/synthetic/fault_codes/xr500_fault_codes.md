---
doc_id: FC-XR500-002
title: XR-500 Fault Code Reference
revision: E
equipment_model: XR-500
doc_type: fault_codes
---

## Vacuum and Pick Faults
Fault E-101 (Vacuum Low) indicates the pick head could not achieve grip vacuum. Likely causes are a clogged vacuum filter, a cracked nozzle tip, or a leaking vacuum line. Recommended action: run a grip self-test and replace the vacuum filter if the gauge reads below 60 kPa. Fault E-102 (Grip Lost) indicates vacuum was achieved but lost during transport, commonly from a marginal nozzle seal or a partially clogged filter.

## Motion and Gantry Faults
Fault E-201 (Home Fail) indicates the gantry did not detect its home position during homing. Likely causes are a blocked home sensor or a loose drive belt. Recommended action: inspect the home sensor and check belt tension in Diagnostic Mode. Fault E-202 (Position Error) indicates the gantry overshot or undershot a target position, often due to belt wear or an obstruction in the motion path.

## Contactor and Test Faults
Fault E-301 (Continuity Fail) indicates one or more contactor pins did not make contact with the seated device. Likely causes are worn or bent pins or socket contamination. Recommended action: inspect and clean the contactor per SOP, and replace the assembly if pins are out of planarity. Fault E-302 (Systematic Test Fail) across many devices suggests a socket-level issue rather than device failures.

## Thermal Faults
Fault E-401 (Setpoint Not Reached) indicates the thermal unit could not achieve the target temperature. Likely causes are a coolant flow fault or a failed heater element. Note that heater element service involves high voltage and stored energy and must be escalated to a qualified engineer rather than handled in the field. Fault E-402 (Coolant Flow Low) indicates insufficient coolant circulation.
