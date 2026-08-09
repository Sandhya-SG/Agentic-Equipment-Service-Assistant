---
doc_id: MAN-XR500-001
title: XR-500 Semiconductor Test Handler — Service Manual
revision: C
equipment_model: XR-500
doc_type: manual
---

## Overview
The XR-500 is an automated pick-and-place test handler used to move semiconductor devices between input trays, the test contactor socket, and output bins. This manual covers routine service, fault isolation, and component replacement for the handler subsystems: the gantry, the pick head, the contactor interface, and the thermal conditioning unit. Always confirm the handler is in Maintenance Mode before performing any physical service.

## Handler Operational Modes
The XR-500 operates in three modes. Production Mode runs the full test flow at rated throughput. Maintenance Mode disables automatic motion and unlocks service panels; it is required for all hands-on service. Diagnostic Mode runs individual subsystem self-tests and reports status codes without moving devices. Switching to Maintenance Mode automatically de-energizes the gantry servo drives.

## Pick Head Service
The pick head uses vacuum to grip devices. Loss of grip is most often caused by a clogged vacuum filter, a cracked nozzle tip, or a failed vacuum sensor. To service the pick head, enter Maintenance Mode, then inspect the nozzle tip for chips or debris. Replace the vacuum filter cartridge if the vacuum gauge reads below 60 kPa during a grip self-test. Reseat the nozzle and confirm the gauge returns to the 80-90 kPa working range.

## Gantry and Motion
The gantry positions the pick head over the trays, socket, and bins. Positioning errors present as missed picks or misaligned placement. Check the drive belt tension and confirm the home sensor triggers during a homing self-test in Diagnostic Mode. A gantry that fails to home usually indicates a blocked home sensor or a loose belt. Do not manually force the gantry while the servo drives are energized.

## Contactor Interface
The contactor socket makes temporary electrical contact with the device under test. Intermittent test failures across many devices often trace to worn contactor pins or contamination on the socket. Inspect the pins under magnification for flattening or bent tips. Clean the socket with the approved solvent and a lint-free swab. Replace the contactor assembly if pin planarity is out of specification.

## Thermal Conditioning Unit
The thermal unit heats or cools devices to the target test temperature. Devices arriving at the socket outside the temperature window will produce systematic test failures. Verify the thermal setpoint matches the test program and confirm the coolant loop is circulating. A thermal unit that cannot reach setpoint may indicate a coolant flow fault or a failed heater element.
