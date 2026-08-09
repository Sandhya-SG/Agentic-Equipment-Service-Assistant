---
doc_id: SOP-XR500-021
title: XR-500 Contactor Pin Inspection and Cleaning
revision: A
equipment_model: XR-500
doc_type: sop
---

## Purpose and Scope
This SOP covers inspection and cleaning of the contactor socket pins on the XR-500 handler. Use it when devices show intermittent or systematic test failures that are suspected to originate at the contactor interface rather than the devices themselves.

## Prerequisites
Enter Maintenance Mode. The contactor interface carries test signals only when a device is seated and the socket is actuated; in Maintenance Mode the socket is inactive. Have magnification (10x loupe), the approved cleaning solvent, and lint-free swabs available. Wear an ESD wrist strap connected to the handler ground point before touching the socket.

## Procedure
Open the contactor access panel. Using the loupe, inspect the pin field for flattened tips, bent pins, or debris bridging adjacent pins. Note any pins that sit below the surrounding plane. Dampen a lint-free swab with the approved solvent and gently wipe the pin field in one direction. Allow the socket to dry fully before reactivation. Do not apply lateral force that could bend pins.

## Verification
Reseat a known-good reference device and run a continuity self-test in Diagnostic Mode. All pins should report contact. If specific pins repeatedly fail continuity, the contactor assembly is out of specification and should be replaced per the service manual.
