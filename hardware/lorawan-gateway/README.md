# GeoSentry — LoRaWAN Gateway

## Overview

This directory documents the development, configuration, and troubleshooting of the LoRaWAN gateway for the GeoSentry landslide monitoring and early warning system.

## Hardware

- Raspberry Pi
- RAK5146 SPI LoRa concentrator
- LoRaWAN gateway antennas
- MicroSD card

## Software

- Raspberry Pi OS (Raspbian Buster, as shown in the recorded gateway environment)
- SX1303 HAL
- LoRa packet forwarder
- The Things Network gateway software

## Development Progress

### Completed
- Prepared Raspberry Pi operating system and gateway software environment.
- Configured and tested SPI-related software.
- Conducted gateway startup and reset troubleshooting.
- Identified EXT4 filesystem errors on the Raspberry Pi SD card.
- Used Ubuntu Live USB to access gateway files.
- Created gateway and system configuration backups.
- Verified the gateway files archive.

### Pending
- Complete filesystem recovery and verification.
- Resolve SPI concentrator initialization failures.
- Verify gateway connectivity with the LoRaWAN network server.
- Test receiving packets from the LoRaWAN sensor node.
- Integrate forwarded sensor data with the GeoSentry Flask backend.

## Current Status

**Under development — gateway not yet operational.**

The gateway initialization issue remains unresolved. Filesystem recovery and hardware verification are ongoing.

## Notes

Backup images, private credentials, and device-specific configuration secrets are stored separately and are not included in this repository.