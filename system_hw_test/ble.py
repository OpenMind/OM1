# Scan for nearby Bluetooth LE devices and their services

import asyncio

from bleak import BleakClient, BleakScanner


async def main():
    devices = await BleakScanner.discover()
    for device in devices:
        print()
        print(f"Name: {device.name}")
        print(f"Address: {device.address}")
        print(f"Details: {device.details}")
        print(f"Metadata: {device.metadata}")
        print(f"RSSI: {device.rssi}")

    for device in devices:
        try:
            this_device = await BleakScanner.find_device_by_address(
                device.address, timeout=20
            )
            async with BleakClient(this_device) as client:
                print("Services found for device")
                print(f"\tDevice address:{device.address}")
                print(f"\tDevice name:{device.name}")

                print("\tServices:")
                for service in client.services:
                    print()
                    print(f"\t\tDescription: {service.description}")
                    print(f"\t\tService: {service}")

                    print("\t\tCharacteristics:")
                    for c in service.characteristics:
                        print()
                        print(f"\t\t\tUUID: {c.uuid}"),
                        print(f"\t\t\tDescription: {c.uuid}")
                        print(f"\t\t\tHandle: {c.uuid}"),
                        print(f"\t\t\tProperties: {c.uuid}")

                        print("\t\tDescriptors:")
                        for descrip in c.descriptors:
                            print(f"\t\t\t{descrip}")

        except Exception as e:
            print(f"Could not connect to device with info: {device}")
            print(f"Error: {e}")


asyncio.run(main())
