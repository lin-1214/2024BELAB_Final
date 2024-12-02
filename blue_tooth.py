import bluetooth

def send_message(mac_address):
    
    try:
        sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
        
        print(f"Connecting to {mac_address}...")
        sock.connect((mac_address, 1))  
        sock.send("Alarm activated!")
        print("Message sent.")

    except bluetooth.BluetoothError as e:
        print(f"Bluetooth error: {e}")

    finally:
        print("Closing the connection...")
        sock.close()
