import RPi.GPIO as GPIO
import time
import blue_tooth
import bluetooth  # You'll need to install pybluez: pip install pybluez
from bluetooth import *
import sys

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# Define GPIO pins
BUZZER_PIN = 18  # GPIO18
LED_PIN = 8     # GPIO8

def blink_alarm(times=3, interval=0.5):
    try:
        for _ in range(times):
            GPIO.output(BUZZER_PIN, GPIO.HIGH)
            GPIO.output(LED_PIN, GPIO.HIGH)
            time.sleep(interval)
            GPIO.output(BUZZER_PIN, GPIO.LOW)
            GPIO.output(LED_PIN, GPIO.LOW)
            time.sleep(interval)
            
    except KeyboardInterrupt:
        pass
    
    finally:
        GPIO.output(BUZZER_PIN, GPIO.LOW)
        GPIO.output(LED_PIN, GPIO.LOW)

def connect_bluetooth(target_mac):
    print(f"🔄 Attempting to connect to device: {target_mac}")
    
    try:
        # Create a Bluetooth socket
        sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
        
        # Search for available services on the target device
        services = bluetooth.find_service(address=target_mac)
        
        if not services:
            print("❌ No Bluetooth services found on target device")
            return False
            
        # Connect to the first available service
        port = services[0]["port"]
        name = services[0]["name"]
        host = services[0]["host"]
        
        print(f"📡 Found service: {name}")
        print(f"🔌 Connecting to {host} on port {port}")
        
        # Attempt connection
        sock.connect((host, port))
        print("✅ Successfully connected!")
        
        return sock
        
    except bluetooth.BluetoothError as e:
        print(f"❌ Bluetooth Error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def send_bluetooth_message(sock, message):
    try:
        if sock:
            sock.send(message)
            print(f"📤 Sent message: {message}")
            
            # Wait for response (optional)
            data = sock.recv(1024)
            print(f"📥 Received: {data}")
            return True
    except Exception as e:
        print(f"❌ Failed to send message: {e}")
        return False
    
def disconnect_bluetooth(sock):
    if sock:
        sock.close()
        print("🔌 Bluetooth connection closed")

if __name__ == "__main__":
    try:
        # Setup GPIO pins
        GPIO.setup(BUZZER_PIN, GPIO.OUT)
        GPIO.setup(LED_PIN, GPIO.OUT)
        
        if len(sys.argv) != 2:
            print("Usage: python alarm.py <target_mac>")
            sys.exit(1)
        
        target_mac = sys.argv[1]
        
        # Trigger alarm
        blink_alarm(3)
        
        # Connect and send message
        bt_socket = connect_bluetooth(target_mac)
        if bt_socket:
            send_bluetooth_message(bt_socket, "Alarm triggered!")
            disconnect_bluetooth(bt_socket)
        else:
            print("❌ Failed to connect to the target device")
        
    except KeyboardInterrupt:
        print("\n⚠️ Program interrupted by user")
    finally:
        GPIO.cleanup()
        print("🧹 GPIO cleaned up")
