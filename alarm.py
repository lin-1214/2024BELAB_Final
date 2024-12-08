import RPi.GPIO as GPIO
import time
import serial
import sys

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# Define GPIO pins
BUZZER_PIN = 24  # GPIO24
LED_PIN = 4      # GPIO4

# HC-05 Serial setup
# SERIAL_PORT = '/dev/ttyS0'
SERIAL_PORT = '/dev/serial0'
BAUD_RATE = 9600

def setup_bluetooth():
    try:
        ser = serial.Serial(
            port=SERIAL_PORT,
            baudrate=BAUD_RATE,
            timeout=1,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            bytesize=serial.EIGHTBITS
        )

        time.sleep(0.1)
        return ser
    except Exception as e:
        print(f"❌ Error setting up HC-05: {e}")
        return None

def send_message(ser, message):
    try:
        # Convert string message to bytes using encode()
        ser.write(message.encode('utf-8'))  
        print(f"✅ Sent byte value: {message.encode('utf-8')}")
    except Exception as e:
        print(f"❌ Error sending message: {e}")

def blink_alarm(times=3, interval=0.5):
    try:
        for _ in range(times):
            GPIO.output(BUZZER_PIN, GPIO.HIGH)
            GPIO.output(LED_PIN, GPIO.HIGH)
            time.sleep(interval/2)
            GPIO.output(BUZZER_PIN, GPIO.LOW)
            GPIO.output(LED_PIN, GPIO.LOW)
            time.sleep(interval)
            
    except KeyboardInterrupt:
        pass
    
    finally:
        GPIO.output(BUZZER_PIN, GPIO.LOW)
        GPIO.output(LED_PIN, GPIO.LOW)

if __name__ == "__main__":
    bluetooth_serial = None

    try:
        # Setup GPIO pins
        GPIO.setup(BUZZER_PIN, GPIO.OUT)
        GPIO.setup(LED_PIN, GPIO.OUT)
        
        # Test blink_alarm
        blink_alarm()
        
        # Setup HC-05
        bluetooth_serial = setup_bluetooth()
        if bluetooth_serial is None:
            print("❌ Could not establish Bluetooth connection. Exiting...")
            sys.exit(1)
        
        # Main loop
        while True:
            try:
                send_message(bluetooth_serial, "1")
                time.sleep(1)

            except KeyboardInterrupt:
                print("\n⚠️ Program interrupted by user")
                break
            except Exception as e:
                print(f"❌ Error during communication: {e}")
                break
                
    except KeyboardInterrupt:
        print("\n⚠️ Program interrupted by user")
    finally:
        if bluetooth_serial:
            bluetooth_serial.close()
        GPIO.cleanup()
        print("🧹 Cleanup complete")
