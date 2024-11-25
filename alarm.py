import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# Define GPIO pins
BUZZER_PIN = 17  # GPIO17
LED_PIN = 18     # GPIO18

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

if __name__ == "__main__":
    try:
        blink_alarm(3)

    finally:
        GPIO.cleanup()
