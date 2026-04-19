"""
vibrator.py — Tray vibrator hardware interface.

Controls the vibrator motor via a GPIO pin (Jetson.GPIO).
Falls back to a mock implementation on non-Jetson platforms (dev/Windows).
"""

VIBRATOR_PIN = None  # Set by config; imported lazily to avoid circular import

try:
    import Jetson.GPIO as GPIO
    _HAS_GPIO = True
except Exception:
    _HAS_GPIO = False


class _MockGPIO:
    BCM = OUT = HIGH = LOW = None

    def setmode(self, *a): pass
    def setup(self, *a, **kw): pass
    def output(self, pin, val): pass
    def cleanup(self, *a): pass


if not _HAS_GPIO:
    GPIO = _MockGPIO()


class Vibrator:
    """Controls a vibrator relay on a single GPIO pin."""

    def __init__(self, pin: int):
        self._pin = pin
        self._running = False
        if _HAS_GPIO:
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self._pin, GPIO.OUT, initial=GPIO.LOW)

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self):
        if self._running:
            return
        GPIO.output(self._pin, GPIO.HIGH)
        self._running = True

    def stop(self):
        if not self._running:
            return
        GPIO.output(self._pin, GPIO.LOW)
        self._running = False

    def restart(self):
        self.stop()
        self.start()

    def cleanup(self):
        self.stop()
        if _HAS_GPIO:
            GPIO.cleanup(self._pin)
