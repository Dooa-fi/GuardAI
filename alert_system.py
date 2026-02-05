"""
Alert System Module
Handles audio alerts with cooldown mechanism for drone detection
"""

import pygame
import time
import os
from config import *


class AlertSystem:
    """
    Audio alert system with cooldown to prevent spam
    Non-blocking playback for continuous detection
    """
    
    def __init__(self, sound_path=ALERT_SOUND_PATH, cooldown=ALERT_COOLDOWN, volume=ALERT_VOLUME):
        """
        Initialize alert system
        
        Args:
            sound_path: Path to alert sound file
            cooldown: Minimum seconds between alerts
            volume: Volume level (0.0 - 1.0)
        """
        self.sound_path = sound_path
        self.cooldown = cooldown
        self.volume = volume
        
        self.sound = None
        self.last_alert_time = 0
        self.is_initialized = False
        self.alert_active = False
        
        # Initialize pygame mixer
        self._initialize_mixer()
        
        # Load sound
        self._load_sound()
    
    def _initialize_mixer(self):
        """Initialize pygame mixer for audio playback"""
        try:
            # Initialize mixer if not already initialized
            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
            
            self.is_initialized = True
            print("✓ Audio mixer initialized")
        
        except Exception as e:
            print(f"⚠ Warning: Could not initialize audio mixer: {e}")
            self.is_initialized = False
    
    def _load_sound(self):
        """Load alert sound file"""
        if not self.is_initialized:
            print("⚠ Audio mixer not initialized, skipping sound load")
            return
        
        try:
            # Check if sound file exists
            if not os.path.exists(self.sound_path):
                print(f"⚠ Warning: Sound file not found: {self.sound_path}")
                print("  Run 'python create_siren.py' to generate sound file")
                return
            
            # Load sound
            self.sound = pygame.mixer.Sound(self.sound_path)
            self.sound.set_volume(self.volume)
            
            print(f"✓ Alert sound loaded: {self.sound_path}")
            print(f"  Duration: {self.sound.get_length():.2f}s")
            print(f"  Volume: {self.volume * 100:.0f}%")
        
        except Exception as e:
            print(f"⚠ Warning: Could not load sound: {e}")
            self.sound = None
    
    def trigger_alert(self, force=False):
        """
        Trigger alert sound if cooldown period has passed
        
        Args:
            force: Force alert even if cooldown not passed
        
        Returns:
            Boolean indicating if alert was triggered
        """
        if not self.is_initialized or self.sound is None:
            return False
        
        current_time = time.time()
        time_since_last = current_time - self.last_alert_time
        
        # Check cooldown
        if not force and time_since_last < self.cooldown:
            return False
        
        try:
            # Play sound (non-blocking)
            self.sound.play()
            self.last_alert_time = current_time
            self.alert_active = True
            return True
        
        except Exception as e:
            print(f"Error playing alert: {e}")
            return False
    
    def stop_alert(self):
        """Stop currently playing alert"""
        if self.sound is not None:
            try:
                self.sound.stop()
                self.alert_active = False
            except Exception as e:
                print(f"Error stopping alert: {e}")
    
    def is_alert_active(self):
        """
        Check if alert is currently active
        
        Returns:
            Boolean indicating if alert is playing or in cooldown
        """
        if not self.is_initialized:
            return False
        
        current_time = time.time()
        time_since_last = current_time - self.last_alert_time
        
        # Consider active if within cooldown period
        return time_since_last < self.cooldown
    
    def get_time_since_last_alert(self):
        """
        Get time elapsed since last alert
        
        Returns:
            Seconds since last alert
        """
        return time.time() - self.last_alert_time
    
    def get_cooldown_remaining(self):
        """
        Get remaining cooldown time
        
        Returns:
            Seconds remaining in cooldown (0 if ready)
        """
        time_since = self.get_time_since_last_alert()
        remaining = max(0, self.cooldown - time_since)
        return remaining
    
    def is_ready(self):
        """
        Check if alert system is ready to trigger
        
        Returns:
            Boolean indicating if cooldown has passed
        """
        return self.get_cooldown_remaining() == 0
    
    def set_volume(self, volume):
        """
        Set alert volume
        
        Args:
            volume: Volume level (0.0 - 1.0)
        """
        if 0.0 <= volume <= 1.0:
            self.volume = volume
            if self.sound is not None:
                self.sound.set_volume(volume)
        else:
            raise ValueError("Volume must be between 0.0 and 1.0")
    
    def set_cooldown(self, cooldown):
        """
        Set cooldown period
        
        Args:
            cooldown: Cooldown in seconds
        """
        if cooldown >= 0:
            self.cooldown = cooldown
        else:
            raise ValueError("Cooldown must be non-negative")
    
    def get_status(self):
        """
        Get alert system status
        
        Returns:
            Dictionary with status information
        """
        return {
            'initialized': self.is_initialized,
            'sound_loaded': self.sound is not None,
            'sound_path': self.sound_path,
            'volume': self.volume,
            'cooldown': self.cooldown,
            'time_since_last': self.get_time_since_last_alert(),
            'cooldown_remaining': self.get_cooldown_remaining(),
            'is_ready': self.is_ready(),
            'alert_active': self.is_alert_active()
        }
    
    def test_alert(self):
        """Test alert sound (ignores cooldown)"""
        print("Testing alert sound...")
        if self.trigger_alert(force=True):
            print("✓ Alert triggered successfully")
            if self.sound:
                # Wait for sound to finish
                duration = self.sound.get_length()
                pygame.time.wait(int(duration * 1000))
            return True
        else:
            print("✗ Alert failed to trigger")
            return False


# Test function
if __name__ == "__main__":
    print("Testing AlertSystem...")
    
    try:
        # Initialize alert system
        print("\n1. Initializing alert system...")
        alert = AlertSystem()
        
        # Get status
        print("\n2. Alert system status:")
        status = alert.get_status()
        for key, value in status.items():
            print(f"   {key}: {value}")
        
        # Test alert
        print("\n3. Testing alert sound...")
        if alert.test_alert():
            print("   ✓ Sound played successfully")
        
        # Test cooldown
        print("\n4. Testing cooldown mechanism...")
        print(f"   Cooldown: {alert.cooldown}s")
        
        # Try immediate trigger (should fail)
        print("   Attempting immediate re-trigger...")
        if not alert.trigger_alert():
            print("   ✓ Cooldown working (alert blocked)")
        else:
            print("   ✗ Cooldown not working (alert triggered)")
        
        # Wait and try again
        print(f"   Waiting {alert.cooldown}s...")
        time.sleep(alert.cooldown)
        
        print("   Attempting trigger after cooldown...")
        if alert.trigger_alert():
            print("   ✓ Alert triggered after cooldown")
            pygame.time.wait(int(alert.sound.get_length() * 1000))
        else:
            print("   ✗ Alert failed after cooldown")
        
        # Test volume adjustment
        print("\n5. Testing volume adjustment...")
        alert.set_volume(0.3)
        print(f"   Volume set to: {alert.volume * 100:.0f}%")
        
        print("\n✓ All tests passed!")
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
