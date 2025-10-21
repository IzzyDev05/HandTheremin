from pyo import Server, Sine, SigTo, ButLP, Mix
import random

class AudioSynthesizer:
    def __init__(self, init_freq=261.63, init_amp=0.5, voices=2, detune_cents=3, bend_range=2):
        """
        Continuous tone synthesizer with amplitude, frequency, and pitch bend control.
        :param init_freq: Initial frequency (Hz)
        :param init_amp: Initial amplitude (0–1)
        :param voices: Number of detuned oscillators
        :param detune_cents: Detune range (+/- cents)
        :param bend_range: Max pitch bend range in semitones (+/-)
        """
        # --- Audio Server ---
        self.server = Server().boot()
        self.server.start()

        # --- Parameters ---
        self.base_freq = init_freq
        self.bend_range = bend_range
        self.pitch_mod = 0.0  # range: -1.0 to +1.0

        # --- Smooth control signals ---
        self.freq_sig = SigTo(value=init_freq, time=0.05)
        self.amp_sig = SigTo(value=init_amp, time=0.1)

        # --- Create detuned oscillators ---
        self.oscillators = []
        for i in range(voices):
            cents = random.uniform(-detune_cents, detune_cents)
            detune_ratio = 2 ** (cents / 1200.0)
            osc = Sine(freq=self.freq_sig * detune_ratio, mul=self.amp_sig / voices)
            self.oscillators.append(osc)

        # --- Mix and filter ---
        mix = Mix(self.oscillators, voices=2)
        self.filtered = ButLP(mix, freq=800).out()

    # ======================
    # Control methods
    # ======================
    def set_frequency(self, freq: float):
        """Smoothly update the *base* frequency (not including modulation)."""
        self.base_freq = float(freq)
        self._update_effective_frequency()

    def set_amplitude(self, amp: float):
        """Smoothly update overall amplitude."""
        self.amp_sig.value = float(amp)

    def set_pitch_mod(self, mod: float):
        """Update pitch bend (from -1 to +1)."""
        self.pitch_mod = max(-1.0, min(1.0, mod))
        self._update_effective_frequency()

    def _update_effective_frequency(self):
        """Recalculate actual frequency with pitch bend applied."""
        bend_semitones = self.pitch_mod * self.bend_range
        bend_ratio = 2 ** (bend_semitones / 12.0)
        effective_freq = self.base_freq * bend_ratio
        self.freq_sig.value = effective_freq

    def stop(self):
        """Stop the synth and release resources."""
        self.server.stop()
        self.server.shutdown()
