from pyo import Server, Sine, SigTo, ButLP, ButHP, Mix, Noise
import random

class AudioSynthesizer:
    def __init__(self, init_freq=261.63, init_amp=0.5, voices=2, detune_cents=3, bend_range=2):
        """
        Analog-style Theremin synthesizer with realistic tone shaping.
        :param init_freq: Initial base frequency (Hz)
        :param init_amp: Initial amplitude (0–1)
        :param voices: Number of slightly detuned oscillators
        :param detune_cents: Random detune range per oscillator (+/-)
        :param bend_range: Pitch bend range in semitones (+/-)
        """
        # --- Server setup ---
        self.server = Server().boot()
        self.server.start()

        # --- Main control parameters ---
        self.base_freq = init_freq
        self.pitch_mod = 0.0
        self.bend_range = bend_range

        # Smooth frequency and amplitude control
        self.freq_sig = SigTo(value=init_freq, time=0.05)
        self.amp_sig = SigTo(value=init_amp, time=0.1)

        # --- Subtle vibrato (natural hand tremor feel) ---
        self.vibrato = Sine(freq=5, mul=0.01, add=1)  # ±1% pitch variation

        # --- Dual detuned oscillators (heterodyning simulation) ---
        self.oscillators = []
        for i in range(voices):
            cents = random.uniform(-detune_cents, detune_cents)
            detune_ratio = 2 ** (cents / 1200.0)
            osc = Sine(freq=self.freq_sig * detune_ratio * self.vibrato, mul=self.amp_sig / voices)
            self.oscillators.append(osc)

        # --- Add gentle harmonic content (2nd harmonic at 20%) ---
        harmonic = Sine(freq=self.freq_sig * 2, mul=self.amp_sig * 0.2)

        # --- Add faint background noise (for analog “air”) ---
        noise = Noise(mul=self.amp_sig * 0.02)

        # --- Mix everything together ---
        mix = Mix(self.oscillators + [harmonic, noise], voices=2)

        # --- Filter to soften the top end (adjust freq for brightness) ---
        self.filtered = ButLP(mix, freq=1200).out()
        #self.filtered = ButHP(mix).out()

    # ======================
    # Control methods
    # ======================
    def set_frequency(self, freq: float):
        """Smoothly update the base frequency (no modulation applied)."""
        self.base_freq = float(freq)
        self._update_effective_frequency()

    def set_amplitude(self, amp: float):
        """Smoothly update overall amplitude."""
        self.amp_sig.value = float(max(0.0, min(1.0, amp)))

    def set_pitch_mod(self, mod: float):
        """Update pitch bend (-1 to +1)."""
        self.pitch_mod = max(-1.0, min(1.0, mod))
        self._update_effective_frequency()

    def _update_effective_frequency(self):
        """Recalculate final oscillator frequency with pitch bend."""
        bend_semitones = self.pitch_mod * self.bend_range
        bend_ratio = 2 ** (bend_semitones / 12.0)
        effective_freq = self.base_freq * bend_ratio
        self.freq_sig.value = effective_freq

    def stop(self):
        """Stop and clean up."""
        self.server.stop()
        self.server.shutdown()
