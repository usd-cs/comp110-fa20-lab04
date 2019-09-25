"""
Module: sound

Module containing classes and functions for working with sound files,
specifically WAV files.

DO NOT MODIFY THIS FILE IN ANY WAY!!!!

Authors:
1) Sat Garcia @ USD
2) Dan Zingaro @ UToronto
"""

import math
import os
import sounddevice
import numpy
import scipy.io.wavfile


"""
The Sample classes support the Sound class and allow manipulation of
individual sample values.
"""

class MonoSample():
    """A sample in a single-channeled Sound with a value."""

    def __init__(self, samp_array, i):
        """Create a MonoSample object at index i from numpy array object
        samp_array, which has access to the Sound's buffer."""

        # negative indices are supported
        if -len(samp_array) <= i <= len(samp_array) - 1:
            self.samp_array = samp_array
            self.index = i
        else:
            raise IndexError('Sample index out of bounds.')


    def __str__(self):
        """Return a str with index and value information."""

        return "Sample at " + str(self.index) + " with value " \
            + str(self.get_value())


    def set_value(self, val):
        """Set this Sample's value to val."""

        self.samp_array[self.index] = int(val)


    def get_value(self):
        """Return this Sample's value."""

        return int(self.samp_array[self.index])


    def get_index(self):
        """Return this Sample's index."""

        return self.index

    def __cmp__(self, other):
        return cmp(self.samp_array[self.index], other.samp_array[other.index])


class StereoSample():
    """A sample in a two-channeled Sound with a left and a right value.

    Attributes:
        all_samples (numpy.ndarray): All of the samples in the associated sound.
        index (int): Index in all_samples where sample is located.
    """

    def __init__(self, all_samples, i):
        """Initialize a StereoSample object.

        Parameters:
            all_samples (numpy.ndarray): All the samples in the sound.
            i (int): Index where this ample is located.

        Raises:
            IndexError: If i isn't a valid index in all_samples
        """

        # negative indices are supported
        if -len(all_samples) <= i <= len(all_samples) - 1:
            self.all_samples = all_samples
            self.index = i
        else:
            raise IndexError('Sample index out of bounds.')


    def __str__(self):
        """Returns a string representation of this sample."""

        return "Sample at " + str(self.index) + " with left value " \
            + str(self.get_left()) + " and right value " + \
            str(self.get_right())


    def set_values(self, left, right):
        """Set this sample's left and right channel values.

        Parameters:
            left (int): New value for the left channel.
            right (int): New value for the right channel.
        """
        if not isinstance(left, int) or not isinstance(right,int):
            raise TypeError("Channel values must both be int")


        self.all_samples[self.index] = [int(left), int(right)]


    def get_values(self):
        """Return this sample's left and right values.

        Returns:
            left (int): Current value of left channel.
            right (int): Current value of right channel.
        """

        return (self.all_samples[self.index, 0], self.all_samples[self.index, 1])


    def set_left(self, new_left_val):
        """Set this sample's left channel value.

        Parameters:
            new_left_val (int): New value for the left channel.
        """
        if not isinstance(new_left_val, int):
            raise TypeError("Channel value must be an int")

        self.all_samples[self.index, 0] = int(new_left_val)


    def set_right(self, new_right_val):
        """Set this sample's right channel value.

        Parameters:
            new_right_val (int): New value for the right channel.
        """
        if not isinstance(new_right_val, int):
            raise TypeError("Channel value must be an int")

        self.all_samples[self.index, 1] = int(new_right_val)


    def get_left(self):
        """Return this sample's left value.

        Returns:
            left (int): Current value of the left channel.
        """

        return int(self.get_values()[0])


    def get_right(self):
        """Return this sample's right value.

        Returns:
            right (int): Current value of the right channel.
        """

        return int(self.get_values()[1])


    def get_index(self):
        """Return this sample's index.

        Returns:
            index (int): The sample's index."""

        return self.index

    def __eq__(self, other):
        """Checks whether this sample and another are equal.
        Two samples are considered equal if their left and right channels have
        the same values.

        Parameters:
            other (SoundSample): Another sound sample to compare.

        Returns:
            equal (bool): Whether two samples are equal (True) or not (False)
        """

        return self.get_left() == other.get_left() \
                and self.get_right() == other.get_right()


class Sound():
    """
    A class representing audio. A sound object consists of a sequence of
    samples.

    Attributes:
        sample_rate (int): Rate at which sound was sampled.
        samples (numpy.ndarray): Sampled data for the sound.
        channels (int): Number of channels in the sound.
        filename (str, optional): Name of file where sound was read from.
        numpy_encoding (numpy.dtype): Encoding used for data.
    """

    def __init__(self, filename=None, samples=None):
        """Create a new Sound object.

        This new sound object is based either on a file (when filename is
        given) or an existing set of samples (when samples is given).
        If both filename and samples is given, the filename takes precedence
        and is used to create the new object.

        Parameters:
            filename (str, optional): The name of a file containing a wav encoded sound.
            samples ((int, numpy.ndarray), optional): Tuple containing sample rate and samples.

        Raises:
            RuntimeError: When neither filename or samples parameter is given.
        """

        self.numpy_encoding = numpy.dtype('int16')  # default encoding
        self.set_filename(filename)

        if filename is not None:
            self.sample_rate, sample_array = scipy.io.wavfile.read(filename)
            self.samples = numpy.ndarray.copy(sample_array)

        elif samples is not None:
            self.sample_rate, self.samples = samples

        else:
            raise RuntimeError("No arguments were given to the Sound constructor.")

        if len(self.samples.shape) == 1:
            self.channels = 1
        else:
            self.channels = self.samples.shape[1]
        self.numpy_encoding = self.samples.dtype


    def __eq__ (self, other):
        """Compares this sound with another one.

        Two sounds are considered equal if they have the same number of channels
        and all of their samples match.

        Parameters:
            other (Sound): The sound to compare this one to.

        Returns:
            equal (bool): True if self and other are equal, false otherwise
        """
        if self.get_channels() == other.get_channels():
            return numpy.all(self.samples == other.samples)
        else:
            return False

    def __str__(self):
        """Return a string representation of this sound."""

        return "Sound of length " + str(len(self))


    def __iter__(self):
        """Return an iterator to allow iterating through the samples in this
        sound."""

        if self.channels == 1:
            for i in range(len(self)):
                yield MonoSample(self.samples, i)

        elif self.channels == 2:
            for i in range(len(self)):
                yield StereoSample(self.samples, i)


    def __len__(self):
        """Return the number of samples in this sound."""

        return len(self.samples)


    def __add__(self, other_sound):
        """Return a new sound that starts with this sound and ends with another
        sound.

        Parameters:
            other_sound (Sound): The sound object that will be the second part
            of the new sound.

        Returns:
            combined (Sound): A sound that begins with the samples in this sound
            and is followed by the samples in the other sound.
        """

        combined = self.copy()
        combined.append(other_sound)
        return combined


    def __mul__(self, num):
        """Return a new sound that is this sound repeated multiple times.

        Parameters:
            num (int): The number of times to repeat this sound in the new sound.

        Returns:
            repeated (Sound): A sound that has the samples of this sound object
            repeated num times.
        """

        repeated = self.copy()
        for _ in range(int(num) - 1):
            repeated.append(self)
        return repeated


    def copy(self):
        """Create a copy of this Sound.

        This copy is "deep" in that modifying the samples in it will not affect
        this sound (and vice versa).

        Returns:
            new_copy (Sound): A deep copy of this sound.
        """

        return Sound(samples=(self.sample_rate, self.samples.copy()))


    def append_silence(self, num_samples):
        """Adds silence to the end of this sound.

        Parameters:
            num_samples (int): Number of (silent) samples added.

        Notes:
            Silence is represented by samples with 0 values for all the channels.
        """

        if self.channels == 1:
            silence_array = numpy.zeros(num_samples, self.numpy_encoding)
        else:
            silence_array = numpy.zeros((num_samples, 2), self.numpy_encoding)

        self.append(Sound(samples=(self.sample_rate, silence_array)))


    def append(self, snd):
        """Appends a sound to the end of this one.

        Parameters:
            snd (Sound): The sound to append. It sound have the same number of
            channels as this sound (i.e. self).

        Raises:
            ValueError: When there is a mismatch in the number of channels in
            self and snd.
        """

        self.insert(snd, len(self))


    def insert(self, snd, i):
        """Inserts a sound into this one.

        Parameters:
            snd (Sound): The sound to insert. It sound have the same number of
            channels as this sound (i.e. self).
            i (int): The index in this sound where we will insert the other
            sound.

        Raises:
            ValueError: When there is a mismatch in the number of channels in
            self and snd.
        """

        if self.get_channels() != snd.get_channels():
            raise ValueError("Mismatch in number of channels.")
        else:
            first_chunk = self.samples[:i]
            second_chunk = self.samples[i:]
            new_samples = numpy.concatenate((first_chunk,
                                             snd.samples,
                                             second_chunk))
            self.samples = new_samples


    def crop(self, remove_before, remove_after):
        """Crops this sound.

        All samples before and after the specified indices are removed.

        Parameters:
            remove_before (int): Index before which all samples will be removed.
            May be negative.
            remove_after (int): Index after which all samples will be removed.
            May be negative.

        Raises:
            IndexError: If remove_before or remove_after are out of range.
        """

        if remove_before >= len(self) or remove_before < -len(self):
            raise IndexError("remove_before out of range:", remove_before)
        elif remove_after >= len(self) or remove_after < -len(self):
            raise IndexError("remove_after out of range:", remove_after)

        remove_before = remove_before % len(self)
        remove_after = remove_after % len(self)
        self.samples = self.samples[remove_before:remove_after + 1]


    def normalize(self):
        """Performs peak normalization on this sound.

        Notes:
            Peak normalization finds the maximum sample value and scales all
            samples so that this maximum sample value is now the maximum
            allowable sample value (e.g. 32767 for 16-bit samples).
        """

        maximum = self.samples.max()
        minimum = self.samples.min()
        factor = min(32767.0/maximum, 32767.0/abs(minimum))
        numpy.multiply(self.samples, factor, out=self.samples, casting='unsafe')


    def play(self, start_index=0, end_index=-1):
        """Plays the sound.

        Parameters:
            start_index (int, optional): The sample index where to start playing.
            end_index (int, optional): The sample index where to stop playing.

        Raises:
            IndexError: If start_index or end_index are out of range.
        """

        player = self.copy()
        player.crop(start_index, end_index)
        sounddevice.play(player.samples)


    def stop(self):
        """Stop playing of this (and all other) sound."""
        sounddevice.stop()


    def get_sampling_rate(self):
        """Return the number of samples per second for this sound."""

        return self.sample_rate


    def get_sample(self, i):
        """Return a specific sample in this sound.

        Parameters:
            i (int): The index of the desired sample. This may be negative.

        Raises:
            IndexError: When index is out of range.

        Returns:
            sample (MonoSample or StereoSample): The requested sample.
        """

        if i >= len(self) or i < -len(self):
            raise IndexError("i out of range:", i)

        if self.channels == 1:
            return MonoSample(self.samples, i)
        elif self.channels == 2:
            return StereoSample(self.samples, i)


    def get_max(self):
        """Return this sound's highest sample value.

        If this Sound is stereo return the absolute highest for both channels.
        """
        return self.samples.max()


    def get_min(self):
        """Return this sound's lowest sample value.
        
        If this sound is stereo return the absolute lowest for both channels.
        """
        return self.samples.min()


    def get_channels(self):
        """Return the number of channels in this sound."""
        return self.channels


    def set_filename(self, filename=None):
        """Associate filename with this sound.

        If the filename is not given, then it is set to the empty string.

        Parameters:
            filename (str, optional): The name of the file, as a path. This
            should end with the extension ".wav" or ".WAV".

        Raises:
            ValueError: When the filename does not have a ".wav" or ".WAV"
            extension.
            OSError: When filename includes a path to a directory that doesn't
            currently exist.
        """

        if filename is not None:
            # First check that any path that might have been given is a valid
            # directory.
            head, tail = os.path.split(filename)
            if head != "" and not os.path.isdir(head):
                raise OSError(head, "does not exist.")

            # Next, check that we have a valid filename extension.
            file_extension = os.path.splitext(tail)[-1]
            if file_extension not in ['.wav', '.WAV']:
                raise ValueError("Filename must end in .wav or .WAV")

            self.filename = filename
        else:
            self.filename = ''


    def get_filename(self):
        """Return the filename associated with this sound.

        Returns:
            filename (str): The name of the associated file, as a path.
        """
        return self.filename


    def save_as(self, filename):
        """Save this sound to a specific file and set its filename.

        Parameters:
            filename (str): The name/path of the file. This should have either a
            '.wav' or '.WAV' extension.

        Raises:
            ValueError: When the filename does not have a ".wav" or ".WAV"
            extension.
            OSError: When filename includes a path to a directory that doesn't
            currently exist.
        """

        self.set_filename(filename)
        scipy.io.wavfile.write(self.filename, self.sample_rate, self.samples)

    def save(self):
        """Save this sound to a file, specifically to it's set file name.

        Raises:
            ValueError: When no file name was set for this sound.
        """

        if self.filename == "":
            raise ValueError("No filename set for this sound.")

        scipy.io.wavfile.write(self.filename, self.sample_rate, self.samples)


class Note(Sound):
    """A class that represents a musical note in the C scale.

    Notes are considered sounds: you can do anything with them that you can do
    with sounds, including combining them with other sounds.
    """

    # The frequency of notes of the C scale, in Hz
    frequencies = {'C' : 261.63,
                   'D' : 293.66,
                   'E' : 329.63,
                   'F' : 349.23,
                   'G' : 392.0,
                   'A' : 440.0,
                   'B' : 493.88}

    default_amp = 5000  # The default amplitude of a note.

    def __init__(self, note, note_length, octave=0):
        """Create a new note of a specific length and octave.

        Parameters:
            note (str): The name of the note: may be one of the following
            values: 'C', 'D', 'E', 'F', 'G', 'A', and 'B'
            note_length (int): The duration (in number of samples) of the note.
            octave (int, optional): The octave, relative to the 4th octave (e.g.
            1 will be the 5th octave, while -2 would be the 2nd octave).

        Raises:
            ValueError: When the note is not valid (e.g. "Q")

        Notes:
            Consecutive octaves of the same note (e.g. 'C') have frequencies
            that differ by a factor of 2. For example, the 4th octave of A
            is 440Hz while the 5th octave is 880Hz.
        """

        note = note.upper() # allow lower case by first changing them to upper

        if note not in self.frequencies:
            raise ValueError("Invalid note:", note)

        self.sample_rate = 44100

        freq = self.frequencies[note] * (2.0 ** octave)

        self.samples = create_sine_wave(int(freq), self.default_amp,
                note_length)

        self.set_filename(None)

        self.channels = self.samples.shape[1]
        self.numpy_encoding = self.samples.dtype


"""
Helper Functions
"""

def create_sine_wave(frequency, amplitude, duration):
    """ Creates an array with a sine wave of a specified frequency, amplitude,
    and duration.

    Parameters:
        frequency (float): The frequency of the sine wave.
        amplitude (int): The maximum amplitude of the sine wave.
        duration (int): The duration (in number of samples) of the sine wave.

    Returns:
        samples (numpy.array of numpy.int16): Array of samples with values
        modeling a sine wave of the given frequency and amplitude.
    """

    # Default frequency is in samples per second
    samples_per_second = 44100.0

    # Hz are periods per second
    seconds_per_period = 1.0 / frequency
    samples_per_period = samples_per_second * seconds_per_period

    samples = numpy.array([range(duration), range(duration)], numpy.float)
    samples = samples.transpose()

    # For each value in the array multiply it by 2*Pi, divide by the
    # samples per period, take the sin, and multiply the resulting
    # value by the amplitude.
    samples = numpy.sin((samples * 2.0 * math.pi) / samples_per_period) * amplitude
    envelope(samples, 2)

    # Convert the array back into one with the appropriate encoding
    samples = numpy.array(samples, numpy.dtype('int16'))
    return samples


def envelope(samples, channels):
    """Add an envelope to samples to prevent clicking.

    Parameters:
        samples (numpy.array): The samples to envelope.
        channels (int): The number of channels in each sample.
    """

    attack = 800
    if len(samples) < 3 * attack:
        attack = int(len(samples) * 0.05)

    line1 = numpy.linspace(0, 1, attack * channels)
    line2 = numpy.ones(len(samples) * channels - 2 * attack * channels)
    line3 = numpy.linspace(1, 0, attack * channels)
    enveloped = numpy.concatenate((line1, line2, line3))

    if channels == 2:
        enveloped.shape = (len(enveloped) // 2, 2)

    samples *= enveloped


"""
Global Sound Functions
"""

def load_sound(filename):
    """Return the Sound at file filename. Requires: file is an uncompressed
    .wav file."""

    return Sound(filename=filename)


def create_silent_sound(num_samples):
    """Return a silent Sound num_samples samples long."""

    arr = [[0, 0] for i in range(num_samples)]
    npa = numpy.array(arr, dtype=numpy.dtype('int16'))

    return Sound(samples=(44100, npa))


def get_samples(snd):
    """Return a list of Samples in Sound snd."""

    return [samp for samp in snd]


def get_max_sample(snd):
    """Return Sound snd's highest sample value. If snd is stereo
    return the absolute highest for both channels."""

    return snd.get_max()


def get_min_sample(snd):
    """Return Sound snd's lowest sample value. If snd is stereo
    return the absolute lowest for both channels."""

    return snd.get_min()


def concatenate(snd1, snd2):
    """Return a new Sound object with Sound snd1 followed by Sound snd2."""

    return snd1 + snd2


def append_silence(snd, samp):
    """Append samp samples of silence onto Sound snd."""

    snd.append_silence(samp)


def append(snd1, snd2):
    """Append snd2 to snd1."""

    snd1.append(snd2)


def crop_sound(snd, first, last):
    """Crop snd Sound so that all Samples before int first and
    after int last are removed. Cannot crop to a single sample.
    Negative indices are supported."""

    snd.crop(first, last)


def insert(snd1, snd2, i):
    """Insert Sound snd2 in Sound snd1 at index i."""

    snd1.insert(snd2, i)


def play(snd):
    """Play Sound snd from beginning to end."""

    snd.play()


def play_in_range(snd, first, last):
    """Play Sound snd from index first to last."""

    snd.play(first, last)


def save_as(snd, filename):
    """Save sound snd to filename."""

    snd.save_as(filename)


def stop():
    """Stop playing Sound snd."""

    sounddevice.stop()


def wait_until_played():
    """Waits until all sounds are done playing."""

    sounddevice.wait()


def get_sampling_rate(snd):
    """Return the Sound snd's sampling rate."""

    return snd.get_sampling_rate()


def get_sample(snd, i):
    """Return Sound snd's Sample object at index i."""

    return snd.get_sample(i)


"""
Global Sample Functions
"""

def get_index(samp):
    """Return Sample samp's index."""

    return samp.get_index()


def set_value(mono_samp, value):
    """Set MonoSample mono_samp's value to value."""

    mono_samp.set_value(value)


def get_value(mono_samp):
    """Return MonoSample mono_samp's value."""

    return mono_samp.get_value()


def set_values(stereo_samp, left, right):
    """Set StereoSample stereo_samp's left value to left and
    right value to right."""

    stereo_samp.set_values(left, right)


def get_values(stereo_samp):
    """Return StereoSample stereo_samp's values in a tuple, (left, right)."""

    return stereo_samp.get_values()


def set_left(stereo_samp, value):
    """Set StereoSample stereo_samp's left value to value."""

    stereo_samp.set_left(value)


def get_left(stereo_samp):
    """Return StereoSample stereo_samp's left value."""

    return stereo_samp.get_left()


def set_right(stereo_samp, value):
    """Set StereoSample stereo_samp's right value to value."""

    stereo_samp.set_right(value)


def get_right(stereo_samp):
    """Return StereoSample stereo_samp's right value."""

    return stereo_samp.get_right()

def copy(obj):
    """Return a deep copy of sound obj."""

    return obj.copy()

