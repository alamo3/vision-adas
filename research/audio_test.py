import simpleaudio as sa
import os

lane_change_sound = sa.WaveObject.from_wave_file(os.path.dirname(__file__)+'\LCAudio.wav')


def test_sound():
    play_sound = lane_change_sound.play()
    play_sound.wait_done()


if __name__ == "__main__":
    test_sound()
    test_sound()