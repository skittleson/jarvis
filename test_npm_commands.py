from nlp_commands import classify_and_extract

def test_set_timer():
    action, data = classify_and_extract("set a timer for 5 minutes")
    assert action == "set_timer"
    assert data == {"duration": 5, "unit": "minutes"}

def test_set_timer_with_task():
    action, data = classify_and_extract("set a timer for 2 hours for cleaning")
    assert action == "set_timer"
    assert data == {"duration": 2, "unit": "hours"}

def test_set_reminder():
    action, data = classify_and_extract("remind me tomorrow about my appointment")
    assert action == "set_reminder"
    assert data == {"when": "tomorrow", "task": "appointment"}

def test_play_song():
    action, data = classify_and_extract("play this song")
    assert action == "play_song"
    assert data == {}

def test_turn_off_light():
    action, data = classify_and_extract("turn off kitchen light")
    assert action == "turn_off_light"
    assert data == {"location": "kitchen"}

def test_turn_on_light():
    action, data = classify_and_extract("turn on porch light")
    assert action == "turn_on_light"
    assert data == {"location": "porch"}

def test_dim_light():
    action, data = classify_and_extract("dim porch light by 50%")
    assert action == "dim_light"
    assert data == {"location": "porch", "percentage": 50, "task": "%"}

def test_open_garage_door():
    action, data = classify_and_extract("open garage door")
    assert action == "open_garage_door"
    assert data == {}

def test_close_garage_door():
    action, data = classify_and_extract("close garage door")
    assert action == "close_garage_door"
    assert data == {}

if __name__ == "__main__":
    import pytest
    pytest.main()