import os
from MOT_pedestrians.format_checkers.standard_format_checker import VideoFormatChecker
from MOT_pedestrians import ROOT_DIR

test_video_path = os.path.join(ROOT_DIR,"data/videos_for_tests/test.mp4")

def test_check_video_format():
    format_checker = VideoFormatChecker()
    result = format_checker.check_video_format(test_video_path)
    assert result == "mp4"

def test_get_video_resolution():
    format_checker = VideoFormatChecker()
    result = format_checker.get_video_resolution(test_video_path)
    assert result == (960, 400)  # Убедитесь, что width и height правильно определены

def test_get_video_duration():
    format_checker = VideoFormatChecker()
    result = format_checker.get_video_duration(test_video_path)
    assert result == 46.546499999999995