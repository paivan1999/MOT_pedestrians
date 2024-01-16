from MOT_pedestrians.algorithms.detectors.Detectors import CompositionDetector
from MOT_pedestrians.video_savers.saver import VideoSaver
l = [((8,8),0.0,0.66,1.0,1.3)]
for el in range(1,4):
    print(el)
    if el < 3:
        path = f"data/people_on_street/POS{el}.mov"
    else:
        path = f"data/people_on_street/POS{el}.mp4"
    for (winStride,diff_threshold1,diff_threshold2,confidence_threshold,not_bigger_than_by) in l:
        VideoSaver.load(path)
        print(f"\t\t\tload done")
        VideoSaver.set_detector(type="CompositionDetector",
                                winStride = winStride,
                                diff_threshold1 = diff_threshold1,
                                diff_threshold2 = diff_threshold2,
                                confidence_threshold = confidence_threshold,
                                not_bigger_than_by = not_bigger_than_by
                                )
        print(f"\t\t\tset detector")
        VideoSaver.save(f"examples/{el}"
                        f"_wS({winStride})"
                        f"_lt1({str(diff_threshold1).replace('.',',')})"
                        f"_lt2({str(diff_threshold2).replace('.',',')})"
                        f"_ct({str(confidence_threshold).replace('.',',')})"
                        f"_koeff({str(not_bigger_than_by).replace('.',',')})"
                        f".mp4")
        print(f"\t\t\tsaved video")
