import cv2.dnn as opencv
import wolframclient.serializers as wxf


def caffe2wxf(path):
    model = opencv.readNetFromCaffe(
        path + ".prototxt",
        path + ".caffemodel"
    )
    layers = model.getLayerNames()
    npy = {}
    for i in layers:
        try:
            npy[i] = model.getParam(i)
        except Exception:
            pass
    wxf.export(npy, path + '.wxf', target_format='wxf')
