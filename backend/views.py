from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from rest_framework.parsers import JSONParser
from pathlib import Path
import dlib
from contextlib import contextmanager
from keras.utils.data_utils import get_file
from backend.wide_resnet import WideResNet

pretrained_model = "https://github.com/grinat/age_gender_detector/tree/master/models/weights.28-3.73.hdf5"
modhash = 'fbe63257a054c1c5466cfd7bf14646d6'


import base64
import cv2
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO
import numpy as np

def readb64(uri):
    encoded_data = uri.split(',')[1]
    nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is not None:
        h, w, _ = img.shape
        r = 640 / max(w, h)
        return cv2.resize(img, (int(w * r), int(h * r)))


def draw_label(image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
               font_scale=0.8, thickness=1):
    size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    x, y = point
    cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
    cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness, lineType=cv2.LINE_AA)


@contextmanager
def video_capture(*args, **kwargs):
    cap = cv2.VideoCapture(*args, **kwargs)
    try:
        yield cap
    finally:
        cap.release()


def yield_images():
    # capture video
    with video_capture(0) as cap:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        while True:
            # get video frame
            ret, img = cap.read()

            if not ret:
                raise RuntimeError("Failed to capture image")

            yield img




def main(b64string):
    weight_file = get_file("weights.28-3.73.hdf5", pretrained_model, cache_subdir="pretrained_models",
                           file_hash=modhash, cache_dir=str(Path(__file__).resolve().parent))

    # for face detection
    depth = 16
    k = 8
    # load model and weights
    img_size = 64
    model = WideResNet(img_size, depth=depth, k=k)()
    model.load_weights(weight_file)
    image_dir = b64string
    margin = 0.4

    # for face detection
    detector = dlib.get_frontal_face_detector()

    # load model and weights
    image_generator = readb64(image_dir)

    img=image_generator
    input_img = img
    img_h, img_w, _ = np.shape(input_img)

    # detect faces using dlib detector
    detected = detector(input_img, 1)
    faces = np.empty((len(detected), img_size, img_size, 3))

    if len(detected) > 0:
        for i, d in enumerate(detected):
            x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
            xw1 = max(int(x1 - margin * w), 0)
            yw1 = max(int(y1 - margin * h), 0)
            xw2 = min(int(x2 + margin * w), img_w - 1)
            yw2 = min(int(y2 + margin * h), img_h - 1)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            # cv2.rectangle(img, (xw1, yw1), (xw2, yw2), (255, 0, 0), 2)
            faces[i, :, :, :] = cv2.resize(img[yw1:yw2 + 1, xw1:xw2 + 1, :], (img_size, img_size))

        # predict ages and genders of the detected faces
        results = model.predict(faces)
        predicted_genders = results[0]
        ages = np.arange(0, 101).reshape(101, 1)
        predicted_ages = results[1].dot(ages).flatten()

        # draw results
        for i, d in enumerate(detected):
            label = "{}, {}".format(int(predicted_ages[i]),
                                    "M" if predicted_genders[i][0] < 0.5 else "F")
            draw_label(img, (d.left(), d.top()), label)

            string = base64.b64encode(cv2.imencode('.jpg', img)[1]).decode()
            dict = {
                'img': string
            }

            return dict

            # cv2.imshow("result", img)
            # key = cv2.waitKey(-1) if image_dir else cv2.waitKey(30)
            #
            # if key == 27:  # ESC
            #     break

@csrf_exempt
def guess_age(request):
    """
    List all code snippets, or create a new snippet.
    """
    if request.method == 'GET':
        guess = {}
        return JsonResponse(guess, safe=False)

    elif request.method == 'POST':
        data = JSONParser().parse(request)
        base64Str=data['img']
        return JsonResponse(main(base64Str), status=201)


# if __name__ == '__main__':
#     main()

# Create your views here.

# Create your views here.


def home(request):
    return render(request, 'backend/home.html', {})