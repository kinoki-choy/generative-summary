import os
import openai

from urllib.request import urlretrieve
from gphotospy import authorize
from gphotospy.media import Media, MEDIAFILTER, date, date_range
from imageai.Classification import ImageClassification

from local_settings import OPENAI_API_KEY

def init():
    credentials = '.credentials.json'
    service = authorize.init(credentials)
    return Media(service)

def search(dateRange):
    return MEDIA_MANAGER.search(
        filter=[
            MEDIAFILTER.PHOTO,
            # MEDIAFILTER.VIDEO,
            dateRange
        ]
    )

def download(photos):
    media = next(photos)
    try:
        urlretrieve(media['baseUrl'], f'downloads/{media["filename"]}')
        print(f'{media["filename"]} ✓')
        download(photos)
    except StopIteration:
        print('Download completed\n')


def classifyImages():
    prediction = ImageClassification()

    # using ResNet50 model trained model
    prediction.setModelTypeAsResNet50()
    prediction.setModelPath(os.path.join(os.getcwd(), 'models', 'resnet50-19c8e357.pth'))
    prediction.loadModel()

    predictions_list = set()

    print('Classifying images')

    for image in os.listdir('downloads'):
        predictions, probabilities = prediction.classifyImage(os.path.join(os.getcwd(), 'downloads', image))
        print(f'{image}: {predictions[0]}, {int(probabilities[0])}%')
        predictions_list.add(predictions[0])

    return list(predictions_list)



if __name__ == '__main__':
    MEDIA_MANAGER = init()


    dateRange = date_range(
        date(2020, 1, 19),
        date(2020, 1, 20)
    )

    try:
        print(f'{len(list(search(dateRange)))} items in this search')
        download(search(dateRange))
        classification_predictions = classifyImages()
    except TypeError:
        print('No results from this search')
