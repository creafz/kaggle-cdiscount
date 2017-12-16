import os


BASE_DIR = '..'


def path(*rel_path):
    return os.path.join(BASE_DIR, *rel_path)


INPUT_DIR_PATH = path('input')
PICKLED_DATA_PATH = path('pickled_data')
SAVED_MODELS_PATH = path('saved_models')
PREDICTIONS_PATH = path('predictions')
SUBMISSIONS_PATH = path('submissions')
SAMPLE_SUBMISSION_PATH = path('input', 'sample_submission.csv')
PROD_TO_CATEGORY_CSV_PATH = path('input', 'prod_to_category.csv')
TRAIN_BSON_FILE = path('input', 'train.bson')
TEST_BSON_FILE = path('input', 'test.bson')
TRAIN_VALID_DATA_FILENAME = 'train_valid_data.pkl'
TEST_DATA_FILENAME = 'test_data.pkl'
KNOWN_CATEGORIES_FOR_TEST_PRODUCTS_FILENAME = (
    'known_categories_for_test_products.pkl'
)


CRAYON_SERVER_HOSTNAME = 'http://127.0.0.1'
CUDNN_BENCHMARK = True
NUM_WORKERS = 4
SEED = 42
SHUFFLE_SEED = 222
NUM_CLASSES = 5270
SAVE_EACH_ITERATIONS = 15000
ORIGINAL_IMG_SIZE = (180, 180)
IMG_MEAN = [0.782614, 0.766876, 0.755642]
IMG_STD = [0.312855, 0.321510, 0.330830]
