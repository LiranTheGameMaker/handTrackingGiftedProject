import cv2
import numpy as np
from tensorflow import keras
from tensorflow.keras.applications.mobilenet import preprocess_input

class QuickDrawPredictor():
    def __init__(self, model):
        self.model = model
        self.model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.002), loss='categorical_crossentropy',
                      metrics=[keras.losses.categorical_crossentropy, keras.metrics.categorical_accuracy, "accuracy"])

    # store the label codes in a dictionary
    dict_var = {0: 'airplane', 1: 'alarm_clock', 2: 'ambulance', 3: 'angel', 4: 'animal_migration', 5: 'ant',
                6: 'anvil',
                7: 'apple', 8: 'arm', 9: 'asparagus', 10: 'axe', 11: 'backpack', 12: 'banana', 13: 'bandage',
                14: 'barn',
                15: 'baseball', 16: 'baseball_bat', 17: 'basket', 18: 'basketball', 19: 'bat', 20: 'bathtub',
                21: 'beach',
                22: 'bear', 23: 'beard', 24: 'bed', 25: 'bee', 26: 'belt', 27: 'bench', 28: 'bicycle', 29: 'binoculars',
                30: 'bird', 31: 'birthday_cake', 32: 'blackberry', 33: 'blueberry', 34: 'book', 35: 'boomerang',
                36: 'bottlecap', 37: 'bowtie', 38: 'bracelet', 39: 'brain', 40: 'bread', 41: 'bridge', 42: 'broccoli',
                43: 'broom', 44: 'bucket', 45: 'bulldozer', 46: 'bus', 47: 'bush', 48: 'butterfly', 49: 'cactus',
                50: 'cake', 51: 'calculator', 52: 'calendar', 53: 'camel', 54: 'camera', 55: 'camouflage',
                56: 'campfire',
                57: 'candle', 58: 'cannon', 59: 'canoe', 60: 'car', 61: 'carrot', 62: 'castle', 63: 'cat',
                64: 'ceiling_fan', 65: 'cell_phone', 66: 'cello', 67: 'chair', 68: 'chandelier', 69: 'church',
                70: 'circle',
                71: 'clarinet', 72: 'clock', 73: 'cloud', 74: 'coffee_cup', 75: 'compass', 76: 'computer', 77: 'cookie',
                78: 'cooler', 79: 'couch', 80: 'cow', 81: 'crab', 82: 'crayon', 83: 'crocodile', 84: 'crown',
                85: 'cruise_ship', 86: 'cup', 87: 'diamond', 88: 'dishwasher', 89: 'diving_board', 90: 'dog',
                91: 'dolphin',
                92: 'donut', 93: 'door', 94: 'dragon', 95: 'dresser', 96: 'drill', 97: 'drums', 98: 'duck',
                99: 'dumbbell',
                100: 'ear', 101: 'elbow', 102: 'elephant', 103: 'envelope', 104: 'eraser', 105: 'eye',
                106: 'eyeglasses',
                107: 'face', 108: 'fan', 109: 'feather', 110: 'fence', 111: 'finger', 112: 'fire_hydrant',
                113: 'fireplace',
                114: 'firetruck', 115: 'fish', 116: 'flamingo', 117: 'flashlight', 118: 'flip_flops', 119: 'floor_lamp',
                120: 'flower', 121: 'flying_saucer', 122: 'foot', 123: 'fork', 124: 'frog', 125: 'frying_pan',
                126: 'garden', 127: 'garden_hose', 128: 'giraffe', 129: 'goatee', 130: 'golf_club', 131: 'grapes',
                132: 'grass', 133: 'guitar', 134: 'hamburger', 135: 'hammer', 136: 'hand', 137: 'harp', 138: 'hat',
                139: 'headphones', 140: 'hedgehog', 141: 'helicopter', 142: 'helmet', 143: 'hexagon',
                144: 'hockey_puck',
                145: 'hockey_stick', 146: 'horse', 147: 'hospital', 148: 'hot_air_balloon', 149: 'hot_dog',
                150: 'hot_tub',
                151: 'hourglass', 152: 'house', 153: 'house_plant', 154: 'hurricane', 155: 'ice_cream', 156: 'jacket',
                157: 'jail', 158: 'kangaroo', 159: 'key', 160: 'keyboard', 161: 'knee', 162: 'ladder', 163: 'lantern',
                164: 'laptop', 165: 'leaf', 166: 'leg', 167: 'light_bulb', 168: 'lighthouse', 169: 'lightning',
                170: 'line',
                171: 'lion', 172: 'lipstick', 173: 'lobster', 174: 'lollipop', 175: 'mailbox', 176: 'map',
                177: 'marker',
                178: 'matches', 179: 'megaphone', 180: 'mermaid', 181: 'microphone', 182: 'microwave', 183: 'monkey',
                184: 'moon', 185: 'mosquito', 186: 'motorbike', 187: 'mountain', 188: 'mouse', 189: 'moustache',
                190: 'mouth', 191: 'mug', 192: 'mushroom', 193: 'nail', 194: 'necklace', 195: 'nose', 196: 'ocean',
                197: 'octagon', 198: 'octopus', 199: 'onion', 200: 'oven', 201: 'owl', 202: 'paint_can',
                203: 'paintbrush',
                204: 'palm_tree', 205: 'panda', 206: 'pants', 207: 'paper_clip', 208: 'parachute', 209: 'parrot',
                210: 'passport', 211: 'peanut', 212: 'pear', 213: 'peas', 214: 'pencil', 215: 'penguin', 216: 'piano',
                217: 'pickup_truck', 218: 'picture_frame', 219: 'pig', 220: 'pillow', 221: 'pineapple', 222: 'pizza',
                223: 'pliers', 224: 'police_car', 225: 'pond', 226: 'pool', 227: 'popsicle', 228: 'postcard',
                229: 'potato',
                230: 'power_outlet', 231: 'purse', 232: 'rabbit', 233: 'raccoon', 234: 'radio', 235: 'rain',
                236: 'rainbow',
                237: 'rake', 238: 'remote_control', 239: 'rhinoceros', 240: 'river', 241: 'roller_coaster',
                242: 'rollerskates', 243: 'sailboat', 244: 'sandwich', 245: 'saw', 246: 'saxophone', 247: 'school_bus',
                248: 'scissors', 249: 'scorpion', 250: 'screwdriver', 251: 'sea_turtle', 252: 'see_saw', 253: 'shark',
                254: 'sheep', 255: 'shoe', 256: 'shorts', 257: 'shovel', 258: 'sink', 259: 'skateboard', 260: 'skull',
                261: 'skyscraper', 262: 'sleeping_bag', 263: 'smiley_face', 264: 'snail', 265: 'snake', 266: 'snorkel',
                267: 'snowflake', 268: 'snowman', 269: 'soccer_ball', 270: 'sock', 271: 'speedboat', 272: 'spider',
                273: 'spoon', 274: 'spreadsheet', 275: 'square', 276: 'squiggle', 277: 'squirrel', 278: 'stairs',
                279: 'star', 280: 'steak', 281: 'stereo', 282: 'stethoscope', 283: 'stitches', 284: 'stop_sign',
                285: 'stove', 286: 'strawberry', 287: 'streetlight', 288: 'string_bean', 289: 'submarine',
                290: 'suitcase',
                291: 'sun', 292: 'swan', 293: 'sweater', 294: 'swing_set', 295: 'sword', 296: 't-shirt', 297: 'table',
                298: 'teapot', 299: 'teddy-bear', 300: 'telephone', 301: 'television', 302: 'tennis_racquet',
                303: 'tent',
                304: 'The_Eiffel_Tower', 305: 'The_Great_Wall_of_China', 306: 'The_Mona_Lisa', 307: 'tiger',
                308: 'toaster',
                309: 'toe', 310: 'toilet', 311: 'tooth', 312: 'toothbrush', 313: 'toothpaste', 314: 'tornado',
                315: 'tractor', 316: 'traffic_light', 317: 'train', 318: 'tree', 319: 'triangle', 320: 'trombone',
                321: 'truck', 322: 'trumpet', 323: 'umbrella', 324: 'underwear', 325: 'van', 326: 'vase', 327: 'violin',
                328: 'washing_machine', 329: 'watermelon', 330: 'waterslide', 331: 'whale', 332: 'wheel',
                333: 'windmill',
                334: 'wine_bottle', 335: 'wine_glass', 336: 'wristwatch', 337: 'yoga', 338: 'zebra', 339: 'zigzag'}
    label_list = list(dict_var.values())

    def predict(self, img_n):
        if type(img_n) == str:
            img = cv2.imread(img_n, cv2.IMREAD_GRAYSCALE)
        elif type(img_n) == np.ndarray:
            img = cv2.cvtColor(img_n, cv2.COLOR_BGR2GRAY)
        x = np.zeros((80, 80))
        x[:, :] = self.resize_keep_aspect_ratio(img, (80, 80))
        x = preprocess_input(x).astype(np.float32)
        x = 1 - x
        x = self.remove_rows_all_ones(x.copy())
        x = self.remove_columns_all_ones(x.copy())
        x = self.pad_right_bottom(x, (80,80), 0)
        if np.any(x > 0):
            np.where(x > 0, 1, -1)
        x = x.reshape(1, 80, 80, 1).astype(np.float32)
        prediction = self.model.predict(x)
        return prediction

    def process(self, prediction):
        return np.argsort(-prediction, axis=1)

    def pad_right_bottom(self, data, target_size=(80, 80), pad_value=0):
        # Calculate the amount of padding needed for each dimension
        height_padding = target_size[0] - data.shape[0]
        width_padding = target_size[1] - data.shape[1]
        # Create the padding tuple with appropriate padding values
        padding_tuple = ((0, height_padding), (0, width_padding))
        # Pad the array using the calculated padding and pad_value
        return np.pad(data, padding_tuple, mode='constant', constant_values=pad_value)

    def remove_rows_all_ones(self, data):
        return data[~np.all(data == 0, axis=1)]

    def remove_columns_all_ones(self, data):
        columns = np.all(data == 0, axis=0)
        return data[:, ~columns]

    def resize_keep_aspect_ratio(self, img, target_shape=(80, 80)):
        # Get height and width
        height, width = img.shape[:2]
        # Determine the smaller dimension for cropping
        crop_size = min(height, width)
        # Calculate top-left corner coordinates for the crop
        top_left_x = (width - crop_size) // 2
        top_left_y = (height - crop_size) // 2
        # Crop the image
        cropped_img = img[top_left_y:top_left_y + crop_size, top_left_x:top_left_x + crop_size]
        return cv2.resize(cropped_img, target_shape, interpolation=cv2.INTER_AREA)
