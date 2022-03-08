CLASSES = { 0:'Speed limit (20km/h)',
            1:'Speed limit (30km/h)', 
            2:'Speed limit (50km/h)', 
            3:'Speed limit (60km/h)', 
            4:'Speed limit (70km/h)', 
            5:'Speed limit (80km/h)', 
            6:'End of speed limit (80km/h)', 
            7:'Speed limit (100km/h)', 
            8:'Speed limit (120km/h)', 
            9:'No passing', 
            10:'No passing veh over 3.5 tons', 
            11:'Right-of-way at intersection', 
            12:'Priority road', 
            13:'Yield', 
            14:'Stop', 
            15:'No vehicles', 
            16:'Veh > 3.5 tons prohibited', 
            17:'No entry', 
            18:'General caution', 
            19:'Dangerous curve left', 
            20:'Dangerous curve right', 
            21:'Double curve', 
            22:'Bumpy road', 
            23:'Slippery road', 
            24:'Road narrows on the right', 
            25:'Road work', 
            26:'Traffic signals', 
            27:'Pedestrians', 
            28:'Children crossing', 
            29:'Bicycles crossing', 
            30:'Beware of ice/snow',
            31:'Wild animals crossing', 
            32:'End speed + passing limits', 
            33:'Turn right ahead', 
            34:'Turn left ahead', 
            35:'Ahead only', 
            36:'Go straight or right', 
            37:'Go straight or left', 
            38:'Keep right', 
            39:'Keep left', 
            40:'Roundabout mandatory', 
            41:'End of no passing', 
            42:'End no passing veh > 3.5 tons' }


from PIL import Image
import numpy as np
from tensorflow import keras

model = keras.models.load_model('./model0.h5')
print(model.summary())


def preprocess(pil_image, h=30, w=30):
    im = pil_image.resize((h, w))
    im = np.array(im)/255
    return np.array([im])

if __name__=='__main__':
    im = Image.open('./00000.png')
    bx = preprocess(im)
    ps = model.predict(bx)

    class_id = model.predict_classes(bx)[0]
    print('class:', class_id, CLASSES[class_id])
    print(ps)
    # class: 16 Veh > 3.5 tons prohibited
    # [[7.6989456e-12 1.5715316e-11 1.3856005e-14 1.2578464e-13 4.6011118e-15
    # 6.3684731e-15 2.5371108e-11 3.2866346e-12 2.2464474e-12 6.5410116e-10
    # 2.4464592e-14 1.0336382e-14 4.8257107e-07 7.9203655e-17 4.1968384e-11
    # 6.2554081e-14 9.9999940e-01 3.1131485e-08 3.9646190e-16 3.6706743e-13
    # 9.5654999e-15 5.4683757e-15 2.0528405e-11 3.4883792e-14 5.6658393e-14
    # 8.4538022e-15 2.7845751e-18 1.6231902e-15 1.6568963e-13 3.8507026e-13
    # 1.1341652e-13 3.3779121e-15 2.6625894e-11 2.8649372e-12 3.9697880e-12
    # 6.7827254e-14 2.5915995e-12 7.3398343e-11 1.2008908e-10 6.2170935e-11
    # 6.2660078e-08 5.8956187e-11 9.2974001e-13]]