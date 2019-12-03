import numpy
import cv2
import os
import pickle
from train.faces_train import train



opt = """
    1 - Inserir diretório das imagens a serem treinadas no modelo
    2 - Treinar com imagens a partir da WebCam
    Escolha sua opção:"""

option = int(input(opt))

if option == 1:

    dir_ex = """
    Insira o diretório das imagens
    Exemplo: 'C:\\diretorio\das\imagens'
    """

    diretorio = str(input(dir_ex))

    train(diretorio, option)

if option == 2:

    nome_da_pessoa = str(input('Digite seu nome para ser salvo no sistema: '))

    imagem_camera = cv2.VideoCapture(0)

    cascade_face = cv2.CascadeClassifier('../cascades/data/haarcascade_frontalface_alt2.xml')

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    image_dir = os.path.join(BASE_DIR, 'images')

    path = image_dir + '\\' + nome_da_pessoa

    if os.path.isdir(path):
        pass
    else:
        os.mkdir(image_dir + "\\" + nome_da_pessoa)

    contador_imagem = 0

    #Extraindo imagens da webcam e salvando no diretorio images/nome_da_pessoa
    while (True):

        ret, frame = imagem_camera.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = cascade_face.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

        for (x, y, w, h) in faces:

            only_face_frame = frame[y: y + h, x:x + w]

            img_name = '../train/images/' + nome_da_pessoa + '/' + str(contador_imagem) + '_' + nome_da_pessoa + '.jpg'

            cv2.imwrite(img_name, only_face_frame)

            color = (255, 255, 0)

            stroke = 2

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, stroke)

        cv2.imshow('Ola', frame)

        if cv2.waitKey(20) & 0xFF == ord('q') or contador_imagem >=100:
            break

        contador_imagem = contador_imagem + 1

    train('', 1)
    imagem_camera.release()
    cv2.destroyAllWindows()


recognizer = cv2.face.LBPHFaceRecognizer_create()

recognizer.read('trainner.yml')

labels = {'person_name':1}

with open('labels.pickle', 'rb') as f:
    labels = pickle.load(f)
    labels = {v:k for k,v in labels.items()}


dir_predict_img = str(input('Digite o diretório da imagem a ser predita: '))

predict_image = cv2.imread(dir_predict_img)

predict_image = cv2.cvtColor(predict_image, cv2.COLOR_BGR2GRAY)

id_, conf = recognizer.predict(predict_image)

if conf >= 4 and conf <= 85:
    name = labels[id_]
    color = (255, 255, 255)
    stroke = 2
    cv2.putText(predict_image, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, color, stroke, cv2.LINE_AA)
    cv2.rectangle(predict_image, (x, y), (x + w, y + h), color, stroke)
cv2.imshow('Olá', cv2.resize(predict_image, (600, 600)))
cv2.waitKey(10000)

if cv2.waitKey(20) & 0xFF == ord('q'):
    cv2.destroyAllWindows()






