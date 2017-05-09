import cv2
import numpy as np
import neurolab as nl

np.set_printoptions(suppress=True)

def mostrar(img):
    cv2.imshow('detected circles',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def cortadora(img):

    aux = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    cimg = img

    circles = cv2.HoughCircles(aux,cv2.HOUGH_GRADIENT,1,2000, param1=59,param2=50,minRadius=50,maxRadius=10000)
    #cv2.HoughCircles
    circles = np.uint16(np.around(circles))
    if len(circles) != 0:

        for i in circles[0,:]:
            crop_cimg = cimg[(i[1]-i[2]*0.72):(i[1]+i[2]*0.72), (i[0]-i[2]*0.72):(i[0]+i[2]*0.72)]
        final = cv2.resize(crop_cimg,(30, 30), interpolation = cv2.INTER_CUBIC)
        return final
    else:
        print "No se ha encontrado ningun limon"
        return [0]

def colores(linea):
    lista = [0,0,0,0,0]
    for each in linea:
        if( (each[2] >= 20 and each[2] <= 30) and (each[1] >= 40 and each[1] <= 70) ):
            lista[0] = lista[0] + 1
        elif( (each[2] >= 30 and each[2] <= 60) and (each[1] >= 70 and each[1] <= 120) ):
            lista[1] = lista[1] + 1
        elif( (each[2] >= 60 and each[2] <= 120) and (each[1] >= 120 and each[1] <= 140) ):
            lista[2] = lista[2] + 1
        elif( (each[2] >= 120 and each[2] <= 200) and (each[1] >= 140 and each[1] <= 200) ):
            lista[3] = lista[3] + 1
        elif( (each[2] >= 0 and each[2] <= 20) and (each[1] >= 0 and each[1] <= 40) ):
            lista[4] = lista[4] + 1
        #else:
        #    lista[5] = lista[5] + 1
    return np.array(lista)

def interprete(resul):
    if resul[0] >= 0.95:
        print "El limon se encuentra en estado maduro"
        print "Espere 5 dias para consumirlo"
    if resul[1] >= 0.95:
        print "El limon se encuentra en estado bueno"
        print "Le qudan 8 dias para consumirlo en estado optimo"
    if resul[2] >= 0.95:
        print "El limon se encuentra en estado pasado"
        print "Le qudan 3 dias para consumirlo antes de que se pase"
    if resul[3] >= 0.95:
        print "El limon se encuentra en estado muy pasado"
        print "Se recomienda ya no consumirlo, aunque si lo hace debe ser inmediatamente"
    if resul[4] >= 0.95:
        print "El limon se encuentra en estado podrido"
        print "No consuma este limon"


im1 = cv2.imread('a5.jpg',1)

out = cortadora(im1)
#mostrar(out)

listaux = []
if len(out) != 1:
    for j in out:
        for k in j:
            listaux.append(k)

listaux = np.array( listaux )
fin = np.array( colores(listaux) )
#print fin

net = nl.load("funcional.net")
resultado = net.sim([fin])
print resultado

interprete(resultado[0])




#
