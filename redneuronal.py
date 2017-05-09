import cv2
import numpy as np
import neurolab as nl
import glob


def lector(dir):
    cv_img = []
    total = 0
    for img in glob.glob(dir+"/*.jpg"):
        n = cv2.imread(img)
        #n = cv2.medianBlur(n,5)
        cv_img.append(n)
        total += 1
    return [cv_img,total]


m = lector("1. Maduro")      #10
b = lector("2. Bueno")       #10
p = lector("3. Pasado")      #4
mp = lector("4. Muy pasado") #7
po = lector("5. Podrido")    #8


listimg = []
listimg.extend(m[0])
listimg.extend(b[0])
listimg.extend(p[0])
listimg.extend(mp[0])
listimg.extend(po[0])

nm = m[1]
nb = b[1]
npa = p[1]
nmp = mp[1]
npo = po[1]


print len(listimg)

def creartarget(m,b,p,mp,po):
    salida = []
    for i in range(m):
        salida.append([1,0,0,0,0])
    for i in range(b):
        salida.append([0,1,0,0,0])
    for i in range(p):
        salida.append([0,0,1,0,0])
    for i in range(mp):
        salida.append([0,0,0,1,0])
    for i in range(po):
        salida.append([0,0,0,0,1])

    return np.array( salida )

target = creartarget(nm,nb,npa,nmp,npo)


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
            # draw the outer circle
            cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
            # draw the center of the circle
            #cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
            #print (i[0],i[1], i[2])
            crop_cimg = cimg[(i[1]-i[2]*0.72):(i[1]+i[2]*0.72), (i[0]-i[2]*0.72):(i[0]+i[2]*0.72)]

        final = cv2.resize(crop_cimg,(30, 30), interpolation = cv2.INTER_CUBIC)
        return final
    else:
        print "No se ha encontrado ningun limon"
        return [0]

def colores(linea):
    lista = [0,0,0,0,0]
    for each in linea:
        if( (each[2] >= 20 and each[2] <= 50) and (each[1] >= 40 and each[1] <= 70) and each[0] <= 40 ):
            lista[0] = lista[0] + 1
        elif( (each[2] >= 40 and each[2] <= 80) and (each[1] >= 70 and each[1] <= 120) ):
            lista[1] = lista[1] + 1
        elif( (each[2] >= 80 and each[2] <= 120) and (each[1] >= 120 and each[1] <= 150) ):
            lista[2] = lista[2] + 1
        elif( (each[2] >= 120 and each[2] <= 200) and (each[1] >= 140 and each[1] <= 200) ):
            lista[3] = lista[3] + 1
        elif( (each[2] >= 0 and each[2] <= 20) and (each[1] >= 0 and each[1] <= 40) and each[0] >= 40 ):
            lista[4] = lista[4] + 1
        #else:
        #    lista[5] = lista[5] + 1
    return np.array(lista)


#Cortadaro para cada una de las imagenes
listfinal = []
for i in listimg:
    listaux = []
    aux = cortadora(i)
    #Comprobar que ha encontrado un limon
    if len(aux) != 1:
        for j in aux:
            for k in j:
                listaux.append(k)
        listfinal.append(listaux)

entry = np.array( listfinal )

final = []
for i in entry:
    aux = colores(i)
    final.append(aux)

final = np.array(final)
print "final"
print final


net = nl.net.newff( [[-10, 10]]*5, [19, 5] )

err = net.train(final, target, show= 300, epochs = 50000)

net.save("redfinal.net")


#
