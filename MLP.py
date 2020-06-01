#Perceptron multicapa
# para cálculo de función AND, !AND, OR, !OR, XOR y !XOR de n entradas
import math
import random
import string
import os

# Crea una matriz para almacenar los pesos
def matriz(x,y):
    m = []
    for i in range (x):
        m.append([0.0]*y)
    return m

# Función de tipo sigmoide
def sigmoide(x):
    return math.tanh(x)

#Derivada de función de tipo sigmiode
def dsigmoide(x):
    return 1.0 - x**2

#Inicialización
def iniciar_perceptron():
    global nodos_ent, nodos_ocu, nodos_sal, pesos_ent, pesos_sal
    global act_ent, act_ocu, act_sal
    random.seed(0)

    # Sumamos uno para umbral en nodos entrada
    nodos_ent = nodos_ent + 1

    # Activación de los nodos
    act_ent = [1.0]*nodos_ent
    act_ocu = [1.0]*nodos_ocu
    act_sal = [1.0]*nodos_sal

    # Crear matrices de pesos
    pesos_ent = matriz (nodos_ent, nodos_ocu)
    pesos_sal = matriz (nodos_ocu, nodos_sal)
    # Inicializar pesos a valores aleatorios
    for i in range(nodos_ent):
        for j in range(nodos_ocu):
            pesos_ent[i][j] = random.uniform(-0.5, 0.5)
    for j in range(nodos_ocu):
        for k in range(nodos_sal):
            pesos_sal[j][k] = random.uniform(-0.5, 0.5)

# Actualizar valor de los nodos
def actualizar_nodos(entradas):
    global nodos_ent, nodos_ocu, nodos_sal, pesos_ent, pesos_sal
    global act_ent, act_ocu, act_sal
    if len(entradas) != nodos_ent-1:
        raise ValueError('Numero de Nodos de entrada incorrectos')

    # Activación en nodos de entrada
    for i in range(nodos_ent-1):
        act_ent[i] = entradas[i]

    # activación de nodos ocultos
    for j in range(nodos_ocu):
        sum = 0.0
        for i in range(nodos_ent):
            sum = sum + act_ent[i] * pesos_ent[i][j]
        act_ocu[j] = sigmoide(sum)

    # activación en nodos de salida
    for k in range(nodos_sal):
        sum = 0.0
        for j in range(nodos_ocu):
            sum = sum + act_ocu[j] * pesos_sal[j][k]
        act_sal[k] = sigmoide(sum)

    return act_sal[:]

#retropropagación de errores
def retropropagacion(objetivo, l):
    global nodos_ent, nodos_ocu, nodos_sal, pesos_ent, pesos_sal
    global act_ent, act_ocu, act_sal
    if len(objetivo) != nodos_sal:
        raise ValueError('Numero de objetivos incorrectos')

    # error en nodos de salida
    delta_salida = [0.0] * nodos_sal
    for k in range(nodos_sal):
        error = objetivo[k]-act_sal[k]
        delta_salida[k] = dsigmoide(act_sal[k]) * error

    # error en nodos ocultos
    delta_oculto = [0.0] * nodos_ocu
    for j in range(nodos_ocu):
        error = 0.0
        for k in range(nodos_sal):
            error = error + delta_salida[k]*pesos_sal[j][k]
        delta_oculto[j] = dsigmoide(act_ocu[j]) * error

    # actualizar pesos de nodos de salida
    for j in range(nodos_ocu):
       for k in range(nodos_sal):
           cambio = delta_salida[k]*act_ocu[j]
           pesos_sal[j][k] = pesos_sal[j][k] + l*cambio

    # actualizar pesos de nodos de entrada
    for i in range(nodos_ent):
        for j in range(nodos_ocu):
            cambio = delta_oculto[j]*act_ent[i]
            pesos_ent[i][j] =pesos_ent[i][j] + l*cambio

    # calcular error
    error = 0.0
    for k in range (len(objetivo)):
        error = error + 0.5*(objetivo[k]-act_sal[k])**2
    return error

#clasificar patron
def clasificar(patron):
    for p in patron:
        print(p[0],'->',actualizar_nodos(p[0]))

#clasificar N entadas
def clasificaN(patron):
    eval = patron[0][0]
    tamano = len(eval)
    resultFinal = 0

    while tamano > 1:
        newEval = []
        newEval.append(eval[0])
        newEval.append(eval[1])
        result = actualizar_nodos(newEval)
        if result[0] > 0.4 :
            resultFinal = 1
        else:
            resultFinal = 0
        eval.pop(0)
        eval.pop(0)
        eval.append(resultFinal)
        tamano = len(eval)

    return resultFinal

# entrenamiento del perceptrón
def entrenar_perceptron(patron, l, max_iter=1000):
    for i in range(max_iter):
        error = 0.0
        for p in patron:
            entradas = p [0]
            objetivo = p [1]
            actualizar_nodos(entradas)
            error = error + retropropagacion(objetivo,l)
        # Salir si alcanzamos el limite inferior de error deseado
        if error < 0.001:
            print("\n\n El error es de: ",error)
            break

#Datos de entrada para AND
datos_ent_AND = [[[0,0], [0]],
                 [[0,1], [0]],
                 [[1,0], [0]],
                 [[1,1], [1]]]

#Datos de entrada para or
datos_ent_OR = [[[0,0], [0]],
                [[0,1], [1]],
                [[1,0], [1]],
                [[1,1], [1]]]

#Datos de entrada para XOR
datos_ent_XOR = [[[0,0], [0]],
                 [[0,1], [1]],
                 [[1,0], [1]],
                 [[1,1], [0]]]

nodos_ent=2 # dos neuronas de entrada
nodos_ocu=2 # dos neuronas ocultas
nodos_sal=1 # una neurona de salida
l=0.5

def limpiaPantalla():
    os.system('cls||clear')

def negarResultado(num):
    if num == 1:
        return 0
    else: return 1

def entrenarMLP(datos):
    limpiaPantalla()
    print("\n Entrenando nuestra MLP con los siguientes datos de entrada")
    for d in datos:
        print(' ', d[0], '->', d[1])
    entrenar_perceptron(datos, l)

def usarDatos(datos , estado):
    entrenarMLP(datos)
    print(""" 
    \n El error es muy pequeño, estamos listos para comenzar 
    \n Ahora podras evaluar la entrada que desees,
    \n Ten encuenta que la forma correcta de escribir la entrada es la siguiente:
    \n  * Solo podras escribir 0 y 1 
    \n  * Seperar las entradas por ',' 
    \n  como el siguiente ejemplo: 0 , 0 , 1
    \n\n 1 Escribir entrada
    \n 2. Salir
          """)
    resp = True
    while resp:
        resp = input("Selecciona una opción válida: ")
        if resp == "1":
            entada = input("Ingrese la entrada que desee:")
            objetoEntrada = []
            for val in entada.split(','):
                objetoEntrada.append(int(val))
            print("El resultado es el siguiente")
            if len(entada.split(',')) > 1 :
                if estado == 1:
                    print(entada.split(','),'->',clasificaN([[objetoEntrada]]))
                else:
                    print(entada.split(','),'->',[negarResultado(clasificaN([[objetoEntrada]]))])
            else :
                print("Ingresa una entrada válida")
            print("""
                   1. Desea escribir otra entrada
                   2. Salir
                   """)
        elif resp=="2":
            limpiaPantalla()
            break
        else:
            print("\n No ingresaste una opción valida")
            resp = True

iniciar_perceptron()

ans=True
while ans:
    print ("""
    Multi Layer Perceptron
    Resolviendo compuertas lógicas
    
    1. AND
    2. AND negada
    3. OR
    4. OR negada
    5. XOR 
    6. XOR negada
    7. Salir
    """)
    ans = input("Selecciona que puerta lógica te gustaría evaluar: ")
    if ans=="1":
        usarDatos(datos_ent_AND , 1)
    elif ans=="2":
        usarDatos(datos_ent_AND , -1)
    elif ans=="3":
        usarDatos(datos_ent_OR, 1)
    elif ans=="4":
        usarDatos(datos_ent_OR, -1)
    elif ans=="5":
        usarDatos(datos_ent_XOR, 1)
    elif ans=="6":
        usarDatos(datos_ent_XOR, -1)
    elif ans=="7":
        print("\n Hasta luego.")
        break
    else:
        print("\n No ingresaste una opción valida")
        ans = True