import numpy as np
import cv2
import tensorly as tl
import os
from math import ceil
from tensorly import unfold
from tensorly import fold
from math import sqrt
import matplotlib.pyplot as plt




winSize = (20,20)
blockSize = (4,4)
blockStride = (2,2)
cellSize = (2,2)
nbins = 4
nBIN = 1
derivAperture = 1
winSigma = 4.
histogramNormType = 0
L2HysThreshold = 2.0000000000000001e-01
gammaCorrection = 0
nlevels = 64
hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nBIN,derivAperture,winSigma,
                        histogramNormType,L2HysThreshold,gammaCorrection,nlevels)

def scalar_prod(A, B):
    sum = 0
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            for k in range(A.shape[2]):
                    sum += A[i,j,k] * B[i, j, k]
    return sum

def ten_norm(A):
    return sqrt(scalar_prod(A, A))

CROPED = True
DIMENSION = 20
NEW_DIMENSION = 9
NUM_OF_FRAMES = 127
NUM_OF_TRAIN_FILES = 6
NUM_OF_FILES = 18
DIFF_FILE = 6000
NUM_OF_DIFFERENT_SIGNS = 2
debug = False 
PLOT = False

directory_to_save = "../videos_done"
directory = os.fsencode(directory_to_save)
max_h = 0
max_w = 0
TENZOR =tl.tensor(np.zeros((NEW_DIMENSION,NEW_DIMENSION,NUM_OF_FRAMES*nbins,NUM_OF_FILES)))
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    x = int(filename.split('_')[1])
    y = int(filename.split('_')[2])
    x2 = int(filename.split('_')[3])
    y2 = int(filename.split('_')[4].split('.')[0])
    h=y2-y
    w=x2-x
    if max_h < h:
        max_h = h
    if max_w < w:
        max_w = w
num_of_file = 0
tmp=0
tmp_array = np.zeros(NUM_OF_FILES)
name_array = np.zeros(NUM_OF_FILES)
for file in os.listdir(directory):
    num_of_file += 1
    filename = os.fsdecode(file)
    cap = cv2.VideoCapture(directory_to_save + "/" + filename)

    cnt = 0

    w_frame, h_frame = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps, frames = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)

    x = int(filename.split('_')[1])
    y = int(filename.split('_')[2])
    x2 = int(filename.split('_')[3])
    y2 = int(filename.split('_')[4].split('.')[0])
    h=y2-y
    w=x2-x
    
    word = tl.tensor(np.zeros((NEW_DIMENSION, NEW_DIMENSION, (NUM_OF_FRAMES+1)*nbins)))
    
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret==True:
            frame = cv2.resize(frame, (320,240))
            if CROPED:
                frame = frame[y:y2, x:x2]
            frame = cv2.resize(frame,(DIMENSION,DIMENSION))
            cv2.imshow('Croped Frame',frame)
            
            hist = hog.compute(frame) 
            
            hist = hist.reshape((9,9,nbins))
            if cnt < 127:
                for i in range(nbins):
                    word[:,:,cnt*nbins+i] = hist[:,:,i]
            cnt += 1 # Counting frames

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()
    if cnt == 149:
        cnt =128
    word = word[:,:,0:(cnt-1)*nbins] #ovo sam promjenio mate sjeti me
    cnt = word.shape[2]
    final_t = tl.tensor(np.zeros((NEW_DIMENSION, NEW_DIMENSION, NUM_OF_FRAMES * nbins)))
    max_cnt = final_t.shape[2]
    
    if cnt<max_cnt:
        #print(cnt)
        num_of_dup = ceil(final_t.shape[2]/word.shape[2])
        #print("num_of_dup: ", num_of_dup)
        for i in range(cnt):
            fill = (i+1)*num_of_dup
            spent = i+1
            if max_cnt -  fill == cnt-spent:
                fill = i*num_of_dup
                for m in range(cnt-i):
                    for k in range(max_cnt-fill):
                        final_t[:,:,fill + k] = word[:,:,i+m]
                        break
            if max_cnt -  fill < cnt-spent:
                fill = i*num_of_dup
                for k in range(max_cnt-fill):
                    final_t[:,:,fill + k] = word[:,:,i]
                break

            for j in range(num_of_dup):
                final_t[:,:,i*num_of_dup+j] = word[:,:,i]
    elif cnt == max_cnt:
        final_t=word
    else:
        print("Error: max_cnt is set wrong!")
        exit(1)
    TENZOR[:,:,:,tmp]= final_t
    name = int(filename.split("_")[0])
    name_array[tmp] = name
    tmp_array[tmp] = 1
    if name - DIFF_FILE > 0:
        tmp_array[tmp] = 2
    tmp +=1
h = NEW_DIMENSION
w = NEW_DIMENSION
dim = NUM_OF_FRAMES * NUM_OF_TRAIN_FILES * nbins
test_x = np.zeros(NUM_OF_TRAIN_FILES)
test_y = np.zeros(NUM_OF_TRAIN_FILES)
test_x_counter = 0
test_y_counter = 0
first_word = tl.tensor(np.zeros((h, w, dim)))
second_word = tl.tensor(np.zeros((h, w, dim)))
num_of_ones = 0
num_of_twos = 0 
num_of_frames_in_first= 0
num_of_frames_in_second= 0
for i in range(NUM_OF_FILES):
    print(tmp_array[i])
    if tmp_array[i] == 1 and num_of_ones<NUM_OF_TRAIN_FILES:
        num_of_ones += 1
        test_x[test_x_counter] = i
        test_x_counter += 1
        x, y, z = TENZOR[:,:,:,i].shape
        first_word[:,:,num_of_frames_in_first:(num_of_frames_in_first+z)] = TENZOR[:,:,:,i]
        num_of_frames_in_first += z
    elif tmp_array[i] == 2 and num_of_twos<NUM_OF_TRAIN_FILES:
        test_y[test_y_counter] = i
        test_y_counter += 1
        num_of_twos+=1
        x, y, z = TENZOR[:,:,:,i].shape
        second_word[:,:,num_of_frames_in_second:(num_of_frames_in_second+z)] = TENZOR[:,:,:,i]
        num_of_frames_in_second += z
first_word = first_word[:,:,0:num_of_frames_in_first]
second_word = second_word[:,:,0:num_of_frames_in_second]
number_of_words = 2
tensors = [tl.tensor(np.zeros((h, w, first_word.shape[2]))), tl.tensor(np.zeros((h, w, second_word.shape[2])))]

tensors[1] = first_word
tensors[0] = second_word

compression = NUM_OF_FRAMES*NUM_OF_TRAIN_FILES * nbins
u1 = [np.linalg.svd(unfold(tensors[0], 0))[0]]

u2 = [np.linalg.svd(unfold(tensors[0], 1))[0]]

u3 = [np.linalg.svd(unfold(tensors[0], 2))[0][:,0:compression]]


M = [tl.tenalg.mode_dot(tl.tenalg.mode_dot(tl.tenalg.mode_dot(tensors[0], np.transpose(u1[0]),0), np.transpose(u2[0]), 1), np.transpose(u3[0]), 2)]
S = [tl.tenalg.mode_dot(tl.tenalg.mode_dot(tl.tenalg.mode_dot(tensors[0], np.transpose(u1[0]),0), np.transpose(u2[0]), 1), np.transpose(u3[0]), 2)]

u1.append(np.linalg.svd(unfold(tensors[1], 0))[0])

u2.append(np.linalg.svd(unfold(tensors[1], 1))[0])
u3.append(np.linalg.svd(unfold(tensors[1], 2))[0][:,0:compression])
M.append(tl.tenalg.mode_dot(tl.tenalg.mode_dot(tl.tenalg.mode_dot(tensors[1], np.transpose(u1[1]),0), np.transpose(u2[1]), 1), np.transpose(u3[1]), 2))
S.append(tl.tenalg.mode_dot(tl.tenalg.mode_dot(tl.tenalg.mode_dot(tensors[1], np.transpose(u1[1]),0), np.transpose(u2[1]), 1), np.transpose(u3[1]), 2))

smallest = float("inf")
second_index = -1

if debug:
    tmp_array2 = np.zeros(S[0].shape[2])
    for j in range(NUM_OF_DIFFERENT_SIGNS):
        print("This is ",j,": ")
        for i in range(S[j].shape[2]):
            tmp_array2[i] = np.linalg.norm(S[j][:,:,i])
            print(i,": ",np.linalg.norm(S[j][:,:,i]))
    # Initialize a Figure 
    
    # Add Axes to the Figure
    y = np.arange(S[j].shape[2])
    plt.plot(y, tmp_array2)
    plt.show()
    if PLOT:
        exit()

score = 0
for first in range(NUM_OF_TRAIN_FILES,NUM_OF_TRAIN_FILES+1):
    for second in range(NUM_OF_TRAIN_FILES,NUM_OF_TRAIN_FILES+1):
        compression_for_first = 2540
        compression_for_second = 2540
        S[0] = M[0][:,:,0:compression_for_first]
        S[1] = M[1][:,:,0:compression_for_second]
        if debug:
            for j in range(NUM_OF_DIFFERENT_SIGNS):
                print("This is ",j,": ")
                for i in range(S[j].shape[2]):
                    print(i,": ",np.linalg.norm(S[j][:,:,i]))

        score = 0
        for i in range(NUM_OF_FILES):
            Z = TENZOR[:,:,:,i]
            smallest = float("inf")
            second_smallest = float("inf")
            indeks = -1
            for j in range (NUM_OF_DIFFERENT_SIGNS):

                _,_,num_slices = Z.shape

                tensor_sum  = tl.tensor(np.zeros((NEW_DIMENSION,NEW_DIMENSION,num_slices)))
                koef = 0
                compression = -1
                if j == 0:
                    compression = compression_for_first
                else:
                    compression = compression_for_second
                for l in range(0,int(compression/NUM_OF_FRAMES*nbins)):
                    A_i = tl.tensor(np.zeros((NEW_DIMENSION,NEW_DIMENSION,num_slices*nbins)))
                    A_i = tl.tenalg.mode_dot(tl.tenalg.mode_dot(S[j][:,:,l:l+NUM_OF_FRAMES*nbins], u1[j], 0), u2[j], 1)
                    if debug:
                        print("This is norm(A_i) for ",j ,". word: ", ten_norm(A_i))
                        print("These are scalar products of Z and A_i:", scalar_prod(Z,A_i))
                    if A_i.shape == Z.shape:
                        koef = scalar_prod(Z,A_i)/scalar_prod(A_i, A_i)
                        tensor_sum += koef*A_i
                norma = ten_norm(Z-tensor_sum)
                if norma < smallest: 
                    second_smallest = smallest
                    smallest = norma
                    indeks = j+1
                elif norma < second_smallest:
                    second_smallest = norma

            if tmp_array[i] == indeks:
                if debug:
                    print("The smallest is: ", smallest)
                    print("Second smallest is: ", second_smallest)
                    
                if i in test_x or i in test_y:
                    print("Hit trained")
                else:
                    print("Hit not trained")
                print("Hit sign = ", tmp_array[i], "a file je ", name_array[i])
                score+=1
            elif debug:        
                print("The smallest is: ", smallest)
                print("Second smallest is: ", second_smallest)
                print("Missed")
                print("")



        print(score,"/", NUM_OF_FILES , "compression=", compression_for_first, compression_for_second)  
        exit()

