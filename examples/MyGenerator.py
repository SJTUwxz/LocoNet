#coding=utf-8
import os
import sys
import numpy as np
from PIL import Image,ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from keras.preprocessing.image import ImageDataGenerator

class MyGenerator(object) :
    def __init__(self,rescale = 1./255, num_classes=2,  *args, **kwargs):
        self.rescale = rescale

    def get_file_len(self, file):
        return len(open(file, 'r').read().splitlines())

    def label_generator(self,label_file_path,target_size=(299,299),
                        batch_size = 64,shuffle = False,not_break = False,
                        nb_output = 1, limit_number = None, fovea = False):
        '''This is a generator to generate picture\'s array from label file,
        the content of label_file should be like this:
        /home/xxx/xxx/xxx.jpg 1,
        the former part is the path of pic and the latter number is the class number of the pic
        input:
            label_file_path: the path of label file
            target_size: not used now
            batch_size: easy to understand, recommended number is 16 32 or 64
            shuffle: shuffle the list of file, recommend setting to True
            not_break: if you want to have more than one epoch and use fit_generator in keras, you should set it to True
            nb_output: if your model has more than one output, set the number of output
            limit_number: if you just want to use part of the data in label file, you can set this number. When you set limit_number, shuffle should be True
        yield:
            batch_size*(x,y*nb_output)'''
        if limit_number != None: 
            assert shuffle,'When you have limit_number, you must shuffle'
            assert type(limit_number) == int,'limit_number must be int'
        assert os.path.isfile(label_file_path),'label文件错误%s'%(label_file_path)
        label_file=open(label_file_path)
        self.x_path_list = []
        self.y_list = []
        for line in label_file:
            x,y=line.split()
            if not os.path.exists(x):
                continue
            self.x_path_list.append(x)
            self.y_list.append(int(y))
        self.data_num = len(self.y_list)
        if limit_number!=None:
            self.x_path_list,self.y_list = self.shuffle(self.x_path_list,self.y_list)
            self.x_path_list = self.x_path_list[0:limit_number]
            self.y_list = self.y_list[0:limit_number]
            self.data_num = min(self.data_num, limit_number)
        nb_class = len(set(self.y_list))
        print ("%d pic in label file, %d classes"%(len(self.y_list),nb_class))
        nb_iterate = len(self.y_list) // batch_size
        assert nb_iterate>0,"文件数不能小于batch_size"
        e=Encoder(rescale = self.rescale)
        while True:
            if shuffle:
                self.x_path_list,self.y_list = self.shuffle(self.x_path_list,self.y_list)
            for i in range(nb_iterate):
                start_pos = i*batch_size
                end_pos = start_pos + batch_size
                if (nb_output == 1):
                    yield e.batch_arraylize(self.x_path_list[start_pos:end_pos],size = target_size),e.batch_onehot(self.y_list[start_pos:end_pos],nb_class = 2)
                else:
                    yield e.batch_arraylize(self.x_path_list[start_pos:end_pos],size = target_size),[e.batch_onehot(self.y_list[start_pos:end_pos], nb_class = 2)]*nb_output

            if not_break: continue
            else: break
        return
        

    def shuffle(self,x_path_list,y_list):
        indice = list(range(len(y_list)))
        shuffle_x=[None]*len(y_list)
        shuffle_y=[None]*len(y_list)
        import random
        random.shuffle(indice)
        for i,j in enumerate(indice):
            shuffle_x[i] = x_path_list[j]
            shuffle_y[i] = y_list[j]
        return shuffle_x,shuffle_y

    def get_x_path_list(self):
        return self.x_path_list
   

class Encoder(object):
    def __init__(self,rescale = 1.0):
        self.rescale=rescale

    def arraylize(self,path,size = (299,299),fovea = False):
        col,row = size
        try:
            img = Image.open(path)
            if img.mode != 'RGB': img = img.convert('RGB')
            if fovea:
                img_x,img_y = img.size
                img_x /=2
                img_y /=2
                region = (int(img_x-col/2),int(img_y-row/2), 
                            int(img_x+col/2),int(img_y+row/2))
                img = img.crop(region)
                if img.size != (col,row):
                    img = img.resize((col,row))
            else:
                img = img.resize((col,row))
            img_arr = np.array(img)
            # img_arr[:,:,0] -= 104
            # img_arr[:,:,1] -= 117
            # img_arr[:,:,2] -= 123
            img_arr = img_arr*self.rescale
        except:
            img_arr = np.zeros([size[0],size[1],3])
            print ("img broken:{}".format(path))
        return img_arr

    def onehot(self,y,nb_class):
        onehot = np.zeros(nb_class,np.int32)
        onehot[y] = 1
        return onehot

    def batch_arraylize(self,file_list,size = (299,299),channel=3, fovea = False):
        col,row = size
        batch_img_arr = np.zeros((len(file_list),row,col,channel))
        for i,path in enumerate(file_list):
            batch_img_arr[i] = self.arraylize(path,size = size,fovea=fovea)
        return batch_img_arr

    def batch_onehot(self,indice_list,nb_class):
        batch_onehot_arr = np.zeros((len(indice_list),nb_class),np.int32)
        for i,indice in enumerate(indice_list):
            batch_onehot_arr[i] = self.onehot(indice,nb_class)
        return batch_onehot_arr

class CascadeGenerator:
    def __init__(self,rescale = 1./255):
        self.rescale = rescale


    def label_generator(self,label_file_path,target_size=(299,299),
                        batch_size = 64,shuffle = False,not_break = False,
                        nb_output = 1,label_to_use = 1):
        assert os.path.isfile(label_file_path),'label文件错误%s'%(label_file_path)
        label_file=open(label_file_path)
        self.x_path_list = []
        self.y_list = []
        for line in label_file:
            print (line)
            x,l_1,l_2=line.split()
            if not os.path.exists(x):
                continue
            assert label_to_use == 1 or label_to_use == 2,"label_to_use can't be %d"%label_to_use
            if label_to_use == 1:
                self.y_list.append(int(l_1))
                self.x_path_list.append(x)
            else:
                if l_1 == 0: continue
                else:
                    self.x_path_list.append(x)
                    self.y_list.append(int(l_2))
        nb_class = len(set(self.y_list))
        print ("%d pic in label file, %d classes"%(len(self.y_list),nb_class))
        nb_iterate = len(self.y_list) // batch_size
        assert nb_iterate>0,"文件数不能小于batch_size"
        e=Encoder(rescale = self.rescale)
        while True:
            if shuffle:
                self.x_path_list,self.y_list = self.shuffle(self.x_path_list,self.y_list)
            for i in range(nb_iterate):
                start_pos = i*batch_size
                end_pos = start_pos + batch_size
                if (nb_output == 1):
                    yield e.batch_arraylize(self.x_path_list[start_pos:end_pos],size = target_size),e.batch_onehot(self.y_list[start_pos:end_pos],nb_class = 2)
                else:
                    yield e.batch_arraylize(self.x_path_list[start_pos:end_pos],size = target_size),[e.batch_onehot(self.y_list[start_pos:end_pos], nb_class = 2)]*nb_output

            if not_break: continue
            else: break
        return

    def test_generator(self,label_file_path,target_size=(299,299),
                        batch_size = 64,not_break = True,
                        nb_output = 1):
        assert os.path.isfile(label_file_path),'label文件错误%s'%(label_file_path)
        label_file=open(label_file_path)
        self.x_path_list = []
        self.l1_list = []
        self.l2_list = []
        for line in label_file:
            x,l_1,l_2=line.split()
            if not os.path.exists(x):
                continue
            self.x_path_list.append(x)
            self.l1_list.append(int(l_1))
            self.l2_list.append(int(l_2))
        #nb_class = len(set(self.l1_list))
        print ("%d pic in label file"%len(self.l1_list))
        nb_iterate = len(self.y_list) // batch_size
        assert nb_iterate>0,"文件数不能小于batch_size"
        e=Encoder(rescale = self.rescale)
        while True:
            #if shuffle:
                #self.x_path_list,self.y_list = self.shuffle(self.x_path_list,self.y_list)
            for i in range(nb_iterate):
                start_pos = i*batch_size
                end_pos = start_pos + batch_size
                if (nb_output == 1):
                    yield e.batch_arraylize(self.x_path_list[start_pos:end_pos],size = target_size),e.batch_onehot(self.l1_list[start_pos:end_pos],nb_class = 2),e.batch_onehot(self.l2_list[start_pos:end_pos],nb_class = 2)

                else:
                    yield e.batch_arraylize(self.x_path_list[start_pos:end_pos],size = target_size),[e.batch_onehot(self.l1_list[start_pos:end_pos], nb_class = 2)]*nb_output,[e.batch_onehot(self.l2_list[start_pos:end_pos], nb_class = 2)]*nb_output

            if not_break: continue
            else: break
        return
       

    def shuffle(self,x_path_list,y_list):
        indice = list(range(len(y_list)))
        shuffle_x=[None]*len(y_list)
        shuffle_y=[None]*len(y_list)
        import random
        random.shuffle(indice)
        for i,j in enumerate(indice):
            shuffle_x[i] = x_path_list[j]
            shuffle_y[i] = y_list[j]
        return shuffle_x,shuffle_y

    def get_x_path_list(self):
        return self.x_path_list
 

if __name__ == '__main__':
    pass
