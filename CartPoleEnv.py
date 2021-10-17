import numpy as np
import EasySerial


class RealEnvMake:
    slogn = ">>>CartPoleEnv->RealEnvMake<<<"
    ser = 0
    observation = np.array([0,0,0,0])
    reward = 0
    done = 0
    action = 0

    def __init__(self):
        print(self.slogn)
        self.ser = EasySerial.LinuxBackground()
        if self.ser.connect(1):
            print("Env Init Successed.")
            # self.ser.write("(-1,-1,-1)\n")
            # str, count = self.ser.read(3)
            # if count == 17 and str == "(-1,-1,-1,-1,-1)\n":
            #     print("Env Init Successed.")
            # else:
            #     print("Failed to Init Env!")
        else:
            print("Failed to Init Env!")

    def __strcheck(self,str,count):
        if count >= 11 and str.count('(') == 1 and str.count(')') == 1 and str.count(',') == 4:
            return True
        else:
            return False

    def __str2data(self,str):
        strlist = str.split(',')
        intlist = list(map(int,strlist))
        flolist = [round(intlist[0]/25000,3),round(intlist[1]/25000,3),round(intlist[2]/651.899,3),round(intlist[3]/651.899,3)]
        return np.array(flolist),intlist[4]

    def config(self):
        pass

    def reset(self):
        self.done = 1
        while(self.done):
            str, count = self.ser.read()
            if self.__strcheck(str,count):
                str = str[str.find('(')+1:str.find(')')]
                self.observation, self.done = self.__str2data(str)
            else:
                print("failed to check!")
        return self.observation

    def step(self, action):
        str = "(%d,0,0)\n" % int(action)
        self.ser.write(str)
        str, count = self.ser.read()
        if self.__strcheck(str, count):
            str = str[str.find('(') + 1:str.find(')')]
            self.observation, self.done = self.__str2data(str)
            if(self.done):
                self.reward = 1
            else:
                self.reward = 1
        else:
            print("failed to check!")
        return self.observation,self.reward,self.done

    def end(self):
        self.ser.close()
