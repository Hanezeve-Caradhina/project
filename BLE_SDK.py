import math
from typing import TextIO
import os
import bluetooth as bt
import time
import re

# parameters
DEBUG = True
ALT_SPD = 114514
CONGESTION_CONTROL_ENABLED = False
PAIR_PIN = 2501
disServUUID = "0000FFE0"
disCharUUID = "0000FFE1"
# disNameSPP = "HCRCVSPP\r\n"
disNameSPP = "OPPO Reno7 Pro 5G"
disNameBLE = "HCRCVBLE\r\n"
ADDR_FILE = "addr.dat"
X_OFFSET = 500


class vehicleCat:
    # This class is to define vehicle categories.
    # Default value of each vehicle should be 0.
    undefined = 0
    bike = 1
    carSmall = 2
    carBig = 3


class vehicle:
    def __init__(self, ID=0, cat=0, xVal=0, yVal=0, spd=0, isAlt=False):
        self.name = ID
        self.cat = cat
        self.xVal = xVal
        self.yVal = yVal
        self.dis = math.sqrt(xVal**2 + yVal**2)
        self.spd = spd
        self.alt = isAlt


class vehicleControlPanel:
    def __init__(self):
        # dict format: {id(int):data(vehicle)}
        self.nowDict = dict()
        self.preDict = dict()
        self.nowTick = time.time()
        self.preTick = time.time()

    def update(self, data):
        self.preTick = self.nowTick
        self.nowTick = time.time()
        self.preDict = self.nowDict
        nowDictKeys = self.nowDict.keys()
        self.nowDict.clear()
        try:
            for key in data:
                t = data[key]
                if key not in nowDictKeys:
                    self.nowDict[key] = vehicle(t.name, t.cat, t.dis, t.ang)
                else:
                    p = self.preDict[key]
                    dy = p.yVal - (p.dis * math.sin(p.ang))
                    sp = dy / (self.nowTick - self.preTick)
                    self.nowDict[key] = vehicle(t.name, t.cat, t.dis, t.ang, sp, sp > ALT_SPD)
        except Exception as e:
            print("[VEHICLE CTRL] upd error:\r\n", e)


class commControlPanel:
    def __init__(self):
        self.blePowerCTRL("ON")
        self.disAddr = self.readAddrFromFile()
        if self.disAddr == "":
            self.findDevice(114514)
        self.disPort = 1
        self.bleSocket = bt.BluetoothSocket(bt.RFCOMM)

        if CONGESTION_CONTROL_ENABLED:
            self.windowPosNow = 1
            self.windowSizeNow = 1
            self.windowThreshold = 1
            self.windowRstCnt = 0

    @staticmethod
    def blePowerCTRL(sta="ON"):
        try:
            os.system("rfkill unblock bluetooth")
            if sta == "ON":
                os.system("hciconfig hci0 up")
                os.system("bluetoothctl power on")
            else:
                os.system("bluetoothctl power off")
        except Exception as e:
            print("[POWER CTRL] ", e)
            os.system("bluetoothctl power off")

    # nearby_devices = bt.discover_devices(5, flush_cache=True, lookup_names=True)

    # print("Found {} devices.".format(len(nearby_devices)))

    # print(nearby_devices)

    # sock = bt.BluetoothSocket(bt.RFCOMM)

    def findDevice(self, times):
        while times >= 0:
            times -= 1
            devices = bt.discover_devices(duration=5, lookup_names=True, flush_cache=True, lookup_class=False)

            print("asdfghj  110")
            time.sleep(3)

            print(devices)
            for item in devices:
                if (item[1] == disNameSPP) and self.checkMacAddr(item[0]):
                    print("[COMM CTRL][DEVICE SCAN] FOUND: ", item)
                    self.disAddr = item[0]
                    self.writeAddrToFile(item[0])
                    return True
        print("[COMM CTRL][DEVICE SCAN] Fail.")
        return False

    @staticmethod
    def checkMacAddr(macAddr):
        pattern = re.compile(r"^\s*([0-9a-fA-F]{2}:){5}[0-9a-fA-F]{2}\s*$")
        if pattern.match(macAddr):
            return True
        else:
            return False

    def readAddrFromFile(self):
        addrFile = open(ADDR_FILE, "r")
        addr = addrFile.read()
        addrFile.close()
        print("[COMM CTRL][READ ADDR] " + addr)
        if self.checkMacAddr(addr):
            print("[COMM CTRL][READ ADDR] success")
            return addr
        else:
            print("[COMM CTRL][READ ADDR] fail")
            return ""

    @staticmethod
    def writeAddrToFile(addr):
        addrFile = open(ADDR_FILE, "w", encoding="utf-8")
        addrFile.write(addr)
        addrFile.close()

    def packetSend(self, cont):
        contToSend = self.contEncode(cont)
        try:
            if self.disAddr == "":
                raise Exception("MAC Address illegal: " + self.disAddr)
            if CONGESTION_CONTROL_ENABLED:
                self.bleSocket.connect((self.disAddr, self.disPort))
                self.bleSocket.send(contToSend)
                if self.windowPosNow >= self.windowSizeNow:
                    if self.windowSizeNow >= self.windowThreshold:
                        self.windowSizeNow += 1
                    else:
                        self.windowSizeNow <<= 1
                    self.windowPosNow = 0
                    self.windowRstCnt = 0
                    self.bleSocket.close()
                self.windowPosNow += 1
                # self.bleSocket.close()
                print("[COMM CTRL][PKT] PKT Send OK")
                # print(time.time())
            else:
                self.bleSocket.send(contToSend)
                print("[COMM CTRL][PKT] PKT Send OK")
        except Exception as e:
            print("[COMM CTRL][PKT] socket error:\r\n", e)
            if CONGESTION_CONTROL_ENABLED:
                self.windowThreshold = self.windowSizeNow >> 1
                self.windowRstCnt += 1
                if self.windowRstCnt >= 3:
                    self.windowSizeNow = 1
                    self.windowRstCnt = 0
                else:
                    self.windowSizeNow = self.windowThreshold
                self.bleSocket.close()
                self.bleSocket = bt.BluetoothSocket(bt.RFCOMM)
            else:
                self.bleSocket.close()
                self.bleSocket = bt.BluetoothSocket(bt.RFCOMM)
            self.bleSocket.close()
            self.bleSocket = bt.BluetoothSocket(bt.RFCOMM)
            self.bleSocket.connect((self.disAddr, self.disPort))

    @staticmethod
    def checkContentCat(t):
        if t.cat not in [0, 1, 2, 3]:
            return 1
        else:
            return 0

    def contEncode(self, cont):
        # cont is a dict from vehicleControlPanel
        ret = "-="
        ret += "t:%d;" % (len(cont))
        for key in cont:
            t = cont[key]
            if self.checkContentCat(t):
                print("[COMM CTRL][ENCODE][ERR] type error")
                ret += "%d,%d,%d,%d;" % (0, t.xVal, t.yVal, t.alt)
            else:
                ret += "%d,%d,%d,%d;" % (t.cat, t.xVal, t.yVal, t.alt)
        if DEBUG:
            print("[COMM CTRL][ENCODE] " + ret)
        return ret + "\r\n"

# disAddr = "C4:22:04:06:09:E5"
# disAddr = "04:22:04:06:09:E5"
disAddr = "04:6C:59:ba:b3:00"

if __name__ == '__main__':

    print("Hello World")

    def sendQwQtoDis():
        port = 1

        try:
            print("connect to dis")
            sock = bt.BluetoothSocket(bt.RFCOMM)
            sock.connect((disAddr, port))
            sock.send("QwQ")
            sock.close()

        except Exception as e:
            print("[SOCKET ERROR] ", e)


    tempCnt = 20

    time.sleep(3)

    commCtrl = commControlPanel()



    dicToSend = {0: vehicle(0, 1, 114, 30, 23, 1), 1: vehicle(1, 2, 233, 60, 23, 0)}

    temp = vehicle(0, 1, 114, 30, 23, 1)

    print(math.sin(45/180*math.pi))
    print(math.cos(60/180*math.pi))

    while tempCnt >= 0:
        tempCnt -= 1
        commCtrl.packetSend(dicToSend)
        # commCtrl.packetSend({"QwQ": vehicle()})
        # sendQwQtoDis()
        # print("QwQ sent")
        time.sleep(0.1)
