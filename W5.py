def f_kdvHesapla(urun, kdv_orani):
    kdvliFiyat = (urun * kdv_orani / 100) + urun
    return kdvliFiyat

class hafta5:
    urun = 0
    kdv_orani = 0

    def __init__(self, u, k):
        self.urun = u
        self.kdv_orani = k


    def kdvHesapla (self):
        self.atmasyon = 15
        kdvliFiyat = ( self.urun * self.kdv_orani / 100 )  + self.urun
        return kdvliFiyat

    def kareAl(self, sayi):
        return sayi * sayi

    def urunSil (self):
        del self.urun


class hafta6 (hafta5):


    def kupHesapla(self, sayi):
        return super().kareAl(sayi)

gumruk = hafta6.kupHesapla(5)

elma = 40
armut = 30

hesapla = hafta5(elma, armut)
print(hesapla.atmasyon)

hesapla.kdvHesapla()
hesapla.urunSil()
print(hesapla.atmasyon)

hesapla.urun
hesapla.kdv_orani

hesapla2 = hafta5(35, 55)
hesapla2.kdvHesapla()


# kdvli_elma = (elma * 15 / 100) + elma
kdvli_elma = (elma * 15 / 100) + elma
kdvli_armut = (armut  * 15 / 100) + armut

kdvli_elma_func = f_kdvHesapla (elma, 15)
kdvli_armut_func = f_kdvHesapla (armut, 18)

kdvli_elma_class = hafta5.kdvHesapla (elma, 15)
kdvli_armut_class  = hafta5.kdvHesapla (armut, 18)
kdvli_armut_class



cay = 100

def hesap (sayi):
    print(cay)
    if sayi % 2 == 0:
        return sayi * sayi
    elif sayi % 2 == 1:
        return sayi * sayi * sayi

hesap(5)
print(cay)

def result (sayi1, sayi2):
    print(sayi2)
    return hesap(sayi1) + hesap (sayi2)

resultx = result (5, 8)
resultx = result (sayi1=5, sayi2 = 8)
resultx = result (sayi2=8, sayi1 = 5)

def goruntule (resultx):
    print(resultx)

sayi1 = 5
sayi2 = 8
goruntule(result (sayi2=sayi2, sayi1 = sayi1))
isim = goruntule("Zeki Özen")
isim




class Deneme:

    def __init__(self):
        self.__init__()

    def func1 (self, x):
        y = x + 5
        self.y = y
        return y

    def func2 (self, z):
        k = z - 5
        print(self.y)
        return k


result = Deneme.func1(10)
result


def biseyYap:
    pass


x = 5