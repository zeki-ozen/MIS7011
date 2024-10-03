# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

total = sum(i for i in range(1, 101) if i % 7 == 0)
for i in range(1, 101):
    if i % 7 == 0:
        print(i)

x = 5
y = x + 5
print(2024)
print("Zeki")

isim = "Zeki"

x = 1
x = 11

x = 10
x = -5
print(type(x))


isim = "Zeki"

print(type(isim))

durum = True
status =  False
print(type(durum))

pi = 3.14
print(type(pi))

sanal = 3j
print(type(sanal))

a = None
if (a is None):
    print("Değer yok")

sanal = 3j
# buraya yazıilanları derleyici/yorumlayıcı çalıştırmaz
# dflgjdfkgdfgn eoıtıeroıt
a = None

# butunleşik veri tipleri
# list
# tuple
# dict
# set


#list veri tipi
# listeler [] ile tanımlanır
# listeler sıralıdır
# listeler değiştirilebilir
# listeler farklı veri tiplerini içerebilir
# listeler aynı elemanı birden fazla kez içerebilir
# listeler iç içe olabilir
# listelerin elemanları indeks ile erişilebilir
# listelerin elemanları değiştirilebilir
# listelerin elemanları silinebilir
# listelerin elemanları sıralanabilir
# listelerin elemanları aranabilir
# liste indisleri 0'dan başlar

# listelerin tanımlanması
# liste = [eleman1, eleman2, eleman3]
# liste = list(eleman1, eleman2, eleman3)
# liste = list()
# liste = []

ornekliste = [1, 2, 3, 4, 5]
print(ornekliste)

isimlistesi = ["Zeki", "İsmail", "Zülal", "Neslişah"]

isimlistesi[1]

# eksi indisler sondan başlar
isimlistesi[-1] # Neslişah

# listelerin elemanları değiştirilebilir
isimlistesi[1] = "İbrahim"
print(isimlistesi)

# listeden eleman silinmesi
# del liste[indis]
del isimlistesi[1]
print(isimlistesi)


# listenin eleman sayısını bulma
print(len(isimlistesi))

# listeye eleman ekleme
# liste.append(eleman)
isimlistesi.append("İsmail")
print(isimlistesi)

# listeye farklı veri tiplerinde eleman ekleme
isimlistesi.append(100)
print(isimlistesi)
isimlistesi.append(3.14)
print(isimlistesi)

# listeye değişken ekleme
isimlistesi.append(x)
print(isimlistesi)

# listeye liste ekleme
isimlistesi.append([8735834, 3987])
print(isimlistesi)
isimlistesi[7][0] # 8735834

# set veri tipi
# setler {} ile tanımlanır
# setler sırasızdır
# setler değiştirilebilir
# setler farklı veri tiplerini içerebilir
# setler aynı elemanı bir kez içerir
# setler iç içe olamaz
# setlerin elemanları indeks ile erişilemez
# setlerin elemanları değiştirilebilir
# setlerin elemanları silinebilir
# setlerin elemanları sıralanamaz
# setlerin elemanları aranabilir

# set tanımlama
# set = {eleman1, eleman2, eleman3}
# set = set(eleman1, eleman2, eleman3)
# set = set()
# set = {}
ornekset = {1, 2, 3, 4, 5}
print(ornekset)

ornekset2 = {1, 2, 3, "Zeki", 3.14, 5, 1}
print(ornekset2)

# setlerde elemanlar sırasızdır
# setlerde indeksleme yapılamaz
print(ornekset2[0]) # hata verir
print(ornekset2(0)) # hata verir

# setlere eleman ekleme
# set.add(eleman)
ornekset.add(6) # {1, 2, 3, 4, 5, 6}
print(ornekset)

# setlerde eleman silme
# set.remove(eleman)
ornekset.remove(6) # {1, 2, 3, 4, 5}
print(ornekset)
#ornekset.remove(1) # {2, 3, 4, 5}
# setlerde elemana ulaşma
# eleman in set
print(6 in ornekset) # False    6 elemanı sette yok
print(5 in ornekset) # True     5 elemanı sette var

# setlerde eleman sayısını bulma
print(len(ornekset)) # 5


# tuple veri tipi
# tuple () ile tanımlanır
# tuple sıralıdır
# tuple değiştirilemez
# tuple farklı veri tiplerini içerebilir
# tuple aynı elemanı birden fazla kez içerebilir
# tuple iç içe olabilir
# tuple elemanları indeks ile erişilebilir
# tuple elemanları değiştirilemez
# tuple elemanları silinemez
# tuple elemanları sıralanabilir
# tuple elemanları aranabilir
# tuple indisleri 0'dan başlar


# tuple tanımlama
# tuple = (eleman1, eleman2, eleman3)
# tuple = tuple(eleman1, eleman2, eleman3)

ornektuple = (1, 2, 3, 4, 5)
print(ornektuple)

isimtuple = ("Zeki", "İsmail", "Zülal", "Neslişah")
print(isimtuple)

# tuple elemanlarına erişme
print(isimtuple[0]) # Zeki
print(isimtuple[-1]) # Neslişah

# tuple elemanları değiştirilemez
isimtuple[1] = "İbrahim" # hata verir

# tuple elemanları silinemez
del isimtuple[1] # hata verir


# dict veri tipi
# dict {} ile tanımlanır
# dict sırasızdır
# dict değiştirilebilir
# dict farklı veri tiplerini içerebilir
# dict aynı elemanı bir kez içerir
# dict iç içe olabilir

# dict elemanları key-value şeklinde tanımlanır
# dict elemanları key ile erişilebilir
# dict elemanları değiştirilebilir
# dict elemanları silinebilir
# dict elemanları sıralanamaz


# dict tanımlama
# dict = {key1: value1, key2: value2, key3: value3}
# dict = dict(key1=value1, key2=value2, key3=value3)
# dict = dict()
# dict = {}

ornekdict = {"ad": "Zeki", "soyad": "Özen", "yas": 41}
print(ornekdict)

# dict elemanlarına erişme
print(ornekdict["soyad"]) # Özen

# dict elemanları değiştirme
ornekdict["yas"] = 42
print(ornekdict)

# dict elemanları silme
del ornekdict["yas"]
print(ornekdict)

# dict elemanlarına ekleme
ornekdict["cinsiyet"] = "Erkek"
print(ornekdict)

# 1-50 arası çift sayıları yazdır
for i in range(1, 51):
    if i % 2 == 0:
        print(i)
        
# listeye eleman ekleme
liste = []
for i in range(1, 51):
    if i % 2 == 0:
        liste.append(i)

print(liste)

# listeyi sırada yazdırma
for i in liste:
    print(i)

# listeyi tersten yazdırma
for i in liste[::-1]:
    print(i)

