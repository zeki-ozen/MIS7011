from sympy import false

print("Zeki")
print("Zeki")
print("Zeki")

# Temel döngüler
# for
# while
# do-while


# aralık belirten komut
# range(3) -> 0, 1, 2
# range(3, 6) -> 3, 4, 5
# range(3, 6, 2) -> 3, 5
# range(6, 3, -1) -> 6, 5, 4

x = range(3)
list(range(1, 10)) # [1, 2, 3, 4, 5, 6, 7, 8, 9]  [1 ,10[
range (1, 10, 2) # [1, 3, 5, 7, 9]  [1, 10[
range (10, 1, 2)
list(range (10, 1, -2))
list(range (1, 10, -2))

# mod operatoru


# for dongusu, dongunın kac kere donmesi gerektigini biliyorsak kullanılır.
for i in range(10):
    print(i + 1 , ". Zeki")

# 50-150 arası sayıları ekrana yazdırma

for sayi in range (50, 151):
    print(sayi)

10 % 6 # 4
True or False and False # true
True or (False and False) # true

(False and False or True) # false
(False and False) or True # true

# explain the output of the following code

(3 and 5)
(3 or 5)
# 3 and 5


a = list()

for sayi in range (50, 151):
    if (sayi % 3 == 0):
        a.append(sayi)
        print(sayi)

b = list()
for sayi in range (51, 151, 3):
        b.append(sayi)
        print(sayi)

a == b

if (a == b):
    print("A ve B eşit")

== # is equal to

!= # is not equal to
<> # is not equal to
~= # is not equal to

a = 5
b = 3

if (a > b):
    print("a, b'den büyük")
else:
    print("b, a'dan büyük")

# ternary operator
c = "a büyük" if a > b else  "b büyük"
c

l1 = list[1, 4, 8, 12, 9]

for sayac in l1:
    print(sayac)

# whlile dongusu, dongunun kac kere donmesi gerektigini bilmiyorsak kullanılır.


while (kosul):
    # dongu icerisinde yapilacak is
    # kosulun degismesi gerekiyor

sayac = 0
while (True):
    print(sayac, "Zeki")
    sayac += 1
    if (sayac == 20):
        break

sayac = 0
while (sayac < 20):
    print(sayac, "Zeki")
    sayac += 1

sayac = 0
while (False and sayac < 20):
    print(sayac, "Zeki")
    sayac += 1


# do-while eşleniği
sayac = 1
while (true)
    işler
    işler
    işler
    if (sayac bişeyse):
        break

