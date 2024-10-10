x = 5
y = 10

a = 15
b = 10
c = 11

if (a + c > b):
    print("a + c > b")
else:
    print("a + c < b")


x = 20
y = 10
b = 15

if (x + y > b):
    print("x + y > b")
    print("Zeki")
elif (x + y < b):
    print("x + y < b")
    print("Neslişah")
elif (x - y > b):
    print("x - y > b")
    print("Ahmet")
else:
    print("x + y = b")
    print("İbrahim")




x = 20
y = 10
b = 15

if (x + y > b):
    print("x + y > b")
elif (x + b > y):
    print("x + b > y")



x = 20
y = 10
b = 15

if (x > y + b):
    print("x > y + b")
elif (y > x + b):
    print("y > x + b")





x = 5
x = x + 5

x = 5
x += 5


x = 5
# x = x + 1
# x +=1

# switch case ornegi
x = 5
if x == 1:
    print("Bir")
elif x == 2:
    print("İki")

# match case ornegi
x = 5
match x:
    case 1:
        print("Bir")
    case 2:
        print("İki")
    case _: # default
        print("1 ve 2 değil")


gun = "pazar"
match gun:
    case "pazartesi":
        print("hafta ici")
    case "salı":
        print("hafta ici")
    case "çarşamba":
        print("hafta ici")
    case "perşembe":
        print("hafta ici")
    case "cuma":
        print("hafta ici")
    case "cumartesi":
        print("hafta sonu")
    case "pazar":
        print("hafta sonu")
    case _:
        print("Geçersiz gün")








# 1-50 arası çift sayıların toplamını yazdırma
toplam = 0
for i in range(1, 51):
    if i % 2 == 0:
        toplam += i
print(toplam)